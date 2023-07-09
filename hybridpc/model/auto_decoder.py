from pyexpat import features
from sklearn.metrics import jaccard_score
import torch
import time
import os
import torch.nn as nn
import numpy as np
import math
import torchmetrics
import pytorch_lightning as pl
import hydra
from hybridpc.optimizer.optimizer import cosine_lr_decay
from hybridpc.model.module import Backbone, ImplicitDecoder
from torch.nn import functional as F

class AutoDecoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        # Shared latent code across both decoders
        # set automatic_optimization to False
        self.automatic_optimization = False
        
        self.latent_dim = cfg.model.network.latent_dim
        self.seg_loss_weight = cfg.model.network.seg_loss_weight

        self.backbone = Backbone(
            input_channel=self.latent_dim, output_channel=cfg.model.network.modulation_dim, block_channels=cfg.model.network.blocks,
            block_reps=cfg.model.network.block_reps
        )

        self.seg_decoder = ImplicitDecoder(
            cfg.model.network.modulation_dim,
            cfg.model.network.seg_decoder.input_dim,
            cfg.model.network.seg_decoder.hidden_dim,
            cfg.model.network.seg_decoder.num_hidden_layers_before_skip,
            cfg.model.network.seg_decoder.num_hidden_layers_after_skip,
            cfg.data.category_num,
        )
        self.functa_decoder = ImplicitDecoder(
            cfg.model.network.modulation_dim,
            cfg.model.network.functa_decoder.input_dim,
            cfg.model.network.functa_decoder.hidden_dim,
            cfg.model.network.functa_decoder.num_hidden_layers_before_skip,
            cfg.model.network.functa_decoder.num_hidden_layers_after_skip,
            1,
        )
      # The inner loop optimizer is applied to the latent code
        self.inner_optimizer = torch.optim.SGD([p for p in self.parameters()], lr=cfg.model.optimizer.inner_loop_lr)



    def forward(self, data_dict, latent_code):
        # points = torch.cat([data_dict['points'], self.latent_code.repeat(data_dict['points'].size(0), 1, 1)], dim=-1)
        # query_points = torch.cat([data_dict['query_points'], self.latent_code.repeat(data_dict['query_points'].size(0), 1, 1)], dim=-1)
        # points = torch.cat([data_dict['points'], self.latent_code.repeat(data_dict['points'].size(0), 1)], dim=-1)
        # query_points = torch.cat([data_dict['query_points'], self.latent_code.repeat(data_dict['query_points'].size(0), 1)], dim=-1)
        
        modulations = self.backbone(latent_code, data_dict['voxel_coords']) # B, C
        segmentation = self.seg_decoder(modulations, data_dict['points'], data_dict["voxel_indices"])  # embeddings (B, C) coords (N, 3) indices (N, )
        values = self.functa_decoder(modulations, data_dict['query_points'], data_dict["query_voxel_indices"]) # embeddings (B, D1) coords (M, 3) indices (M, )

        return {"segmentation": segmentation, "values": values}

    def _loss(self, data_dict, output_dict):
        # seg_loss = F.cross_entropy(output_dict['segmentation'].view(-1, self.hparams.cfg.data.category_num), data_dict['labels'].view(-1))
        seg_loss = F.cross_entropy(output_dict['segmentation'], data_dict['labels'])

        value_loss = F.mse_loss(output_dict['values'], data_dict['values'])

        return seg_loss, value_loss, self.seg_loss_weight * seg_loss + (1 - self.seg_loss_weight) * value_loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
    #     return optimizer

    def configure_optimizers(self):
        dummy_param = torch.zeros((1,), requires_grad=True, device=self.device)  # Dummy tensor
        latent_optimizer = torch.optim.SGD([dummy_param], lr=self.hparams.cfg.model.optimizer.inner_loop_lr)
        outer_optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.cfg.model.optimizer.lr)
        return outer_optimizer

    
    def training_step(self, data_dict):
        batch_size = data_dict["voxel_coords"].shape[0] # B voxels
        latent_code = torch.zeros(batch_size, self.latent_dim, requires_grad=True, device=self.device)

        # Creating the optimizer for latent_code
        outer_optimizer = self.optimizers()
        latent_optimizer = torch.optim.SGD([latent_code], lr=self.hparams.cfg.model.optimizer.inner_loop_lr)

        # Inner loop
        for _ in range(self.hparams.cfg.model.optimizer.inner_loop_steps):  # Perform multiple steps
            output_dict = self.forward(data_dict, latent_code)
            _, value_loss, _ = self._loss(data_dict, output_dict)
            self.manual_backward(value_loss)
            
            # Step the latent optimizer and zero its gradients
            latent_optimizer.step()
            latent_optimizer.zero_grad()

        self.log("train/value_loss", value_loss, on_step=True, on_epoch=False)

        # Outer loop
        output_dict = self.forward(data_dict, latent_code)
        _, value_loss, total_loss = self._loss(data_dict, output_dict)
        self.manual_backward(total_loss)
        outer_optimizer.step()
        outer_optimizer.zero_grad()

        self.log("train/value_loss", value_loss, on_step=True, on_epoch=False)
        self.log("train/loss", total_loss, on_step=True, on_epoch=False)

        return total_loss

    

    def on_train_epoch_end(self):
        cosine_lr_decay(
            self.trainer.optimizers[0], self.hparams.cfg.model.optimizer.lr, self.current_epoch,
            self.hparams.cfg.model.lr_decay.decay_start_epoch, self.hparams.cfg.model.trainer.max_epochs, 1e-6
        )

    def validation_step(self, data_dict, idx):
        torch.set_grad_enabled(True)
        batch_size = data_dict["voxel_coords"].shape[0]  # B voxels
        latent_code = torch.zeros(batch_size, self.latent_dim, requires_grad=True, device=self.device)

        # Creating the optimizer for latent_code
        latent_optimizer = torch.optim.SGD([latent_code], lr=self.hparams.cfg.model.optimizer.inner_loop_lr)

        # Inner loop
        for _ in range(self.hparams.cfg.model.optimizer.inner_loop_steps):  # Perform multiple steps
            output_dict = self.forward(data_dict, latent_code)
            _, value_loss, _ = self._loss(data_dict, output_dict)
            self.manual_backward(value_loss)

            # Step the latent optimizer and zero its gradients
            latent_optimizer.step()
            latent_optimizer.zero_grad()

        self.log("val/value_loss", value_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # No outer loop for validation

        # Calculating the metrics
        pred_labels = torch.argmax(output_dict['segmentation'], dim=-1)  # (B, N)
        iou = jaccard_score(data_dict['labels'].view(-1).cpu().numpy(), pred_labels.view(-1).cpu().numpy(), average=None).mean()
        acc = (pred_labels == data_dict['labels']).float().mean()

        self.log("val/iou", iou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


    # def on_validation_epoch_end(self):
    #     if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
    #         avg_seg_loss = torch.stack([self.trainer.callback_metrics['val/seg_loss']]).mean()
    #         avg_value_loss = torch.stack([self.trainer.callback_metrics['val/value_loss']]).mean()
    #         avg_total_loss = torch.stack([self.trainer.callback_metrics['val/total_loss']]).mean()
    #         avg_iou = torch.stack([self.trainer.callback_metrics['val/iou']]).mean()
    #         avg_acc = torch.stack([self.trainer.callback_metrics['val/acc']]).mean()
            
    #         self.log("avg_val_seg_loss", avg_seg_loss, prog_bar=True)
    #         self.log("avg_val_value_loss", avg_value_loss, prog_bar=True)
    #         self.log("avg_val_total_loss", avg_total_loss, prog_bar=True)
    #         self.log("avg_val_iou", avg_iou, prog_bar=True)
    #         self.log("avg_val_acc", avg_acc, prog_bar=True)
