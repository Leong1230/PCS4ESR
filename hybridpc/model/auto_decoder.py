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
        self.latent_code = nn.Parameter(torch.randn(cfg.model.network.latent_dim))
        self.seg_loss_weight = cfg.model.network.seg_loss_weight

        self.seg_decoder = ImplicitDecoder(
            cfg.model.network.latent_dim,
            cfg.model.network.seg_decoder.input_dim,
            cfg.model.network.seg_decoder.hidden_dim,
            cfg.model.network.seg_decoder.num_hidden_layers_before_skip,
            cfg.model.network.seg_decoder.num_hidden_layers_after_skip,
            cfg.data.voxel_category_num,
        )
        self.functa_decoder = ImplicitDecoder(
            cfg.model.network.latent_dim,
            cfg.model.network.functa_decoder.input_dim,
            cfg.model.network.functa_decoder.hidden_dim,
            cfg.model.network.functa_decoder.num_hidden_layers_before_skip,
            cfg.model.network.functa_decoder.num_hidden_layers_after_skip,
            1,
        )
      # The inner loop optimizer is applied to the latent code
        self.inner_optimizer = torch.optim.SGD([self.latent_code], lr=cfg.inner_loop_lr)

    def configure_optimizers(self):
        # The outer loop optimizer is applied to all parameters except the latent code
        outer_optimizer = torch.optim.SGD([p for p in self.parameters() if p is not self.latent_code], 
                                          lr=self.hparams.cfg.model.optimizer.lr)
        return [self.inner_optimizer, outer_optimizer]

    def forward(self, data_dict):
        # points = torch.cat([data_dict['points'], self.latent_code.repeat(data_dict['points'].size(0), 1, 1)], dim=-1)
        # query_points = torch.cat([data_dict['query_points'], self.latent_code.repeat(data_dict['query_points'].size(0), 1, 1)], dim=-1)
        # points = torch.cat([data_dict['points'], self.latent_code.repeat(data_dict['points'].size(0), 1)], dim=-1)
        # query_points = torch.cat([data_dict['query_points'], self.latent_code.repeat(data_dict['query_points'].size(0), 1)], dim=-1)
        
        segmentation = self.seg_decoder(self.latent_code.repeat(data_dict['points'].size(0), 1), data_dict['points'])  # embeddings (B, D1) coords (B, N, D2)
        values = self.functa_decoder(self.latent_code.repeat(data_dict['query_points'].size(0), 1), data_dict['query_points']) # embeddings (B, D1) coords (B, M, D2)

        return {"segmentation": segmentation, "values": values}

    def _loss(self, data_dict, output_dict):
        seg_loss = F.cross_entropy(output_dict['segmentation'].view(-1, self.hparams.cfg.data.voxel_category_num), data_dict['labels'].view(-1))
        value_loss = F.mse_loss(output_dict['values'], data_dict['values'])

        return seg_loss, value_loss, self.seg_loss_weight * seg_loss + (1 - self.seg_loss_weight) * value_loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
    #     return optimizer

    def training_step(self, data_dict, idx, optimizer_idx):
        output_dict = self.forward(data_dict)

        # Inner loop
        if optimizer_idx == 0:
            seg_loss, _, _ = self._loss(data_dict, output_dict)
            self.log("train/seg_loss", seg_loss, on_step=True, on_epoch=False)
            return seg_loss
        
        # Outer loop
        elif optimizer_idx == 1:
            _, value_loss, total_loss = self._loss(data_dict, output_dict)
            self.log("train/value_loss", value_loss, on_step=True, on_epoch=False)
            self.log("train/loss", total_loss, on_step=True, on_epoch=False)
            return total_loss
    

    def on_train_epoch_end(self):
        cosine_lr_decay(
            self.trainer.optimizers[0], self.hparams.cfg.model.optimizer.lr, self.current_epoch,
            self.hparams.cfg.model.lr_decay.decay_start_epoch, self.hparams.cfg.model.trainer.max_epochs, 1e-6
        )

    def validation_step(self, data_dict, idx):
        output_dict = self.forward(data_dict)
        seg_loss, value_loss, total_loss = self._loss(data_dict, output_dict)
        
        # Convert the predicted segmentations to discrete labels
        pred_labels = torch.argmax(output_dict['segmentation'], dim=-1) # (B, N)
        iou = jaccard_score(data_dict['labels'].view(-1).cpu().numpy(), pred_labels.view(-1).cpu().numpy(), average=None).mean()
        acc = (pred_labels == data_dict['labels']).float().mean()

        self.log("val/seg_loss", seg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/value_loss", value_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/iou", iou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
            avg_seg_loss = torch.stack([self.trainer.callback_metrics['val/seg_loss']]).mean()
            avg_value_loss = torch.stack([self.trainer.callback_metrics['val/value_loss']]).mean()
            avg_total_loss = torch.stack([self.trainer.callback_metrics['val/total_loss']]).mean()
            avg_iou = torch.stack([self.trainer.callback_metrics['val/iou']]).mean()
            avg_acc = torch.stack([self.trainer.callback_metrics['val/acc']]).mean()
            
            self.log("avg_val_seg_loss", avg_seg_loss, prog_bar=True)
            self.log("avg_val_value_loss", avg_value_loss, prog_bar=True)
            self.log("avg_val_total_loss", avg_total_loss, prog_bar=True)
            self.log("avg_val_iou", avg_iou, prog_bar=True)
            self.log("avg_val_acc", avg_acc, prog_bar=True)
