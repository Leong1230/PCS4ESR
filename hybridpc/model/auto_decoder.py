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
from module import Backbone, ImplicitDecoder
from torch.nn import functional as F

class AutoDecoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        # Shared latent code across both decoders
        self.latent_code = torch.nn.Parameter(torch.randn(cfg.data.latent_dim))
        self.loss_weight = cfg.model.loss_weight

        self.seg_decoder = ImplicitDecoder(
            cfg.model.latent_dim,
            cfg.model.seg_decoder.input_dim,
            cfg.model.seg_decoder.hidden_dim,
            cfg.model.seg_decoder.num_hidden_layers_before_skip,
            cfg.model.seg_decoder.num_hidden_layers_after_skip,
            cfg.data.category_num,
        )
        self.functa_decoder = ImplicitDecoder(
            cfg.model.latent_dim,
            cfg.hparams.model.functa_decoder.input_dim,
            cfg.hparams.model.functa_decoder.hidden_dim,
            cfg.hparams.model.functa_decoder.num_hidden_layers_before_skip,
            cfg.hparams.model.functa_decoder.num_hidden_layers_after_skip,
            1,
        )

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.cfg.model.optimizer, params=self.parameters())

    def forward(self, data_dict):
        points = torch.cat([data_dict['points'], self.latent_code.repeat(data_dict['points'].size(0), 1, 1)], dim=-1)
        query_points = torch.cat([data_dict['query_points'], self.latent_code.repeat(data_dict['query_points'].size(0), 1, 1)], dim=-1)
        
        segmentation = self.seg_decoder(points)
        values = self.functa_decoder(query_points)

        return {"segmentation": segmentation, "values": values}

    def _loss(self, data_dict, output_dict):
        seg_loss = F.cross_entropy(output_dict['segmentation'], data_dict['labels'])
        value_loss = F.mse_loss(output_dict['values'], data_dict['values'])

        return seg_loss, value_loss, self.loss_weight * seg_loss + (1 - self.loss_weight) * value_loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
    #     return optimizer

    def training_step(self, data_dict, idx):
        output_dict = self.forward(data_dict)
        seg_loss, value_loss, loss = self._loss(data_dict, output_dict)
        self.log("train/seg_loss", seg_loss, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.hparams.cfg.data.batch_size)
        self.log("train/value_loss", value_loss, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.hparams.cfg.data.batch_size)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.hparams.cfg.data.batch_size)
        return loss
    

    def validation_step(self, data_dict, idx):
        output_dict = self.forward(data_dict)
        loss = self._loss(data_dict, output_dict)
        
        # Convert the predicted segmentations to discrete labels
        pred_labels = torch.argmax(output_dict['segmentation'], dim=1)
        iou = jaccard_score(data_dict['labels'].cpu().numpy(), pred_labels.cpu().numpy(), average=None)
        acc = (pred_labels == data_dict['labels']).float().mean()

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        return {"val_loss": loss, "iou": iou, "acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_iou = np.mean([x['iou'] for x in outputs])
        class_iou = np.mean([x['iou'] for x in outputs], axis=0)
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        
        self.log("avg_val_loss", avg_loss, prog_bar=True)
        self.log("avg_val_iou", avg_iou, prog_bar=True)
        self.log("class_val_iou", class_iou, prog_bar=True)
        self.log("avg_val_acc", avg_acc, prog_bar=True)