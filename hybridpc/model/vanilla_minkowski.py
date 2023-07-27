from pyexpat import features
from sklearn.metrics import jaccard_score
import torch
import time
import os
import torch.nn as nn
import numpy as np
import math
import torchmetrics
import open3d as o3d
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import hydra
from hybridpc.optimizer.optimizer import cosine_lr_decay
from hybridpc.model.module import Backbone, ImplicitDecoder, Dense_Generator
from hybridpc.model.general_model import GeneralModel
from hybridpc.evaluation.semantic_segmentation import *
from torch.nn import functional as F

class Vanilla_Minkowski(GeneralModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters()
        # Shared latent code across both decoders
        # set automatic_optimization to False
        self.automatic_optimization = False

        self.backbone = Backbone(
            backbone_type=cfg.model.network.backbone_type,
            input_channel=self.latent_dim, output_channel=cfg.model.network.modulation_dim, block_channels=cfg.model.network.blocks,
            block_reps=cfg.model.network.block_reps,
            sem_classes=cfg.data.classes
        )
        self.val_test_step_outputs = []


    def forward(self, data_dict):
        backbone_output_dict = self.backbone(
            data_dict["voxel_features"], data_dict["voxe_coords"], data_dict["voxel_indices"]
        )
        return backbone_output_dict

    def _loss(self, data_dict, output_dict):
        seg_loss = F.cross_entropy(output_dict['semantic_scores'], data_dict['labels'].long(), ignore_index=-1)

        return seg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer

    def configure_optimizers(self):
        outer_optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.cfg.model.optimizer.lr)

        return outer_optimizer

    
    def training_step(self, data_dict):

        output_dict = self.forward(data_dict)
        seg_loss = self._loss(data_dict, output_dict)

        self.log("train/seg_loss", seg_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.data.batch_size)

        return seg_loss

    

    def on_train_epoch_end(self):
        cosine_lr_decay(
            self.trainer.optimizers[0], self.hparams.cfg.model.optimizer.lr, self.current_epoch,
            self.hparams.cfg.model.lr_decay.decay_start_epoch, self.hparams.cfg.model.trainer.max_epochs, 1e-6
        )

    def validation_step(self, data_dict, idx):
        output_dict = self.forward(data_dict)
        seg_loss = self._loss(data_dict, output_dict)

        self.log("val/seg_loss", seg_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

        # After the loop, plot the value_loss for each step
        # plt.plot(value_loss_list)
        # plt.xlabel('Step')
        # plt.ylabel('Value Loss')
        # plt.show()
        # No outer loop for validation

        # Calculating the metrics
        semantic_predictions = torch.argmax(output_dict['seg_loss'], dim=-1)  # (B, N)
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["labels"], ignore_label=-1)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["labels"], ignore_label=-1)
        self.log(
            "val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.log(
            "val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )


