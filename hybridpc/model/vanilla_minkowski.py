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
from hybridpc.model.module import Backbone, MinkUNetBackbone, ImplicitDecoder, Dense_Generator
from hybridpc.model.general_model import GeneralModel
from hybridpc.evaluation.semantic_segmentation import *
from torch.nn import functional as F

class VanillaMinkowski(GeneralModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters(cfg)
        # Shared latent code across both decoders
        # set automatic_optimization to False
        # self.automatic_optimization = False
        self.feature_in = cfg.model.network.feature_in # random_latent/ voxel_features
        self.use_decoder = cfg.model.network.use_decoder # whether to use decoder
        if self.use_decoder:
            output_channel = cfg.model.network.modulation_dim
        else:
            output_channel = cfg.data.classes # use the number of classes as the output channel
        if self.feature_in == "random_latent":
            input_channel = cfg.model.network.latent_dim
        else: 
            input_channel = cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3
        self.backbone = MinkUNetBackbone(
            input_channel,
            output_channel
        )
        self.seg_decoder = ImplicitDecoder(
            "seg",
            cfg.model.network.modulation_dim,
            cfg.model.network.seg_decoder.input_dim,
            cfg.model.network.seg_decoder.hidden_dim,
            cfg.model.network.seg_decoder.num_hidden_layers_before_skip,
            cfg.model.network.seg_decoder.num_hidden_layers_after_skip,
            cfg.data.classes
        )
        self.val_test_step_outputs = []


    def forward(self, data_dict, latent_code):
        if self.feature_in == "random_latent":
            backbone_output_dict = self.backbone(
            latent_code, data_dict["voxel_coords"], data_dict["voxel_indices"]
        )
        else:
            backbone_output_dict = self.backbone(
                data_dict["voxel_features"], data_dict["voxel_coords"], data_dict["voxel_indices"]
            )

        if self.use_decoder:
            backbone_output_dict['semantic_scores'] = self.seg_decoder(backbone_output_dict['voxel_features'].F, data_dict['points'], data_dict["voxel_indices"])  # embeddings (B, C) coords (N, 3) indices (N, )
        else:
            backbone_output_dict['semantic_scores'] = backbone_output_dict['point_features'] # use the point features as the semantic scores
        return backbone_output_dict

    def _loss(self, data_dict, output_dict):
        seg_loss = F.cross_entropy(output_dict['semantic_scores'], data_dict['labels'].long(), ignore_index=-1)

        return seg_loss

    def configure_optimizers(self):
        outer_optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.model.optimizer.lr)

        return outer_optimizer

    
    def training_step(self, data_dict):
        voxel_num = data_dict["voxel_coords"].shape[0] # B voxels
        latent_code = torch.ones(voxel_num, self.hparams.model.network.latent_dim, requires_grad=True, device=self.device)
        output_dict = self.forward(data_dict, latent_code)
        seg_loss = self._loss(data_dict, output_dict)

        self.log("train/seg_loss", seg_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.data.batch_size)
        semantic_predictions = torch.argmax(output_dict['semantic_scores'], dim=-1)  # (B, N)
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["labels"], ignore_label=-1)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["labels"], ignore_label=-1)
        self.log(
            "train/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.log(
            "train/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )


        return seg_loss

    def on_train_epoch_end(self):
        # Update the learning rates for both optimizers
        cosine_lr_decay(
            self.trainer.optimizers[0], self.hparams.model.optimizer.lr, self.current_epoch,
            self.hparams.model.lr_decay.decay_start_epoch, self.hparams.model.trainer.max_epochs, 1e-6
        )

    def validation_step(self, data_dict, idx):
        voxel_num = data_dict["voxel_coords"].shape[0] # B voxels
        latent_code = torch.ones(voxel_num, self.hparams.model.network.latent_dim, requires_grad=True, device=self.device)
        output_dict = self.forward(data_dict, latent_code)
        seg_loss = self._loss(data_dict, output_dict)

        self.log("val/seg_loss", seg_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

      # Calculating the metrics
        semantic_predictions = torch.argmax(output_dict['semantic_scores'], dim=-1)  # (B, N)
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["labels"], ignore_label=-1)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["labels"], ignore_label=-1)
        self.log(
            "val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.log(
            "val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.val_test_step_outputs.append((semantic_accuracy, semantic_mean_iou))


