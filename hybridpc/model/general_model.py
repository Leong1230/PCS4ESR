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
from hybridpc.evaluation.semantic_segmentation import *
from torch.nn import functional as F

class GeneralModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        # Shared latent code across both decoders
        # set automatic_optimization to False
    #     self.automatic_optimization = False
        
    #     self.latent_dim = cfg.model.network.latent_dim
    #     self.seg_loss_weight = cfg.model.network.seg_loss_weight

    #     self.functa_backbone = Backbone(
    #         backbone_type=cfg.model.network.backbone_type,
    #         input_channel=self.latent_dim, output_channel=cfg.model.network.modulation_dim, block_channels=cfg.model.network.blocks,
    #         block_reps=cfg.model.network.block_reps,
    #         sem_classes=cfg.data.classes
    #     )
    #     self.seg_backbone = Backbone(
    #         backbone_type=cfg.model.network.backbone_type,
    #         input_channel=self.latent_dim, output_channel=cfg.model.network.modulation_dim, block_channels=cfg.model.network.blocks,
    #         block_reps=cfg.model.network.block_reps,
    #         sem_classes=cfg.data.classes
    #     )

    #     self.seg_decoder = ImplicitDecoder(
    #         "seg",
    #         cfg.model.network.modulation_dim,
    #         cfg.model.network.seg_decoder.input_dim,
    #         cfg.model.network.seg_decoder.hidden_dim,
    #         cfg.model.network.seg_decoder.num_hidden_layers_before_skip,
    #         cfg.model.network.seg_decoder.num_hidden_layers_after_skip,
    #         cfg.data.classes
    #     )
    #     self.functa_decoder = ImplicitDecoder(
    #         "functa",
    #         cfg.model.network.modulation_dim,
    #         cfg.model.network.functa_decoder.input_dim,
    #         cfg.model.network.functa_decoder.hidden_dim,
    #         cfg.model.network.functa_decoder.num_hidden_layers_before_skip,
    #         cfg.model.network.functa_decoder.num_hidden_layers_after_skip,
    #         1
    #     )

    #     self.dense_generator = Dense_Generator(
    #         self.functa_decoder,
    #         cfg.data.voxel_size,
    #         cfg.model.dense_generator.num_steps,
    #         cfg.model.dense_generator.num_points,
    #         cfg.model.dense_generator.threshold,
    #         cfg.model.dense_generator.filter_val
    #     )
    # #   # The inner loop optimizer is applied to the latent code
    # #     self.inner_optimizer = torch.optim.SGD([p for p in self.parameters()], lr=cfg.model.optimizer.inner_loop_lr)
    # #     # The outer loop optimizer is applied to the model parameters
        self.val_test_step_outputs = []



    # def forward(self, data_dict, latent_code):
        
    #     seg_modulations = self.seg_backbone(latent_code, data_dict['voxel_coords']) # B, C
    #     functa_modulations = self.functa_backbone(latent_code, data_dict['voxel_coords']) # B, C

    #     # modulations = latent_code
    #     seg_scores = self.seg_decoder(seg_modulations.F, data_dict['points'], data_dict["voxel_indices"])  # embeddings (B, C) coords (N, 3) indices (N, )
    #     udf_values = self.functa_decoder(functa_modulations.F, data_dict['query_points'], data_dict["query_voxel_indices"]) # embeddings (B, D1) coords (M, 3) indices (M, )

        # return {"seg_scores": seg_scores, "udf_values": udf_values}

    # def _loss(self, data_dict, output_dict):
    #     seg_loss = F.cross_entropy(output_dict['seg_scores'], data_dict['labels'].long(), ignore_index=-1)

    #     udf_loss = F.mse_loss(output_dict['udf_values'], data_dict['udf_values'])

    #     # loss_i = torch.nn.L1Loss(reduction='none')(torch.clamp(output_dict['values'], max=self.hparams.cfg.data.udf_queries.max_dist),torch.clamp(data_dict['values'], max=self.hparams.cfg.data.udf_queries.max_dist))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
    #     # value_loss = loss_i.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

    #     return seg_loss, udf_loss, self.seg_loss_weight * seg_loss + (1 - self.seg_loss_weight) * udf_loss

    # # def configure_optimizers(self):
    # #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
    # #     return optimizer

    # def configure_optimizers(self):
    #     outer_optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.cfg.model.optimizer.lr)

    #     return outer_optimizer

    
    def training_step(self, data_dict):
        pass 

    def on_train_epoch_end(self):
        pass

    def validation_step(self, data_dict, idx):
        pass

    def on_validation_epoch_end(self):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.model.network.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            all_gt_insts_bbox = []
            all_sem_acc = []
            all_sem_miou = []
            for semantic_accuracy, semantic_mean_iou in self.val_test_step_outputs:
                all_sem_acc.append(semantic_accuracy)
                all_sem_miou.append(semantic_mean_iou)
            self.val_test_step_outputs.clear()

            self.print("Evaluating semantic segmentation ...")

            sem_miou_avg = np.mean(np.array(all_sem_miou))
            sem_acc_avg = np.mean(np.array(all_sem_acc))
            self.print(f"Semantic Accuracy: {sem_acc_avg}")
            self.print(f"Semantic mean IoU: {sem_miou_avg}")

            if self.hparams.model.inference.save_predictions:
                save_dir = os.path.join(
                    self.hparams.exp_output_root_path, 'inference', self.hparams.model.inference.split,
                    'predictions'
                )
                # save_prediction(
                #     save_dir, all_pred_insts, self.hparams.cfg.data.mapping_classes_ids,
                #     self.hparams.cfg.data.ignore_classes
                # )
                self.print(f"\nPredictions saved at {os.path.abspath(save_dir)}")

            # if self.hparams.model.inference.visualize_udf:
            #     self.udf_visualization(data_dict, output_dict, latent_code, self.current_epoch, udf_loss)



