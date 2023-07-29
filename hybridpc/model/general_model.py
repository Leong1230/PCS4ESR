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
        self.val_test_step_outputs = []

    
    def training_step(self, data_dict):
        pass 

    def on_train_epoch_end(self):
        cosine_lr_decay(
            self.trainer.optimizers[0], self.hparams.model.optimizer.lr, self.current_epoch,
            self.hparams.model.lr_decay.decay_start_epoch, self.hparams.model.trainer.max_epochs, 1e-6
        )

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



