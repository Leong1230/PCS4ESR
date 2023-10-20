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
import pl_bolts
import hydra
from hybridpc.optimizer.optimizer import cosine_lr_decay
from hybridpc.model.module import Backbone, ImplicitDecoder, Dense_Generator
from hybridpc.evaluation.semantic_segmentation import *
from torch.nn import functional as F

class GeneralModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.training_stage = cfg.model.training_stage
        self.val_test_step_outputs = []
        self.best_metric_value = -np.inf
        self.metric = torchmetrics.ConfusionMatrix(
            task = 'multiclass',
            num_classes=cfg.data.classes
        )


    def configure_optimizers(self):
        if self.training_stage == 1:
            params_to_optimize = list(self.encoder.parameters()) + list(self.udf_decoder.parameters())
        else :
            params_to_optimize = list(self.seg_backbone.parameters()) + list(self.seg_decoder.parameters())
        
        if self.hparams.model.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                momentum=0.9,
                weight_decay=1e-4,
            )
            scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=int(self.hparams.model.optimizer.warmup_steps_ratio * self.hparams.model.trainer.max_steps),
                max_epochs=self.hparams.model.trainer.max_steps,
                eta_min=0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }

        elif self.hparams.model.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-4
            )
            return optimizer

        else:
            logging.error('Optimizer type not supported')

    
    def training_step(self, data_dict):
        pass 

    def on_train_epoch_end(self):
        if self.hparams.model.optimizer.name == 'Adam':
            # Update the learning rates for Adam optimizers
            cosine_lr_decay(
                self.trainer.optimizers[0], self.hparams.model.optimizer.lr, self.current_epoch,
                self.hparams.model.lr_decay.decay_start_epoch, self.hparams.model.trainer.max_epochs, 1e-6
            )
    def validation_step(self, data_dict, idx):
        pass

    def on_validation_epoch_end(self):
        # evaluate instance predictions
        if self.training_stage !=1:
            if self.current_epoch > self.hparams.model.network.prepare_epochs:
                confusion_matrix = self.metric.compute().cpu().numpy()
                self.metric.reset()
                ious = per_class_iou(confusion_matrix) * 100
                accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100
                miou = np.nanmean(ious)
                macc = np.nanmean(accs)
                def compare(prev, cur):
                    return prev < cur 
                
                if compare(self.best_metric_value, miou):
                    self.best_metric_value = miou
                self.log("val_best_mIoU", self.best_metric_value, logger=True)
                self.log("val_mIoU", miou, logger=True)
                self.log("val_mAcc", macc, logger=True)
                self.print(f"mAcc: {macc}")
                self.print(f"mIoU: {miou}")

                # scene level averaged metrics
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


                # if self.hparams.model.inference.save_predictions:
                #     save_dir = os.path.join(
                #         self.hparams.exp_output_root_path, 'inference', self.hparams.model.inference.split,
                #         'predictions'
                #     )
                #     # save_prediction(
                #     #     save_dir, all_pred_insts, self.hparams.cfg.data.mapping_classes_ids,
                #     #     self.hparams.cfg.data.ignore_classes
                #     # )
                #     self.print(f"\nPredictions saved at {os.path.abspath(save_dir)}")


    def test_step(self, data_dict, idx):
        pass

    # def on_test_epoch_end(self):
    #     # evaluate instance predictions
    #     if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
    #         all_pred_insts = []
    #         all_gt_insts = []
    #         all_gt_insts_bbox = []
    #         all_sem_acc = []
    #         all_sem_miou = []
    #         for semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox in self.val_test_step_outputs:
    #             all_sem_acc.append(semantic_accuracy)
    #             all_sem_miou.append(semantic_mean_iou)
    #             all_gt_insts_bbox.append(gt_instances_bbox)
    #             all_gt_insts.append(gt_instances)
    #             all_pred_insts.append(pred_instances)

    #         self.val_test_step_outputs.clear()

    #         if self.hparams.cfg.model.inference.evaluate:
    #             inst_seg_evaluator = GeneralDatasetEvaluator(
    #                 self.hparams.cfg.data.class_names, -1, self.hparams.cfg.data.ignore_classes
    #             )
    #             self.print("Evaluating instance segmentation ...")
    #             inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=True)
    #             obj_detect_eval_result = evaluate_bbox_acc(
    #                 all_pred_insts, all_gt_insts_bbox,
    #                 self.hparams.cfg.data.class_names, self.hparams.cfg.data.ignore_classes, print_result=True
    #             )

    #             sem_miou_avg = np.mean(np.array(all_sem_miou))
    #             sem_acc_avg = np.mean(np.array(all_sem_acc))
    #             self.print(f"Semantic Accuracy: {sem_acc_avg}")
    #             self.print(f"Semantic mean IoU: {sem_miou_avg}")

    #         if self.hparams.cfg.model.inference.save_predictions:
    #             save_dir = os.path.join(
    #                 self.hparams.cfg.exp_output_root_path, 'inference', self.hparams.cfg.model.inference.split,
    #                 'predictions'
    #             )
    #             save_prediction(
    #                 save_dir, all_pred_insts, self.hparams.cfg.data.mapping_classes_ids,
    #                 self.hparams.cfg.data.ignore_classes
    #             )
    #             self.print(f"\nPredictions saved at {os.path.abspath(save_dir)}")



