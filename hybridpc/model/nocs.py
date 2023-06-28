from pyexpat import features
import torch
import time
import os
import torch.nn as nn
import numpy as np
import math
import torchmetrics
import pytorch_lightning as pl
import pyransac3d as pyrsc
from pytorch3d.ops import corresponding_points_alignment
from hybridpc.evaluation.obb_prediction import GeneralDatasetEvaluator
from hybridpc.optimizer import init_optimizer, cosine_lr_decay
from hybridpc.common_ops.functions import common_ops
from hybridpc.util import save_prediction, save_gt
from hybridpc.model.module import Backbone
from hybridpc.model.module import Backbone_NOCS
from hybridpc.evaluation.visualization import *


class NOCS(pl.LightningModule):
    def __init__(self, model, data, optimizer, lr_decay, inference=None):
        super().__init__()
        self.save_hyperparameters()
        self.voxel_size = model.voxel_size
        input_channel = model.use_coord * 3 + model.use_color * 3 + model.use_normal * 3 + model.use_multiview * 128
        self.backbone = Backbone_NOCS(input_channel=input_channel,
                                 output_channel=model.m,
                                 block_channels=model.blocks,
                                 block_reps=model.block_reps,
                                 sem_classes=data.classes)
        # if self.current_epoch > model.prepare_epochs and model.freeze_backbone:
        if model.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    # def configure_optimizers(self):
    #     return init_optimizer(parameters=self.parameters(), **self.hparams.optimizer)

        # log hyperparameters
        

        self.sigmoid = nn.functional.sigmoid
        self.fc1 = nn.Linear(16, 9)
        self.fc2 = nn.Linear(9, 3)
        self.mse_loss = nn.functional.mse_loss
        
    #returns the inlier index by RANSAC algorithm
    def _RANSAC(self, xyz):
        xyz = xyz.detach().cpu().numpy()
        cuboid = pyrsc.Cuboid()
        best_eq, best_inliers = cuboid.fit(xyz, thresh=0.004, maxIteration=5000)
        return best_inliers

    def _forward_voxelize(self, data_dict):
        #output data
        if self.hparams.model.use_coord:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), dim=1)
        data_dict["voxel_feats"] = common_ops.voxelization(data_dict["feats"].to(torch.float32), data_dict["p2v_map"].to(torch.int)) # (M, C), float, cuda
        backbone_output_dict = self.backbone(data_dict["voxel_feats"], data_dict["voxel_locs"], data_dict["v2p_map"])
        features = backbone_output_dict["point_features"][:, 0:self.hparams.model.feature_size]
        if len(data_dict["batch_divide"]) != 1:
            downsample_feat = torch.zeros([len(data_dict["batch_divide"]), self.hparams.model.feature_size, self.voxel_size, self.voxel_size, self.voxel_size], dtype=torch.float).cuda()
            density = torch.zeros([self.voxel_size, self.voxel_size, self.voxel_size], dtype=torch.float).cuda()
            feature_divide_start = 0
            for i in range(len(data_dict["batch_divide"])):
                feature_divide_end = feature_divide_start + data_dict["batch_divide"][i].item()
                current_features = features[feature_divide_start:feature_divide_end]
                xyz_i = data_dict["locs"][feature_divide_start:feature_divide_end]
                feature_divide_start = feature_divide_end
                downsample_feat[i] = self._voxelize(current_features, xyz_i)
            return downsample_feat
        else:
            density = torch.zeros([self.voxel_size, self.voxel_size, self.voxel_size], dtype=torch.float).cuda()
            downsample_feat = torch.zeros([self.hparams.model.feature_size, self.voxel_size, self.voxel_size, self.voxel_size], dtype=torch.float).cuda()
            xyz_i = data_dict["locs"]
            xyz_x = xyz_i[:,0]
            xyz_y = xyz_i[:,1]
            xyz_z = xyz_i[:,2]
            x_min = xyz_x.min()
            y_min = xyz_y.min()
            z_min = xyz_z.min()
            x_max = xyz_x.max()
            y_max = xyz_y.max()
            z_max = xyz_z.max()
            id = 0
            for x, y, z in xyz_i:
                x_grid = ((self.voxel_size-1)*(x-x_min)/(x_max-x_min)).to(torch.int)
                y_grid = ((self.voxel_size-1)*(y-y_min)/(y_max-y_min)).to(torch.int)
                z_grid = ((self.voxel_size-1)*(z-z_min)/(z_max-z_min)).to(torch.int)
                density[x_grid][y_grid][z_grid] = density[x_grid][y_grid][z_grid] + 1
                downsample_feat[:,x_grid,y_grid,z_grid] = features[id]
                id = id + 1
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    for z in range(self.voxel_size):
                        if density[x][y][z] != 0:
                            downsample_feat[:,x,y,z] = downsample_feat[:,x,y,z].clone()/density[x,y,z]
            else:
                return downsample_feat[None,:]
    # will be used during inference
    def forward(self, data_dict):
        output_dict = {}
        #get per-point features
        if self.hparams.model.use_coord:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), dim=1)
        data_dict["voxel_feats"] = common_ops.voxelization(data_dict["feats"].to(torch.float32), data_dict["p2v_map"].to(torch.int)) # (M, C), float, cuda
        backbone_output_dict = self.backbone(data_dict["voxel_feats"], data_dict["voxel_locs"], data_dict["v2p_map"])
        # features = backbone_output_dict["point_features"][:, 0:self.hparams.model.feature_size]
        # x = self.sigmoid(self.fc1(features)).clone()
        coord = backbone_output_dict["canonical_coordinate"]
        # divide different batches
        pred_R = torch.zeros([len(data_dict["batch_divide"]), 3, 3], dtype=torch.float).cuda()
        gt_R = data_dict["R"].to(torch.float32)
        coord_gt_R = []
        pred_T = []
        pred_stacked_R = []
        coord_divide_start = 0
        for i in range(len(data_dict["batch_divide"])):
            coord_divide_end = coord_divide_start + data_dict["batch_divide"][i].item()
            current_xyz = data_dict["locs"][coord_divide_start:coord_divide_end]
            # gt_xyz = data_dict["rotated_locs"][coord_divide_start:coord_divide_end]
            # current_rgb = data_dict["colors"][coord_divide_start:coord_divide_end]
            current_coord = coord[coord_divide_start:coord_divide_end]
            inlier_index = self._RANSAC(current_coord)
            current_xyz = (current_xyz.clone())[None, :].to(dtype=torch.float)
            current_nocs_xyz = (current_coord.clone())[None, :].to(dtype=torch.float)
            R, T, S = corresponding_points_alignment(current_nocs_xyz, current_xyz, estimate_scale=False, allow_reflection=True)    

            coord_divide_start = coord_divide_end
            # current_coord_gt_R = torch.mm(current_nocs_xyz[0, :, :], R[0, :, :]) + T[0, :]
            pred_R[i] = R[0, :, :]
            # pred_T.append(T)
            # pred_stacked_R.append(R)
            # coord_gt_R.append(current_coord_gt_R)
        output_dict["pred_R"] = pred_R
        output_dict["coord"] = coord
        # output_dict["coord_gt_R"] = torch.cat(coord_gt_R, dim=0)
        # output_dict["pred_T"] = torch.cat(pred_T, dim=0)
        # output_dict["pred_stacked_R"] = torch.cat(pred_stacked_R, dim=0)
        return output_dict


    def _rotation_loss(self, data_dict, output_dict):
        gt_R = data_dict["R"].view(-1, 9).to(torch.float32)
        pred_R = output_dict["pred_R"].view(-1, 9).to(torch.float32)
        return self.mse_loss(gt_R, pred_R)

    def _regression_loss(self, data_dict, output_dict):
        gt_xyz = data_dict["rotated_locs"].to(torch.float32)
        pred_xyz = output_dict["coord"].to(torch.float32)
        # T = output_dict["pred_T"].to(torch.float32)
        # R = output_dict["pred_stacked_R"].to(torch.float32)
        # return self.mse_loss(gt_xyz, torch.mm(pred_xyz, R) + T)
        return self.mse_loss(gt_xyz, pred_xyz)
    
    # def _translation_loss(self, data_dict, output_dict):
    #     T = torch.abs(output_dict["pred_T"].to(torch.float32))
    #     return T.mean() 

    def _loss(self, data_dict, output_dict):
        rotation_loss = self._rotation_loss(data_dict, output_dict)
        # translation_loss = self._translation_loss(data_dict, output_dict)
        regression_loss = self._regression_loss(data_dict, output_dict)
        total_loss = self.hparams.model.regression_loss_ratio * regression_loss + (1 -self.hparams.model.regression_loss_ratio) * rotation_loss
        return total_loss

    def training_step(self, data_dict, idx):
        output_dict = self.forward(data_dict)
        loss = self._loss(data_dict, output_dict)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.hparams.data.batch_size)
        return loss

    def validation_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self.forward(data_dict)
        loss = self._loss(data_dict, output_dict)

        # log losses
        self.log("val/loss", loss, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=True, batch_size=1)

        # if self.current_epoch > self.hparams.model.prepare_epochs:
        pred_obb = self._get_obb(data_dict["scan_ids"][0],
                                        data_dict["sem_labels"],
                                        data_dict["instance_ids"],
                                        output_dict["pred_R"][0].cpu(),)
        gt_obb = self._get_obb(data_dict["scan_ids"][0],
                                    data_dict["sem_labels"],
                                    data_dict["instance_ids"],
                                    data_dict["R"][0].cpu())
        return pred_obb, gt_obb

    def validation_epoch_end(self, outputs):
        # evaluate instance predictions
        # if self.current_epoch > self.hparams.model.prepare_epochs:
        all_pred_obbs = []
        all_gt_obbs = []
        for pred_obb, gt_obb in outputs:
            all_pred_obbs.append(pred_obb)
            all_gt_obbs.append(gt_obb)
        obb_direction_evaluator = GeneralDatasetEvaluator(self.hparams.data.class_names, self.hparams.data.ignore_label)
        obb_direction_eval_result = obb_direction_evaluator.evaluate(all_pred_obbs, all_gt_obbs, print_result=True)
        self.log("val_eval/AC_5", obb_direction_eval_result["all_ac_5"], prog_bar=True, on_step=False,
                    on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val_eval/AC_10", obb_direction_eval_result["all_ac_10"], prog_bar=True, on_step=False,
                    on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val_eval/AC_20", obb_direction_eval_result["all_ac_20"], prog_bar=True, on_step=False,
                    on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val_eval/Rerr", obb_direction_eval_result["all_err"], prog_bar=True, on_step=False,
                    on_epoch=True, sync_dist=True, batch_size=1)

    def test_step(self, data_dict, idx):
        start_time = time.time()
        # prepare input and forward
        output_dict = self.forward(data_dict)
        loss = self._loss(data_dict, output_dict)
        end_time = time.time() - start_time

        # log losses
        self.log("test/loss", loss, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=True, batch_size=1)

        # if self.current_epoch > self.hparams.model.prepare_epochs:
        pred_obb = self._get_obb(data_dict["scan_ids"][0],
                                        data_dict["sem_labels"],
                                        data_dict["instance_ids"],
                                        output_dict["pred_R"][0].cpu(),)
        gt_obb = self._get_obb(data_dict["scan_ids"][0],
                                    data_dict["sem_labels"],
                                    data_dict["instance_ids"],
                                    data_dict["R"][0].cpu())
        # angle = np.arccos(np.dot(pred_obb["front"], gt_obb["up"]))
        # if angle < 5/180 * 3.14:
        #     draw_prediction(data_dict, pred_obb["direction_pred"], self.hparams.data.class_names, True)
        # if angle > 30/180 * 3.14:
        #     draw_prediction(data_dict, pred_obb["direction_pred"], self.hparams.data.class_names, False)
        return pred_obb, gt_obb, end_time

    def test_epoch_end(self, results):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.model.prepare_epochs:
            all_pred_obbs = []
            all_gt_obbs = []
            inference_time = 0
            for pred_obb, gt_obb, end_time in results:
                all_pred_obbs.append(pred_obb)
                all_gt_obbs.append(gt_obb)
                inference_time += end_time
            self.print(f"Average inference time: {round(inference_time / len(results), 3)}s per object.")
            if self.hparams.inference.save_predictions:
                save_prediction(self.hparams.inference.output_dir, all_pred_obbs, self.hparams.data.class_names)
                self.custom_logger.info(f"\nPredictions saved at {os.path.join(self.hparams.inference.output_dir, 'instance')}\n")
                save_gt(self.hparams.inference.output_dir, all_pred_obbs, self.hparams.data.class_names)
                self.custom_logger.info(f"\nGround truths saved at {os.path.join(self.hparams.inference.output_dir, 'instance')}\n")

            if self.hparams.inference.evaluate:
                obb_direction_evaluator = GeneralDatasetEvaluator(self.hparams.data.class_names, self.hparams.data.ignore_label)
                obb_direction_eval_result = obb_direction_evaluator.evaluate(all_pred_obbs, all_gt_obbs, print_result=True)
                self.log("test_eval/AC_5", obb_direction_eval_result["all_ac_5"], prog_bar=True, on_step=False,
                            on_epoch=True, sync_dist=True, batch_size=1)
                self.log("test_eval/AC_10", obb_direction_eval_result["all_ac_10"], prog_bar=True, on_step=False,
                            on_epoch=True, sync_dist=True, batch_size=1)
                self.log("test_eval/AC_20", obb_direction_eval_result["all_ac_20"], prog_bar=True, on_step=False,
                            on_epoch=True, sync_dist=True, batch_size=1)
                self.log("test_eval/Rerr", obb_direction_eval_result["all_err"], prog_bar=True, on_step=False,
                            on_epoch=True, sync_dist=True, batch_size=1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer


    def _get_obb(self, scan_id, sem_labels, instance_id, R):
        obb = {}
        semantic_label = sem_labels.detach().cpu().numpy()
        instance_id = instance_id.detach().cpu().numpy()
        R = R.detach().cpu().numpy()
        obb["sem_label"] = np.argmax(np.bincount(semantic_label))
        obb["instance_id"] = np.argmax(np.bincount(instance_id))
        obb["scan_id"] = scan_id
        axis = np.vstack(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
        rotated_axis = np.matmul(axis , R)
        obb["front"] = rotated_axis[0]
        obb["up"] = rotated_axis[2]
        return obb