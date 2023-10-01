from pyexpat import features
from sklearn.metrics import jaccard_score
import torch
import time
import os
import torch.nn as nn
import numpy as np
import math
import torchmetrics
import imageio
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
            output_channel = cfg.model.network.seg_decoder.feature_dim
        else:
            output_channel = cfg.data.classes # use the number of classes as the output channel
        if self.feature_in == "mixed_latent":
            input_channel = cfg.model.network.latent_dim + cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3 
        else: 
            input_channel = cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3
        self.backbone = MinkUNetBackbone(
            input_channel,
            output_channel
        )
        self.seg_decoder = ImplicitDecoder(
            "seg",
            cfg.model.network.seg_decoder, 
            cfg.model.network.seg_decoder.feature_dim,
            cfg.data.voxel_size,
            cfg.data.classes
        )
        self.val_test_step_outputs = []


    def forward(self, data_dict, latent_code):
        if self.feature_in == "mixed_latent":
            backbone_output_dict = self.backbone(
            torch.cat((latent_code, data_dict['voxel_features']), dim=1), data_dict["voxel_coords"], data_dict["voxel_indices"][:, 0]
        )
        else:
            backbone_output_dict = self.backbone(
                data_dict["voxel_features"], data_dict["voxel_coords"], data_dict["voxel_indices"]
            )

        if self.use_decoder:
            backbone_output_dict['semantic_scores'] = self.seg_decoder(backbone_output_dict['voxel_features'].F, data_dict['xyz'], data_dict['points'], data_dict["voxel_indices"])  # embeddings (B, C) coords (N, 3) indices (N, )
        else:
            backbone_output_dict['semantic_scores'] = backbone_output_dict['point_features'] # use the point features as the semantic scores
        return backbone_output_dict
        # return 0 

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
        torch.set_grad_enabled(False)
        scene_name = data_dict['scene_names'][0]  # Assume scene_names is a list of strings
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
        print(f"Scene: {scene_name}, Semantic Accuracy: {semantic_accuracy:.4f}, Semantic Mean IoU: {semantic_mean_iou:.4f}")
        torch.set_grad_enabled(True)
        if self.hparams.model.inference.visualization:
                self.udf_visualization(data_dict, self.current_epoch, semantic_predictions)


    def udf_visualization(self, data_dict, current_epoch, semantic_predictions):
        scene_name = data_dict['scene_names'][0]

        save_dir = os.path.join(
            self.hparams.exp_output_root_path, 'inference', self.hparams.model.inference.split,
            'visualizations'
        )

        if self.hparams.model.inference.type == 'voxel':
            voxel_id = 200

            # Extract points and labels
            original_points = data_dict['points'][:, 0, :][data_dict['voxel_indices'][:, 0] == voxel_id].detach().cpu().numpy()
            gt_labels = data_dict['labels'][data_dict['voxel_indices'][:, 0] == voxel_id].detach().cpu().numpy()
            predicted_labels = semantic_predictions[data_dict['voxel_indices'][:, 0] == voxel_id].detach().cpu().numpy()

            # Color mapping
            unique_labels = np.unique(np.concatenate((gt_labels, predicted_labels)))
            mapping = {label: idx for idx, label in enumerate(unique_labels)}
            # cmap = plt.cm.get_cmap('jet', len(unique_labels))

            cmap = {
                0:  [1, 0, 0],       # Red
                1:  [0, 1, 0],       # Green
                2:  [0, 0, 1],       # Blue
                3:  [1, 1, 0],       # Yellow
                4:  [0, 1, 1],       # Cyan
                5:  [1, 0, 1],       # Magenta
                6:  [0.5, 0.5, 0],   # Olive
                7:  [0.5, 0, 0.5],   # Purple
                8:  [0, 0.5, 0.5],   # Teal
                9:  [0.5, 0.5, 0.5], # Grey
                10: [0.6, 0.2, 0.2], # Darker Red
                11: [0.2, 0.6, 0.2], # Darker Green
                12: [0.2, 0.2, 0.6], # Darker Blue
                13: [0.7, 0.7, 0.2], # Dirty Yellow
                14: [0.2, 0.7, 0.7], # Dirty Cyan
                15: [0.7, 0.2, 0.7], # Dirty Magenta
                16: [0.3, 0.3, 0.1], # Darker Olive
                17: [0.3, 0.1, 0.3], # Darker Purple
                18: [0.1, 0.3, 0.3], # Darker Teal
                19: [0.7, 0.4, 0.1], # Orangey
            }
            
            # Number of unique classes
            num_gt_classes = len(np.unique(gt_labels))
            num_pred_classes = len(np.unique(predicted_labels))

            # Point-level accuracy
            correct_predictions = np.sum(gt_labels == predicted_labels)
            total_points = len(gt_labels)
            accuracy = correct_predictions / total_points

            print(f"Number of GT classes: {num_gt_classes}, Number of predicted classes: {num_pred_classes}, Point-level accuracy: {accuracy:.4f}")


            # 1. Original Point Cloud colored by GT labels
            original_colors = np.array([cmap[mapping[labels]] for labels in gt_labels])[:, :3]
            original_points_cloud = o3d.geometry.PointCloud()
            original_points_cloud.points = o3d.utility.Vector3dVector(original_points)
            original_points_cloud.colors = o3d.utility.Vector3dVector(original_colors)

            # 2. Prediction Point Cloud colored by predicted labels
            pred_colors = np.array([cmap[mapping[labels]] for labels in predicted_labels])[:, :3]
            prediction_pointcloud = o3d.geometry.PointCloud()
            prediction_pointcloud.points = o3d.utility.Vector3dVector(original_points)
            prediction_pointcloud.colors = o3d.utility.Vector3dVector(pred_colors)

            # 3. Error Point Cloud
            error_colors = np.array([[0.5, 0.5, 0.5] if gt == pred else [1, 0, 0] for gt, pred in zip(gt_labels, predicted_labels)])
            error_pointcloud = o3d.geometry.PointCloud()
            error_pointcloud.points = o3d.utility.Vector3dVector(original_points)
            error_pointcloud.colors = o3d.utility.Vector3dVector(error_colors)

            if self.hparams.model.inference.save_predictions:
                filename_base = f'Vanilla-minkowski_Single_voxel_{self.hparams.data.voxel_size}_from_{scene_name}_voxelid_{voxel_id}'
                
                # Save the visualizations with modified filenames
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_origin_class_num_{num_gt_classes}.ply'), original_points_cloud)
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_prediction_class_num_{num_pred_classes}.ply'), prediction_pointcloud)
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_error_accuracy_{accuracy:.4f}.ply'), error_pointcloud)

                # Save rotating videos with modified filenames
                self.save_rotating_video_from_object(original_points_cloud, os.path.join(save_dir, f'{filename_base}_origin_class_num_{num_gt_classes}.mp4'))
                self.save_rotating_video_from_object(prediction_pointcloud, os.path.join(save_dir, f'{filename_base}_prediction_class_num_{num_pred_classes}.mp4'))
                self.save_rotating_video_from_object(error_pointcloud, os.path.join(save_dir, f'{filename_base}_error_accuracy_{accuracy:.4f}.mp4'))

                self.print(f"\nPredictions saved at {os.path.abspath(save_dir)}")


    def save_rotating_video_from_object(self, obj: o3d.geometry.Geometry3D, save_dir: str):
        """
        Generates a rotating video from an Open3D object.

        Parameters:
        - obj: The Open3D object to visualize.
        - save_dir: The directory (including filename) to save the video.
        """
        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)  # invisible window for offscreen rendering
        vis.add_geometry(obj)
        
        # Rotate the view every frame and capture frames
        frames = []
        for i in range(360):  # 360 frames for a complete rotation
            ctr = vis.get_view_control()
            ctr.rotate(5.0, 0.0)  # rotate 5 degrees around z-axis
            vis.poll_events()
            vis.update_renderer()
            frame = vis.capture_screen_float_buffer(False)
            frames.append((np.asarray(frame) * 255).astype(np.uint8))
        
        # Save the frames as a video using imageio
        imageio.mimsave(save_dir, frames, fps=30)
        
        vis.destroy_window()