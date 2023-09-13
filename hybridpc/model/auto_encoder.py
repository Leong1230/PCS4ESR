from pyexpat import features
from sklearn.metrics import jaccard_score
import torch
import time
import os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import imageio
import math
import torchmetrics
import open3d as o3d
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import hydra
from hybridpc.optimizer.optimizer import cosine_lr_decay, adjust_learning_rate
from hybridpc.model.module import Backbone, Encoder, MinkUNetBackbone, ImplicitDecoder, Dense_Generator

from hybridpc.model.general_model import GeneralModel
from hybridpc.evaluation.semantic_segmentation import *
from torch.nn import functional as F

class AutoEncoder(GeneralModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters(cfg)
        # Shared latent code across both decoders
        # set automatic_optimization to False
        # self.automatic_optimization = False
        self.training_stage = cfg.model.training_stage
        
        self.latent_dim = cfg.model.network.latent_dim

        self.encoder = Encoder(cfg)
        self.udf_decoder = ImplicitDecoder(
            "functa",
            cfg.model.network.k_neighbors,
            cfg.model.network.interpolation_mode,
            cfg.model.network.latent_dim,
            cfg.model.network.udf_decoder.input_dim,
            cfg.model.network.udf_decoder.hidden_dim,
            cfg.model.network.udf_decoder.num_hidden_layers_before_skip,
            cfg.model.network.udf_decoder.num_hidden_layers_after_skip,
            1
        )
        if self.training_stage == 2:
            self.seg_backbone = MinkUNetBackbone(
                self.latent_dim + cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3,
                cfg.model.network.seg_decoder.feature_dim
            )

            self.seg_decoder = ImplicitDecoder(
                "seg",
                cfg.model.network.k_neighbors,
                cfg.model.network.interpolation_mode,
                cfg.model.network.seg_decoder.feature_dim,
                cfg.model.network.seg_decoder.input_dim,
                cfg.model.network.seg_decoder.hidden_dim,
                cfg.model.network.seg_decoder.num_hidden_layers_before_skip,
                cfg.model.network.seg_decoder.num_hidden_layers_after_skip,
                cfg.data.classes
            )


        self.dense_generator = Dense_Generator(
            self.udf_decoder,
            cfg.model.network.encoder.voxel_size_out,
            cfg.model.dense_generator.num_steps,
            cfg.model.dense_generator.num_points,
            cfg.model.dense_generator.threshold,
            cfg.model.dense_generator.filter_val,
            cfg.model.dense_generator.type
        )

        # Initialize an empty dictionary to store the latent codes
        self.latent_codes = {}
        if self.training_stage == 2:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.udf_decoder.parameters():
                param.requires_grad = False
                 
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, cfg, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Create a new model instance
        model = cls(cfg)

        # Load the model weights selectively
        encoder_state_dict = {k[len("encoder."):]: v for k, v in checkpoint['state_dict'].items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(encoder_state_dict)

        udf_decoder_state_dict = {k[len("udf_decoder."):]: v for k, v in checkpoint['state_dict'].items() if k.startswith("udf_decoder.")}
        model.udf_decoder.load_state_dict(udf_decoder_state_dict)

        return model
      # The inner loop optimizer is applied to the latent code


    def forward(self, data_dict):
        encodes_dict = self.encoder(data_dict)
        segmentation = 0
        if self.training_stage == 2:
            seg_features = self.seg_backbone(encodes_dict['mixed_latent_codes'], encodes_dict['voxel_coords'], encodes_dict['indices'][:, 0]) # B, C
            segmentation = self.seg_decoder(seg_features['voxel_features'].F, encodes_dict['relative_coords'], encodes_dict["indices"])
            values = self.udf_decoder(encodes_dict['latent_codes'], encodes_dict['query_relative_coords'], encodes_dict['query_indices'])
        else:
            values = self.udf_decoder(encodes_dict['latent_codes'], encodes_dict['query_relative_coords'], encodes_dict['query_indices'])

        return encodes_dict, values, segmentation

    def seg_loss(self, data_dict, outputs):
        loss = F.cross_entropy(outputs, data_dict['labels'].long(), ignore_index=-1)
        
        return loss
    
    def udf_loss(self, encodes_dict, outputs):
        if self.hparams.model.loss.loss_type == "L1":
            loss = torch.nn.L1Loss(reduction='mean')(torch.clamp(outputs, max=self.hparams.data.udf_queries.max_dist), torch.clamp(encodes_dict['values'], max=self.hparams.data.udf_queries.max_dist))
        else:
            loss = F.mse_loss(outputs, encodes_dict['values'])

        return loss

    def configure_optimizers(self):
        if self.training_stage == 1:
            params_to_optimize = list(self.encoder.parameters()) + list(self.udf_decoder.parameters())
        elif self.training_stage == 2:
            params_to_optimize = list(self.seg_backbone.parameters()) + list(self.seg_decoder.parameters())
        else:
            raise ValueError(f"Invalid training stage: {self.training_stage}")
        
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.hparams.model.optimizer.lr)
        return optimizer
    
    def training_step(self, data_dict):
        if self.training_stage == 1:
            """ UDF auto-encoder training stage """
            batch_size = self.hparams.data.batch_size
            encodes_dict, outputs, _ = self.forward(data_dict)
            udf_loss = self.udf_loss(encodes_dict, outputs)
            self.log("train/udf_loss", udf_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

            # if self.current_epoch >= self.hparams.model.dense_generator.prepare_epochs:
            #     if self.hparams.model.inference.visualization:
            #         self.udf_visualization(data_dict, encodes_dict, self.current_epoch, udf_loss)

            return udf_loss
        
        else:
            """ segmentation training stage """

            batch_size = self.hparams.data.batch_size
            encodes_dict, values, outputs = self.forward(data_dict)
            seg_loss = self.seg_loss(data_dict, outputs)
            udf_loss = self.udf_loss(encodes_dict, values)

            self.log("train/udf_loss", udf_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log("train/seg_loss", seg_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            # Calculating the metrics
            semantic_predictions = torch.argmax(outputs, dim=-1)  # (B, N)
            semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["labels"], ignore_label=-1)
            semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["labels"], ignore_label=-1)
            self.log(
                "train/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size
            )         
            self.log(
                "train/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size
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
        if self.training_stage == 1:
            """ UDF auto-encoder training stage """
            batch_size = self.hparams.data.batch_size
            encodes_dict, outputs, _ = self.forward(data_dict)
            udf_loss = self.udf_loss(encodes_dict, outputs)
            self.log("val/udf_loss", udf_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        else:
            """ segmentation training stage """

            batch_size = self.hparams.data.batch_size
            encodes_dict, values, outputs = self.forward(data_dict)
            seg_loss = self.seg_loss(data_dict, outputs)
            udf_loss = self.udf_loss(encodes_dict, values)

            self.log("val/seg_loss", seg_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log("val/udf_loss", udf_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            # Calculating the metrics
            semantic_predictions = torch.argmax(outputs, dim=-1)  # (B, N)
            semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["labels"], ignore_label=-1)
            semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["labels"], ignore_label=-1)
            self.log(
                "val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size
            )         
            self.log(
                "val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size
            )
            self.val_test_step_outputs.append((semantic_accuracy, semantic_mean_iou))

        torch.set_grad_enabled(True)
        if self.current_epoch >= self.hparams.model.dense_generator.prepare_epochs:
            if self.hparams.model.inference.visualization:
                self.udf_visualization(data_dict, encodes_dict, self.current_epoch, udf_loss)



    def test_step(self, data_dict, idx):
        scene_name = data_dict['scene_names'][0]  # Assume scene_names is a list of strings
        voxel_num = data_dict["voxel_nums"][0]    # Assume voxel_nums is a list of ints

        # Inner loop
        if self.hparams.model.inference.term == "UDF":
            batch_size = self.hparams.data.batch_size
            encodes_dict, outputs = self.forward(data_dict)
            udf_loss = self.udf_loss(encodes_dict, outputs)
            self.log("train/udf_loss", udf_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            # Print the loss for this step
            print(f"Scene: {scene_name}, Test UDF Loss: {value_loss.item()}")

            # if self.current_epoch >= self.hparams.model.dense_generator.prepare_epochs:
            #     if self.hparams.model.inference.visualization:
            #         self.udf_visualization(data_dict, encodes_dict, self.current_epoch, udf_loss)
            # Calculating the metrics

    def udf_visualization(self, data_dict, encodes_dict, current_epoch, udf_loss):
        voxel_latents = encodes_dict['latent_codes']
        voxel_num = voxel_latents.shape[0]
        scene_name = data_dict['scene_names'][0]
        loss_type = self.hparams.model.loss.loss_type

        save_dir = os.path.join(
            self.hparams.exp_output_root_path, 'inference', self.hparams.model.inference.split,
            'udf_visualizations'
        )

        if self.hparams.model.dense_generator.type == 'voxel':
            # for voxel_id in range(voxel_num):
            voxel_id = 2
            original_points = encodes_dict['relative_coords'][encodes_dict['indices'] == voxel_id].cpu().numpy()
            original_points_cloud = o3d.geometry.PointCloud()
            original_points_cloud.points = o3d.utility.Vector3dVector(original_points)

            dense_points, duration = self.dense_generator.generate_point_cloud(data_dict, encodes_dict, voxel_latents, voxel_id)
            dense_points_cloud = o3d.geometry.PointCloud()
            dense_points_cloud.points = o3d.utility.Vector3dVector(dense_points)

            if self.hparams.model.inference.show_visualizations:
                o3d.visualization.draw_geometries([dense_points_cloud])
                o3d.visualization.draw_geometries([original_points_cloud])

            if self.hparams.model.inference.save_predictions:
                filename_base = f'Single_voxel_{self.hparams.model.network.encoder.voxel_size_out}_from_{scene_name}_voxelid_{voxel_id}_{loss_type}_udf_loss_{udf_loss:.5f}'
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_dense.ply'), dense_points_cloud)
                self.save_rotating_video_from_object(dense_points_cloud, os.path.join(save_dir, f'{filename_base}_dense.mp4'))
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_origin.ply'), original_points_cloud)
                self.save_rotating_video_from_object(original_points_cloud, os.path.join(save_dir, f'{filename_base}_origin.mp4'))

        elif self.hparams.model.dense_generator.type == 'multiple_voxels':
            # for voxel_id in range(voxel_num):
            voxel_id = voxel_num - 10
            original_indices = encodes_dict['indices'][encodes_dict['indices'][:, 0] == voxel_id]
            voxel_ids = original_indices[0]
            all_original_points = []  # List to accumulate all the absolute coordinates
            all_dense_points = []
            all_query_points = []
            for voxel_id in voxel_ids: 
                voxel_center = encodes_dict['voxel_coords'][:, 1:4][voxel_id] * self.hparams.model.network.encoder.voxel_size_out - self.hparams.model.network.encoder.voxel_size_out/2
                relative_points = (encodes_dict['relative_coords'][encodes_dict['indices'][:, 0] == voxel_id])[:, 0, :].cpu().numpy()
                absolute_points = relative_points + voxel_center.cpu().numpy()
                # Append the absolute coordinates to the all_points list
                all_original_points.append(absolute_points)

                query_relative_points = (encodes_dict['query_relative_coords'][encodes_dict['query_indices'][:, 0] == voxel_id])[:, 0, :].cpu().numpy()
                query_absolute_points = query_relative_points + voxel_center.cpu().numpy()
                all_query_points.append(query_absolute_points)

                original_indices = encodes_dict['indices'][encodes_dict['indices'][:, 0] == voxel_id]
                current_voxel_ids = original_indices[0]
                dense_points, duration = self.dense_generator.generate_point_cloud(data_dict, encodes_dict, voxel_latents, current_voxel_ids)
                all_dense_points.append(dense_points + voxel_center.cpu().numpy())



            all_points_np = np.vstack(all_original_points)
            all_dense_points_np = np.vstack(all_dense_points)
            all_query_points_np = np.vstack(all_query_points)
            # Use open3d to visualize the point cloud
            points_cloud = o3d.geometry.PointCloud()
            points_cloud.points = o3d.utility.Vector3dVector(all_points_np)
            query_points_cloud = o3d.geometry.PointCloud()
            query_points_cloud.points = o3d.utility.Vector3dVector(all_query_points_np)
        
            red_color = [1, 0, 0]
            blue_color = [0, 0, 1]
            points_cloud.paint_uniform_color(red_color)
            query_points_cloud.paint_uniform_color(blue_color)

            # Merge the point clouds
            merged_pcd = points_cloud + query_points_cloud
            original_points_cloud = merged_pcd

            dense_points_cloud = o3d.geometry.PointCloud()
            dense_points_cloud.points = o3d.utility.Vector3dVector(all_dense_points_np)

            if self.hparams.model.inference.show_visualizations:
                o3d.visualization.draw_geometries([dense_points_cloud])
                o3d.visualization.draw_geometries([original_points_cloud])

            if self.hparams.model.inference.save_predictions:
                filename_base = f'Single_voxel_{self.hparams.model.network.encoder.voxel_size_out}_from_{scene_name}_voxelid_{voxel_id}_{loss_type}_udf_loss_{udf_loss:.5f}'
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_dense.ply'), dense_points_cloud)
                self.save_rotating_video_from_object(dense_points_cloud, os.path.join(save_dir, f'{filename_base}_dense.mp4'))
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_origin.ply'), original_points_cloud)
                self.save_rotating_video_from_object(original_points_cloud, os.path.join(save_dir, f'{filename_base}_origin.mp4'))
                    
        else:
            original_points = data_dict["xyz"].cpu().numpy()
            original_points_cloud = o3d.geometry.PointCloud()
            original_points_cloud.points = o3d.utility.Vector3dVector(original_points)

            # dense_points, duration = self.dense_generator.generate_all_voxel_point_clouds(data_dict, functa_modulations['voxel_features'].F)
            dense_points, duration = self.dense_generator.generate_point_cloud(data_dict, functa_modulations['voxel_features'].F, 0)

            dense_points_cloud = o3d.geometry.PointCloud()
            dense_points_cloud.points = o3d.utility.Vector3dVector(dense_points)

            if self.hparams.model.inference.show_visualizations:
                o3d.visualization.draw_geometries([dense_points_cloud])
                o3d.visualization.draw_geometries([original_points_cloud])

            if self.hparams.model.inference.save_predictions:
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{scene_name}_voxel_{self.hparams.model.network.encoder.voxel_size_out}_{loss_type}_udf_loss_{udf_loss:.5f}_dense.ply'), dense_points_cloud)
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{scene_name}_voxel_{self.hparams.model.network.encoder.voxel_size_out}_{loss_type}_udf_loss_{udf_loss:.5f}_origin.ply'), original_points_cloud)            

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

    # def save_rotating_video_from_object(self, obj: o3d.geometry.Geometry3D, save_dir: str):
    #     """
    #     Generates a rotating video from an Open3D object.

    #     Parameters:
    #     - obj: The Open3D object to visualize.
    #     - save_dir: The directory (including filename) to save the video.
    #     """
    #     # Create a visualizer
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window(visible=True)  # visible window for offscreen rendering
    #     vis.add_geometry(obj)
        
    #     frames = []
        
    #     def rotate_view(vis):
    #         ctr = vis.get_view_control()
    #         ctr.rotate(5.0, 0.0)  # rotate 5 degrees around z-axis
            
    #         frame = vis.capture_screen_float_buffer(True)
    #         frames.append((np.asarray(frame) * 255).astype(np.uint8))
            
    #         # Save in intervals to not hold too much in memory
    #         if len(frames) == 360:  # Save after capturing 360 frames, i.e., every 12 seconds at 30fps
    #             imageio.mimsave(save_dir, frames, fps=30)
    #             frames.clear()  # Clear the frames list
            
    #         return False
        
    #     # Set the callback function to be called before rendering
    #     vis.register_animation_callback(rotate_view)
        
    #     # Run the visualizer
    #     vis.run()
        
    #     # After the window is closed
    #     if frames:  # Save any remaining frames
    #         imageio.mimsave(save_dir, frames, fps=30)
        
    #     vis.destroy_window()
