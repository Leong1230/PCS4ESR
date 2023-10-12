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
import cv2
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hydra
from hybridpc.optimizer.optimizer import cosine_lr_decay, adjust_learning_rate
from hybridpc.model.module import Backbone, Encoder, MinkUNetBackbone, ImplicitDecoder, Dense_Generator

from hybridpc.model.general_model import GeneralModel
from hybridpc.evaluation.semantic_segmentation import *
from torch.nn import functional as F

matplotlib.use('Agg')  # Use the Agg backend which is non-interactive and doesn't use tkinter

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
        self.use_decoder = cfg.model.network.use_decoder

        self.udf_decoder = ImplicitDecoder(
            "functa",
            cfg.model.network.udf_decoder,
            cfg.model.network.latent_dim,
            cfg.data.voxel_size,
            1,
            cfg.model.network.udf_decoder.activation        
        )
        if self.training_stage != 1:
            self.seg_backbone = MinkUNetBackbone(
                self.latent_dim + cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3,
                cfg.model.network.seg_decoder.feature_dim
            )

            self.seg_decoder = ImplicitDecoder(
                "seg",
                cfg.model.network.seg_decoder, 
                cfg.model.network.seg_decoder.feature_dim,
                cfg.data.voxel_size,
                cfg.data.classes,
                cfg.model.network.seg_decoder.activation        
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
        self.frames = {}
        if self.training_stage == 2:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.udf_decoder.parameters():
                param.requires_grad = False
        # if self.training_stage==3:
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False
        #     for param in self.udf_decoder.parameters():
        #         param.requires_grad = False
        #     for param in self.seg_backbone.parameters():
        #         param.requires_grad = False
        #     for param in self.seg_decoder.parameters():
        #         param.requires_grad = False
                 
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

    def visualize_udf_value(self, data_dict, values, epoch):
        N = data_dict['values'].shape[0]
        ground_truth = data_dict['values'].cpu().detach().numpy()
        predictions = values.squeeze().cpu().detach().numpy()

        # Sort ground truth and get sorting indices
        sorted_indices = np.argsort(ground_truth)
        sorted_ground_truth = ground_truth[sorted_indices]
        reordered_predictions = predictions[sorted_indices]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_ground_truth, label='Ground Truth', color='blue')
        plt.scatter(np.arange(N), reordered_predictions, label='Predictions', color='red', marker='o')
        plt.fill_between(np.arange(N), sorted_ground_truth, reordered_predictions, color='gray', alpha=0.2)
        plt.legend()
        plt.title("Comparison between Ground Truth and Predictions")
        plt.xlabel("Data Points (sorted by ground truth)")
        plt.ylabel("Value")
        
        # Convert the figure's content to a numpy array and store in the dictionary
        fig = plt.gcf()
        fig.canvas.draw()
        img_arr = np.array(fig.canvas.renderer.buffer_rgba())
        self.frames[epoch] = img_arr
        plt.close()


    def generate_video_from_memory(self, save_dir: str):
        img_array = [(np.array(self.frames[key]) * 255).astype(np.uint8) for key in sorted(self.frames.keys())]

        if len(img_array) == 0:
            print("No frames found!")
            return

        # Save the frames as a video using imageio
        imageio.mimsave(save_dir, img_array, fps=30)


    def forward(self, data_dict):
        encodes_dict = self.encoder(data_dict)
        segmentation = 0
        values = 0
        if self.training_stage != 1:
            seg_features = self.seg_backbone(encodes_dict['mixed_latent_codes'],  encodes_dict['voxel_coords'], encodes_dict['indices'][:, 0]) # B, C
            if self.use_decoder:
                segmentation = self.seg_decoder(seg_features['voxel_features'].F, data_dict['xyz'], encodes_dict['relative_coords'], encodes_dict["indices"])
            else:
                segmentation = seg_features['point_features']

            if self.hparams.model.recompute_udf:
                values = self.udf_decoder(encodes_dict['latent_codes'], encodes_dict['query_absolute_coords'], encodes_dict['query_relative_coords'], encodes_dict['query_indices'])
        else:
            values = self.udf_decoder(encodes_dict['latent_codes'], encodes_dict['query_absolute_coords'], encodes_dict['query_relative_coords'], encodes_dict['query_indices'])

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
        else :
            params_to_optimize = list(self.seg_backbone.parameters()) + list(self.seg_decoder.parameters())
        
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.hparams.model.optimizer.lr)
        return optimizer
    
    def training_step(self, data_dict):
        if self.training_stage == 1:
            """ UDF auto-encoder training stage """
            batch_size = self.hparams.data.batch_size
            encodes_dict, outputs, _ = self.forward(data_dict)
            self.visualize_udf_value(data_dict, outputs, self.current_epoch)
            udf_loss = self.udf_loss(encodes_dict, outputs)
            self.log("train/udf_loss", udf_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

            if self.current_epoch >= self.hparams.model.network.encoder.save_every_epoches and (self.current_epoch - self.hparams.model.network.encoder.save_every_epoches) % self.hparams.model.network.encoder.save_every_epoches == 0:
                self.generate_video_from_memory(f"video_at_epoch_{self.current_epoch}.mp4")


            return udf_loss
        
        elif self.training_stage == 2:
            """ segmentation training stage """

            batch_size = self.hparams.data.batch_size
            encodes_dict, values, outputs = self.forward(data_dict)
            seg_loss = self.seg_loss(data_dict, outputs)
            if self.hparams.model.recompute_udf:
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

        else:
            return 0
    

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
        
        elif self.training_stage == 2:
            """ segmentation training stage """
            
            batch_size = self.hparams.data.batch_size
            encodes_dict, values, outputs = self.forward(data_dict)
            seg_loss = self.seg_loss(data_dict, outputs)
            if self.hparams.model.recompute_udf:
                udf_loss = self.udf_loss(encodes_dict, values)
                self.log("val/seg_loss", seg_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log("val/udf_loss", udf_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
                            # Print the loss for this step
                print(f"Scene: {scene_name}, Test UDF Loss: {udf_loss.item()}")
                
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

        else: # test stage
            scene_name = data_dict['scene_names'][0]  # Assume scene_names is a list of strings
            voxel_num = data_dict["voxel_nums"][0]    # Assume voxel_nums is a list of ints
            # Inner loop

            batch_size = self.hparams.data.batch_size
            encodes_dict, values, segmentations = self.forward(data_dict)
            semantic_predictions = torch.argmax(segmentations, dim=-1)  # (B, N)
            semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["labels"], ignore_label=-1)
            semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["labels"], ignore_label=-1)
            self.log(
                "val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size
            )         
            self.log(
                "val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size
            )

            self.val_test_step_outputs.append((semantic_accuracy, semantic_mean_iou))

            udf_loss = self.udf_loss(encodes_dict, values)
            self.log("val/udf_loss", udf_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            # Print the loss for this step
            print(f"Scene: {scene_name}, Test UDF Loss: {udf_loss.item()}, Semantic Accuracy: {semantic_accuracy:.4f}, Semantic Mean IoU: {semantic_mean_iou:.4f}")

            # torch.set_grad_enabled(True)
            # self.trainer.model.train()
            # if self.hparams.model.inference.visualization:
            #     self.udf_visualization(data_dict, encodes_dict, self.current_epoch, udf_loss)
            torch.set_grad_enabled(True)
            if self.hparams.model.inference.visualization:
                self.udf_visualization(data_dict, encodes_dict, self.current_epoch, udf_loss, semantic_predictions)



    def test_step(self, data_dict, idx):
        scene_name = data_dict['scene_names'][0]  # Assume scene_names is a list of strings
        voxel_num = data_dict["voxel_nums"][0]    # Assume voxel_nums is a list of ints

        # Inner loop
        if self.hparams.model.inference.term == "UDF":
            batch_size = self.hparams.data.batch_size
            encodes_dict, values, segmentations = self.forward(data_dict)
            udf_loss = self.udf_loss(encodes_dict, values)
            self.log("train/udf_loss", udf_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            # Print the loss for this step
            print(f"Scene: {scene_name}, Test UDF Loss: {udf_loss.item()}")
            self.val_test_step_outputs.append((data_dict, encodes_dict, udf_loss))
            # torch.set_grad_enabled(True)
            # self.trainer.model.train()
            # if self.hparams.model.inference.visualization:
            #     self.udf_visualization(data_dict, encodes_dict, self.current_epoch, udf_loss)

    def on_test_epoch_end(self):
        # evaluate instance predictions
        if self.training_stage !=1:
            all_pred_insts = []
            all_gt_insts = []
            all_gt_insts_bbox = []
            all_sem_acc = []
            all_sem_miou = []
            for data_dict, encodes_dict, udf_loss in self.val_test_step_outputs:
                torch.set_grad_enabled(True)
                self.udf_visualization(data_dict, encodes_dict, self.current_epoch, udf_loss)
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

    def get_unique_color(self, voxel_id, num_voxels):
        # You can customize this function further if you have a preferred color scheme
        color_map = plt.cm.jet  # You can change this to any other colormap
        return color_map(voxel_id / num_voxels)[:3]

    def udf_visualization(self, data_dict, encodes_dict, current_epoch, udf_loss, semantic_predictions):
        voxel_latents = encodes_dict['latent_codes']
        voxel_num = voxel_latents.shape[0]
        scene_name = data_dict['scene_names'][0]
        loss_type = self.hparams.model.loss.loss_type

        save_dir = os.path.join(
            self.hparams.exp_output_root_path, 'inference', self.hparams.model.inference.split,
            'visualizations'
        )

        if self.hparams.model.dense_generator.type == 'voxel':
            voxel_id = 120
            # Extract points and labels
            original_points = encodes_dict['relative_coords'][:, 0, :][encodes_dict['indices'][:, 0] == voxel_id].detach().cpu().numpy()
            gt_labels = data_dict['labels'][encodes_dict['indices'][:, 0] == voxel_id].detach().cpu().numpy()
            predicted_labels = semantic_predictions[encodes_dict['indices'][:, 0] == voxel_id].detach().cpu().numpy()
            # original_points = data_dict['points'][data_dict['voxel_indices'][0] == voxel_id].detach().cpu().numpy()
            # gt_labels = data_dict['labels'][data_dict['voxel_indices'[0]] == voxel_id].detach().cpu().numpy()
            # predicted_labels = semantic_predictions[data_dict['voxel_indices'][0] == voxel_id].detach().cpu().numpy()

            # Color mapping
            unique_labels = np.unique(np.concatenate((gt_labels, predicted_labels)))
            mapping = {label: idx for idx, label in enumerate(unique_labels)}
            # cmap = plt.cm.get_cmap('jet', len(unique_labels))

            cmap = {
                0: [1, 0, 0],      # Red
                1: [0, 1, 0],      # Green
                2: [0, 0, 1],      # Blue
                3: [1, 1, 0],      # Yellow
                4: [0, 1, 1],      # Cyan
                5: [1, 0, 1],      # Magenta
                6: [0.5, 0.5, 0],  # Olive
                7: [0.5, 0, 0.5]   # Purple
            }

            # Number of unique classes
            num_gt_classes = len(np.unique(gt_labels))
            num_pred_classes = len(np.unique(predicted_labels))

            # Point-level accuracy
            correct_predictions = np.sum(gt_labels == predicted_labels)
            total_points = len(gt_labels)
            accuracy = correct_predictions / total_points

            print(f"Number of GT classes: {num_gt_classes}, Number of predicted classes: {num_pred_classes}, Point-level accuracy: {accuracy:.4f}")
            # overall_correct_predictions = 0
            # overall_total_points = 0

            # for voxel_id in range(voxel_num):
            #     # Extract points and labels
            #     original_points = encodes_dict['relative_coords'][:, 0, :][encodes_dict['indices'][:, 0] == voxel_id].detach().cpu().numpy()
            #     gt_labels = data_dict['labels'][encodes_dict['indices'][:, 0] == voxel_id].detach().cpu().numpy()
            #     predicted_labels = semantic_predictions[encodes_dict['indices'][:, 0] == voxel_id].detach().cpu().numpy()

            #     # Color mapping
            #     unique_labels = np.unique(gt_labels)
            #     cmap = plt.cm.get_cmap('jet', len(unique_labels))
                
            #     # Number of unique classes
            #     num_gt_classes = len(np.unique(gt_labels))
            #     num_pred_classes = len(np.unique(predicted_labels))

            #     # Point-level accuracy for the current voxel
            #     correct_predictions = np.sum(gt_labels == predicted_labels)
            #     total_points = len(gt_labels)
            #     accuracy = correct_predictions / total_points

            #     # Accumulate the overall counts for accuracy computation
            #     overall_correct_predictions += correct_predictions
            #     overall_total_points += total_points

            #     print(f"Voxel {voxel_id}: Number of GT classes: {num_gt_classes}, Number of predicted classes: {num_pred_classes}, Point-level accuracy: {accuracy:.4f}")

            # overall_accuracy = overall_correct_predictions / overall_total_points
            # print(f"Overall Point-level accuracy for all voxels: {overall_accuracy:.4f}")

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

            # 4. Dense Point Cloud colored uniformly
            dense_points, _ = self.dense_generator.generate_point_cloud(data_dict, encodes_dict, voxel_latents, voxel_id)
            uniform_color = [0, 0, 1]  # e.g. blue color
            dense_points_cloud = o3d.geometry.PointCloud()
            dense_points_cloud.points = o3d.utility.Vector3dVector(dense_points)
            dense_points_cloud.colors = o3d.utility.Vector3dVector(np.tile(uniform_color, (len(dense_points), 1)))

            if self.hparams.model.inference.show_visualizations:
                o3d.visualization.draw_geometries([original_points_cloud])
                o3d.visualization.draw_geometries([prediction_pointcloud])
                o3d.visualization.draw_geometries([error_pointcloud])
                o3d.visualization.draw_geometries([dense_points_cloud])


            if self.hparams.model.inference.save_predictions:
                filename_base = f'Single_voxel_{self.hparams.model.network.encoder.voxel_size_out}_from_{scene_name}_voxelid_{voxel_id}_{loss_type}_udf_loss_{udf_loss:.5f}'
                
                # Save the visualizations with modified filenames
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_origin_class_num_{num_gt_classes}.ply'), original_points_cloud)
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_prediction_class_num_{num_pred_classes}.ply'), prediction_pointcloud)
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_error_accuracy_{accuracy:.4f}.ply'), error_pointcloud)
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_dense.ply'), dense_points_cloud)

                # Save rotating videos with modified filenames
                self.save_rotating_video_from_object(original_points_cloud, os.path.join(save_dir, f'{filename_base}_origin_class_num_{num_gt_classes}.mp4'))
                self.save_rotating_video_from_object(prediction_pointcloud, os.path.join(save_dir, f'{filename_base}_prediction_class_num_{num_pred_classes}.mp4'))
                self.save_rotating_video_from_object(error_pointcloud, os.path.join(save_dir, f'{filename_base}_error_accuracy_{accuracy:.4f}.mp4'))
                self.save_rotating_video_from_object(dense_points_cloud, os.path.join(save_dir, f'{filename_base}_dense.mp4'))


        elif self.hparams.model.dense_generator.type == 'multiple_voxels':
            # for voxel_id in range(voxel_num):
            voxel_id = voxel_num - 10
            original_indices = encodes_dict['indices'][encodes_dict['indices'][:, 0] == voxel_id]
            if self.hparams.model.inference.visualize_given_voxels:
                voxel_ids = self.hparams.model.inference.visualize_ids
            else:
                voxel_ids = original_indices[0]
            all_original_points = []  # List to accumulate all the absolute coordinates
            all_dense_points = []
            all_query_points = []
            all_points_colors = []
            all_query_points_colors = []
            all_dense_points_colors = []
            for index, voxel_id in enumerate(voxel_ids):
                voxel_center = encodes_dict['voxel_coords'][:, 1:4][voxel_id] * self.hparams.model.network.encoder.voxel_size_out + self.hparams.model.network.encoder.voxel_size_out/2 #M, 3
                relative_points = (encodes_dict['relative_coords'][encodes_dict['indices'][:, 0] == voxel_id])[:, 0, :].cpu().numpy()
                absolute_points = relative_points + voxel_center.cpu().numpy()
                # Append the absolute coordinates to the all_points list
                all_original_points.append(absolute_points)

                query_relative_points = (encodes_dict['query_relative_coords'][encodes_dict['query_indices'][:, 0] == voxel_id])[:, 0, :].cpu().numpy()
                query_absolute_points = query_relative_points + voxel_center.cpu().numpy()
                all_query_points.append(query_absolute_points)

                original_indices = encodes_dict['indices'][encodes_dict['indices'][:, 0] == voxel_id] # P,K
                current_voxel_ids = original_indices[0] #, K
                dense_points, duration = self.dense_generator.generate_point_cloud(data_dict, encodes_dict, voxel_latents, current_voxel_ids)
                all_dense_points.append(dense_points + voxel_center.cpu().numpy())

                color = self.get_unique_color(index, len(voxel_ids))     
                all_points_colors.append(np.repeat([color], absolute_points.shape[0], axis=0))
                all_query_points_colors.append(np.repeat([color], query_absolute_points.shape[0], axis=0))
                all_dense_points_colors.append(np.repeat([color], dense_points.shape[0], axis=0))


            # Stack the colors
            all_points_colors_np = np.vstack(all_points_colors)
            all_query_points_colors_np = np.vstack(all_query_points_colors)
            all_dense_points_colors_np = np.vstack(all_dense_points_colors)
            # stack points
            all_points_np = np.vstack(all_original_points)
            all_dense_points_np = np.vstack(all_dense_points)
            all_query_points_np = np.vstack(all_query_points)

            # Use open3d to visualize the point cloud
            points_cloud = o3d.geometry.PointCloud()
            points_cloud.points = o3d.utility.Vector3dVector(all_points_np)
            query_points_cloud = o3d.geometry.PointCloud()
            query_points_cloud.points = o3d.utility.Vector3dVector(all_query_points_np)
            points_cloud.colors = o3d.utility.Vector3dVector(all_points_colors_np)
        
            # red_color = [1, 0, 0]
            blue_color = [0, 0, 1]
            # points_cloud.paint_uniform_color(red_color)
            query_points_cloud.paint_uniform_color(blue_color)

            # Merge the point clouds
            merged_pcd = points_cloud + query_points_cloud
            original_points_cloud = points_cloud

            dense_points_cloud = o3d.geometry.PointCloud()
            dense_points_cloud.points = o3d.utility.Vector3dVector(all_dense_points_np)
            dense_points_cloud.colors = o3d.utility.Vector3dVector(all_dense_points_colors_np)

            if self.hparams.model.inference.show_visualizations:
                o3d.visualization.draw_geometries([dense_points_cloud])
                o3d.visualization.draw_geometries([original_points_cloud])

            if self.hparams.model.inference.save_predictions:
                filename_base = f'Multiple_voxels_{self.hparams.model.network.encoder.voxel_size_out}_from_{scene_name}_voxelid_{voxel_id}_{loss_type}_udf_loss_{udf_loss:.5f}'
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

