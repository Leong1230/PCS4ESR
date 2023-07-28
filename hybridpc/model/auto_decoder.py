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

class AutoDecoder(GeneralModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters(cfg)
        # Shared latent code across both decoders
        # set automatic_optimization to False
        self.automatic_optimization = False
        
        self.latent_dim = cfg.model.network.latent_dim
        self.seg_loss_weight = cfg.model.network.seg_loss_weight

        self.functa_backbone = Backbone(
            backbone_type=cfg.model.network.backbone_type,
            input_channel=self.latent_dim, output_channel=cfg.model.network.modulation_dim, block_channels=cfg.model.network.blocks,
            block_reps=cfg.model.network.block_reps,
            sem_classes=cfg.data.classes
        )
        self.seg_backbone = Backbone(
            backbone_type=cfg.model.network.backbone_type,
            input_channel=self.latent_dim, output_channel=cfg.model.network.modulation_dim, block_channels=cfg.model.network.blocks,
            block_reps=cfg.model.network.block_reps,
            sem_classes=cfg.data.classes
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
        self.functa_decoder = ImplicitDecoder(
            "functa",
            cfg.model.network.modulation_dim,
            cfg.model.network.functa_decoder.input_dim,
            cfg.model.network.functa_decoder.hidden_dim,
            cfg.model.network.functa_decoder.num_hidden_layers_before_skip,
            cfg.model.network.functa_decoder.num_hidden_layers_after_skip,
            1
        )

        self.dense_generator = Dense_Generator(
            self.functa_decoder,
            cfg.data.voxel_size,
            cfg.model.dense_generator.num_steps,
            cfg.model.dense_generator.num_points,
            cfg.model.dense_generator.threshold,
            cfg.model.dense_generator.filter_val
        )
      # The inner loop optimizer is applied to the latent code
        self.inner_optimizer = torch.optim.SGD([p for p in self.parameters()], lr=cfg.model.optimizer.inner_loop_lr)



    def forward(self, data_dict, latent_code):
        
        seg_modulations = self.seg_backbone(latent_code, data_dict['voxel_coords'], data_dict['voxel_indices']) # B, C
        functa_modulations = self.functa_backbone(latent_code, data_dict['voxel_coords'], data_dict['voxel_indices']) # B, C

        # modulations = latent_code
        segmentation = self.seg_decoder(seg_modulations['voxel_features'].F, data_dict['points'], data_dict["voxel_indices"])  # embeddings (B, C) coords (N, 3) indices (N, )
        values = self.functa_decoder(functa_modulations['voxel_features'].F, data_dict['query_points'], data_dict["query_voxel_indices"]) # embeddings (B, D1) coords (M, 3) indices (M, )

        return {"segmentation": segmentation, "values": values}

    def _loss(self, data_dict, output_dict):
        # seg_loss = F.cross_entropy(output_dict['segmentation'].view(-1, self.hparams.cfg.data.category_num), data_dict['labels'].view(-1))
        seg_loss = F.cross_entropy(output_dict['segmentation'], data_dict['labels'].long(), ignore_index=-1)

        value_loss = F.mse_loss(output_dict['values'], data_dict['values'])

        # loss_i = torch.nn.L1Loss(reduction='none')(torch.clamp(output_dict['values'], max=self.hparams.cfg.data.udf_queries.max_dist),torch.clamp(data_dict['values'], max=self.hparams.cfg.data.udf_queries.max_dist))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        # value_loss = loss_i.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

        return seg_loss, value_loss, self.seg_loss_weight * seg_loss + (1 - self.seg_loss_weight) * value_loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
    #     return optimizer

    def configure_optimizers(self):
        outer_optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.model.optimizer.lr)

        return outer_optimizer

    
    def training_step(self, data_dict):
        voxel_num = data_dict["voxel_coords"].shape[0] # B voxels
        latent_code = torch.rand(voxel_num, self.latent_dim, requires_grad=True, device=self.device)

        # Creating the optimizer for latent_code
        outer_optimizer = self.optimizers()
        # outer_optimizer.zero_grad() 
        latent_optimizer = torch.optim.Adam([latent_code], lr=self.hparams.model.optimizer.inner_loop_lr)

        train_loss_list = []  # To store the value_loss for each step
        # Inner loop
        for _ in range(self.hparams.model.optimizer.inner_loop_steps):  # Perform multiple steps
            output_dict = self.forward(data_dict, latent_code)
            seg_loss, value_loss, _ = self._loss(data_dict, output_dict)
            self.manual_backward(value_loss)
            
            # Step the latent optimizer and zero its gradients
            latent_optimizer.step()
            latent_optimizer.zero_grad()
            train_loss_list.append(value_loss.item())  # Store the value_loss for this step

        
        # After the loop, plot the value_loss for each step
        # plt.plot(train_loss_list)
        # plt.xlabel('Step')
        # plt.ylabel('Value Loss')
        # plt.show()
        # plt.clf()  # Clear the current figure

        # self.log("train/value_loss", value_loss, on_step=True, on_epoch=False)

        # Outer loop
        output_dict = self.forward(data_dict, latent_code)
        seg_loss, value_loss, total_loss = self._loss(data_dict, output_dict)
        self.manual_backward(total_loss)
        outer_optimizer.step()
        outer_optimizer.zero_grad()

        self.log("train/udf_loss", value_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, sync_dist=True)
        # Calculating the metrics
        semantic_predictions = torch.argmax(output_dict['segmentation'], dim=-1)  # (B, N)
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["labels"], ignore_label=-1)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["labels"], ignore_label=-1)
        self.log(
            "train/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.log(
            "train/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )

        return total_loss

    

    def on_train_epoch_end(self):
        cosine_lr_decay(
            self.trainer.optimizers[0], self.hparams.model.optimizer.lr, self.current_epoch,
            self.hparams.model.lr_decay.decay_start_epoch, self.hparams.model.trainer.max_epochs, 1e-6
        )

    def validation_step(self, data_dict, idx):
        torch.set_grad_enabled(True)
        voxel_num = data_dict["voxel_coords"].shape[0]  # K voxels
        latent_code = torch.rand(voxel_num, self.latent_dim, requires_grad=True, device=self.device)

        # Creating the optimizer for latent_code
        latent_optimizer = torch.optim.Adam([latent_code], lr=self.hparams.model.optimizer.inner_loop_lr)
        
        value_loss_list = []  # To store the value_loss for each step
        # Inner loop
        for step in range(self.hparams.model.optimizer.inner_loop_steps):  # Perform multiple steps
            latent_optimizer.zero_grad()
            output_dict = self.forward(data_dict, latent_code)
            _, udf_loss,  _ = self._loss(data_dict, output_dict)
            self.manual_backward(udf_loss)

            # Step the latent optimizer and zero its gradients
            latent_optimizer.step()
            latent_optimizer.zero_grad()
            # udf_loss_list.append(udf_loss.item())  # Store the value_loss for this step

        self.log("val/udf_loss", udf_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # After the loop, plot the value_loss for each step
        # plt.plot(value_loss_list)
        # plt.xlabel('Step')
        # plt.ylabel('Value Loss')
        # plt.show()
        # No outer loop for validation

        # Calculating the metrics
        torch.set_grad_enabled(False)
        output_dict = self.forward(data_dict, latent_code)
        semantic_predictions = torch.argmax(output_dict['segmentation'], dim=-1)  # (B, N)
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["labels"], ignore_label=-1)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["labels"], ignore_label=-1)
        self.log(
            "val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.log(
            "val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.val_test_step_outputs.append((semantic_accuracy, semantic_mean_iou))
        if self.current_epoch > self.hparams.model.dense_generator.prepare_epochs:
            if self.hparams.model.inference.visualize_udf:
                self.udf_visualization(data_dict, output_dict, latent_code, self.current_epoch, udf_loss)


    def udf_visualization(self, data_dict, output_dict, latent_code, current_epoch, udf_loss):


        modulations = self.functa_backbone(latent_code, data_dict['voxel_coords']) # B, C 
        voxel_num = modulations.F.shape[0]
        voxel_id = torch.randint(0, voxel_num, (1,)).item()

        dense_points, duration = self.dense_generator.generate_point_cloud(modulations.F, voxel_id)

        # Create dense point cloud
        dense_points_cloud = o3d.geometry.PointCloud()
        dense_points_cloud.points = o3d.utility.Vector3dVector(dense_points)

        # Translate the dense point cloud along z-axis by 1 unit to avoid overlap
        dense_points_cloud.translate([0, 0, 1], relative=False)

        original_points = data_dict["points"][data_dict['voxel_indices'] == voxel_id].cpu().numpy()
        sampled_points = data_dict["query_points"][data_dict['query_voxel_indices'] == voxel_id].cpu().numpy()

        # create point cloud for original points
        original_points_cloud = o3d.geometry.PointCloud()
        original_points_cloud.points = o3d.utility.Vector3dVector(original_points)
        original_points_cloud.paint_uniform_color([1, 0, 0])  # red color

        # create point cloud for sampled points
        sampled_points_cloud = o3d.geometry.PointCloud()
        sampled_points_cloud.points = o3d.utility.Vector3dVector(sampled_points)
        sampled_points_cloud.paint_uniform_color([0, 1, 0])  # green color

        # Merge original and sampled point clouds
        merged_points_cloud = original_points_cloud + sampled_points_cloud

        # visualize the point clouds
        # o3d.visualization.draw_geometries([dense_points_cloud])
        # o3d.visualization.draw_geometries([merged_points_cloud])

        # Save the point clouds
        save_dir = os.path.join(self.hparams.exp_output_root_path)
        o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.hparams.data.voxel_size) + '_epoch_' + str(current_epoch) + 'udf_loss_' + '{:.5f}'.format(udf_loss.item()) + '_dense.ply'), dense_points_cloud)
        o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.hparams.data.voxel_size) + '_epoch_' + str(current_epoch) + '_udf_loss_' + '{:.5f}'.format(udf_loss.item()) + '_merged.ply'), merged_points_cloud)

        flag = 1


