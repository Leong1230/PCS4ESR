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
from hybridpc.model.module import Backbone, MinkUNetBackbone, Dense_Generator
from hybridpc.model.general_model import GeneralModel
from torch.nn import functional as F

class AutoDecoder(GeneralModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters(cfg)
        # Shared latent code across both decoders
        # set automatic_optimization to False
        self.automatic_optimization = False
        self.training_stage = cfg.model.training_stage
        
        self.latent_dim = cfg.model.network.latent_dim
        self.seg_loss_weight = cfg.model.network.seg_loss_weight

        self.functa_backbone = Backbone(
            backbone_type=cfg.model.network.backbone_type,
            input_channel=self.latent_dim, output_channel=cfg.model.network.modulation_dim, block_channels=cfg.model.network.functa_blocks,
            block_reps=cfg.model.network.functa_block_reps,
            sem_classes=cfg.data.classes
        )
        self.functa_decoder = ImplicitDecoder(
            "functa",
            cfg.model.network.modulation_dim,
            cfg.model.network.functa_decoder.input_dim,
            cfg.model.network.functa_decoder.hidden_dim,
            cfg.model.network.functa_decoder.num_hidden_layers_before,
            cfg.model.network.functa_decoder.num_hidden_layers_after,
            1
        )

        if self.training_stage == 2:
            self.seg_backbone = MinkUNetBackbone(
                self.latent_dim,
                cfg.model.network.modulation_dim
            )

            self.seg_decoder = ImplicitDecoder(
                "seg",
                cfg.model.network.modulation_dim,
                cfg.model.network.seg_decoder.input_dim,
                cfg.model.network.seg_decoder.hidden_dim,
                cfg.model.network.seg_decoder.num_hidden_layers_before,
                cfg.model.network.seg_decoder.num_hidden_layers_after,
                cfg.data.classes
            )


        self.dense_generator = Dense_Generator(
            self.functa_decoder,
            cfg.data.voxel_size,
            cfg.model.dense_generator.num_steps,
            cfg.model.dense_generator.num_points,
            cfg.model.dense_generator.threshold,
            cfg.model.dense_generator.filter_val,
            cfg.model.dense_generator.type
        )

        # Initialize an empty dictionary to store the latent codes
        self.latent_codes = {}
        self.latent_optimizers = {}
        # self.inner_optimizer = torch.optim.Adam([torch.zeros(1)], lr=self.hparams.model.optimizer.inner_loop_lr)
        if self.training_stage == 2:
            for param in self.functa_backbone.parameters():
                param.requires_grad = False

            for param in self.functa_decoder.parameters():
                param.requires_grad = False   
                 
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, cfg, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Create a new model instance
        model = cls(cfg)

        # Load the model weights selectively
        functa_backbone_state_dict = {k[len("functa_backbone."):]: v for k, v in checkpoint['state_dict'].items() if k.startswith("functa_backbone.")}
        model.functa_backbone.load_state_dict(functa_backbone_state_dict)

        functa_decoder_state_dict = {k[len("functa_decoder."):]: v for k, v in checkpoint['state_dict'].items() if k.startswith("functa_decoder.")}
        model.functa_decoder.load_state_dict(functa_decoder_state_dict)

        return model
      # The inner loop optimizer is applied to the latent code


    def forward(self, data_dict, latent_code):
        functa_modulations = self.functa_backbone(latent_code, data_dict['voxel_coords'], data_dict['voxel_indices']) # B, C
        segmentation = None
        if self.training_stage == 2:
            seg_modulations = self.seg_backbone(latent_code, data_dict['voxel_coords'], data_dict['voxel_indices']) # B, C
            segmentation = self.seg_decoder(seg_modulations['voxel_features'].F, data_dict['points'], data_dict["voxel_indices"])  # embeddings (B, C) coords (N, 3) indices (N, )

        if self.hparams.model.network.auto_encoder: # use auto encoder by passing the latent code to the functa backbone
            values = self.functa_decoder(functa_modulations['voxel_features'].F, data_dict['query_points'], data_dict["query_voxel_indices"]) # embeddings (B, D1) coords (M, 3) indices (M, )
        else:
            values = self.functa_decoder(latent_code, data_dict['query_points'], data_dict["query_voxel_indices"])
        # values = self.functa_decoder(latent_code, data_dict['query_points'], data_dict["query_voxel_indices"]) # embeddings (B, D1) coords (M, 3) indices (M, )


        return {"segmentation": segmentation, "values": values}

    def _loss(self, data_dict, output_dict, latent_code, stage):
        seg_loss = 0
        if self.training_stage == 2:
            seg_loss = F.cross_entropy(output_dict['segmentation'], data_dict['labels'].long(), ignore_index=-1)
        reg_loss = torch.mean(latent_code.pow(2))

        if stage == "Auto_Decoder":
            reg_loss = reg_loss * self.hparams.model.loss.code_reg_lambda * min(1, self.current_epoch / 100)
        else:
            reg_loss = reg_loss * self.hparams.model.loss.code_reg_lambda 

        if self.hparams.model.loss.loss_type == "L1":
            value_loss = torch.nn.L1Loss(reduction='mean')(torch.clamp(output_dict['values'], max=self.hparams.data.udf_queries.max_dist), torch.clamp(data_dict['values'], max=self.hparams.data.udf_queries.max_dist))
        else:
            value_loss = F.mse_loss(output_dict['values'], data_dict['values'])
        
        if self.hparams.model.loss.use_reg:
            return seg_loss, value_loss + reg_loss
        else:
            return seg_loss, value_loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
    #     return optimizer

    def configure_optimizers(self):
        if self.training_stage == 1:
            params_to_optimize = list(self.functa_backbone.parameters()) + list(self.functa_decoder.parameters())
        elif self.training_stage == 2:
            params_to_optimize = list(self.seg_backbone.parameters()) + list(self.seg_decoder.parameters())
        else:
            raise ValueError(f"Invalid training stage: {self.training_stage}")
        
        outer_optimizer = torch.optim.Adam(params_to_optimize, lr=self.hparams.model.optimizer.lr)
        return outer_optimizer

    
    # def update_inner_optimizer_params(self, latent_code):
    #     # Replace the parameters handled by the inner optimizer
    #     self.inner_optimizer.param_groups[0]['params'] = [latent_code]
    
    def training_step(self, data_dict):
        if self.training_stage == 1:
            """ auto-decoder training stage """

            scene_names = data_dict['scene_names']  # Assume scene_names is a list of strings
            voxel_nums = data_dict["voxel_nums"]    # Assume voxel_nums is a list of ints
            batch_size = len(scene_names)

            # Retrieve or initialize the latent codes
            latent_codes = []
            for idx in range(batch_size):
                if scene_names[idx] in self.latent_codes:
                    latent_code = self.latent_codes[scene_names[idx]]
                else:
                    if self.hparams.model.loss.norm_initialization:
                        std_dev = 0.01
                        latent_code = torch.randn(voxel_nums[idx], self.latent_dim, device=self.device) * std_dev
                        latent_code.requires_grad_()

                    else:
                        latent_code = torch.rand(voxel_nums[idx], self.latent_dim,requires_grad=True, device=self.device)

                    latent_optimizer = torch.optim.Adam([latent_code], lr=self.hparams.model.optimizer.latent_lr)
                    self.latent_optimizers[scene_names[idx]] = latent_optimizer
                    self.latent_codes[scene_names[idx]] = latent_code
                self.latent_optimizers[scene_names[idx]].zero_grad() # zero grads for all latent_optimizers
                latent_codes.append(latent_code)

            # Stack the latent codes into a batch
            latent_code = torch.cat(latent_codes, dim=0)

            # Creating the optimizer for latent_code
            outer_optimizer = self.optimizers()
            # outer_optimizer.param_groups[0]['capturable'] = True
            outer_optimizer.zero_grad()

            output_dict = self.forward(data_dict, latent_code)
            _, value_loss = self._loss(data_dict, output_dict, latent_code, "Auto_Decoder")
            # latent_optimizer.zero_grad()
            self.manual_backward(value_loss)  
            # latent_code = latent_code.detach().requires_grad_()       
            outer_optimizer.step()
            for idx in range(batch_size):
                # self.latent_optimizers[scene_names[idx]].param_groups[0]['capturable'] = True
                self.latent_optimizers[scene_names[idx]].step()
            # self.latent_codes[scene_names[0]] = latent_code  # Detach to avoid tracking its history

            self.log("train/udf_loss", value_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

            return value_loss
        
        else:
            """ segmentation training stage """

            scene_names = data_dict['scene_names']  # Assume scene_names is a list of strings
            voxel_nums = data_dict["voxel_nums"]    # Assume voxel_nums is a list of ints
            batch_size = len(scene_names)

            # Retrieve or initialize the latent codes
            latent_codes = []
            latent_optimizers = []
            need_updates = False
            for idx in range(batch_size):
                if scene_names[idx] in self.latent_codes:
                    latent_code = self.latent_codes[scene_names[idx]]
                    latent_codes.append(latent_code)
                else:
                    need_updates = True
                    if self.hparams.model.loss.norm_initialization:
                        std_dev = 0.01
                        latent_code = torch.randn(voxel_nums[idx], self.latent_dim, device=self.device) * std_dev
                        latent_code.requires_grad_()
                    else:
                        latent_code = torch.rand(voxel_nums[idx], self.latent_dim,requires_grad=True, device=self.device)
                    latent_codes.append(latent_code)
                    latent_optimizers.append(torch.optim.Adam([latent_codes[idx]], lr=self.hparams.model.optimizer.inner_loop_lr))
                    latent_optimizers[idx].zero_grad()

            
            # Create a progress bar
            if need_updates:
                for iter in tqdm(range(self.hparams.model.optimizer.inner_loop_steps), desc='Optimization Progress'):  # Perform multiple steps
                    latent_code = torch.cat(latent_codes, dim=0)
                    output_dict = self.forward(data_dict, latent_code)
                    _, value_loss = self._loss(data_dict, output_dict, latent_code, "Optimization")
                    for idx in range(batch_size):
                        # Adjust learning rate
                        adjust_learning_rate(self.hparams.model.optimizer.inner_loop_lr, latent_optimizers[idx], iter, self.hparams.model.optimizer.inner_loop_steps, self.hparams.model.optimizer.decreased_by)
                        latent_optimizers[idx].zero_grad()
                    self.manual_backward(value_loss)       
                    # Step the latent optimizer and zero its gradients
                    for idx in range(batch_size):
                        latent_optimizers[idx].step()
                    
                if self.hparams.model.network.use_modulation:
                    start_idx = 0
                    functa_modulations = self.functa_backbone(latent_code, data_dict['voxel_coords'], data_dict['voxel_indices']) # B, C
                    for idx in range(batch_size):
                        end_idx = start_idx + data_dict['voxel_nums'][idx]
                        latent = functa_modulations['voxel_features'].F[start_idx:end_idx].detach()
                        self.latent_codes[scene_names[idx]] = latent
                        start_idx = end_idx
                else:
                    for idx in range(batch_size):
                        self.latent_codes[scene_names[idx]] = latent_codes[idx].detach()
                print(f"Train UDF Loss: {value_loss.item()}")

            # Stack the latent codes into a batch
            latent_code = torch.cat(latent_codes, dim=0)
            output_dict = self.forward(data_dict, latent_code)
            seg_loss, _ = self._loss(data_dict, output_dict, latent_code, "Optimization")
            outer_optimizer = self.optimizers()
            outer_optimizer.zero_grad() 
            self.manual_backward(seg_loss)
            outer_optimizer.step()

            self.log("train/seg_loss", seg_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            # Calculating the metrics
            semantic_predictions = torch.argmax(output_dict['segmentation'], dim=-1)  # (B, N)
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

        if self.training_stage == 1:
            # Update the learning rates for all latent optimizers
            for scene_name, optimizer in self.latent_optimizers.items():
                cosine_lr_decay(
                    optimizer, self.hparams.model.optimizer.latent_lr, self.current_epoch,
                    self.hparams.model.lr_decay.decay_start_epoch, self.hparams.model.trainer.max_epochs, 1e-6
                )

    def validation_step(self, data_dict, idx):
        scene_name = data_dict['scene_names'][0]  # Assume scene_names is a list of strings
        voxel_num = data_dict["voxel_nums"][0]    # Assume voxel_nums is a list of ints

        # Inner loop
        if self.training_stage == 1:
            if self.current_epoch > self.hparams.model.network.prepare_epochs:
                torch.set_grad_enabled(True)

                # voxel_nums = data_dict["voxel_nums"]    # Assume voxel_nums is a list of ints

                if self.hparams.model.loss.norm_initialization:
                    std_dev = 0.01
                    latent_code = torch.randn(voxel_num, self.latent_dim, device=self.device) * std_dev
                    latent_code.requires_grad_()
                else:
                    latent_code = torch.rand(voxel_num, self.latent_dim,requires_grad=True, device=self.device)
                latent_optimizer = torch.optim.Adam([latent_code], lr=self.hparams.model.optimizer.inner_loop_lr)
                # Create a progress bar
                use_tqdm = False
                if use_tqdm:
                    pbar = tqdm(range(self.hparams.model.optimizer.inner_loop_steps), desc='Optimization Progress')
                else:
                    pbar = range(self.hparams.model.optimizer.inner_loop_steps)

                for iter in pbar:  # Perform multiple steps
                    # Adjust learning rate
                    adjust_learning_rate(
                        self.hparams.model.optimizer.inner_loop_lr, latent_optimizer, 
                        iter, self.hparams.model.optimizer.inner_loop_steps, 
                        self.hparams.model.optimizer.decreased_by
                    )

                    output_dict = self.forward(data_dict, latent_code)
                    _, value_loss = self._loss(data_dict, output_dict, latent_code, "Optimization ")
                    # print(f"Step {iter}, Loss: {value_loss.item()}")

                    latent_optimizer.zero_grad()
                    self.manual_backward(value_loss)         
                    # Step the latent optimizer and zero its gradients
                    latent_optimizer.step()

                    # Update the postfix of the progress bar with the current loss
                    if use_tqdm:
                        pbar.set_postfix({'udf_loss': value_loss.item()})


                self.log("val/udf_loss", value_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

            # Calculating the metrics
        else: 
            # Retrieve or initialize the latent codes
            udf_loss = 0
            if scene_name in self.latent_codes:
                latent_code = self.latent_codes[scene_name]
            else:
                torch.set_grad_enabled(True)
                if self.hparams.model.loss.norm_initialization:
                    std_dev = 0.01
                    latent_code = torch.randn(voxel_num, self.latent_dim, device=self.device) * std_dev
                    latent_code.requires_grad_()
                else:
                    latent_code = torch.rand(voxel_num, self.latent_dim,requires_grad=True, device=self.device)
                latent_optimizer = torch.optim.Adam([latent_code], lr=self.hparams.model.optimizer.inner_loop_lr)

                use_tqdm = True
                if use_tqdm:
                    pbar = tqdm(range(self.hparams.model.optimizer.inner_loop_steps), desc='Optimization Progress')
                else:
                    pbar = range(self.hparams.model.optimizer.inner_loop_steps)

                for iter in pbar:  # Perform multiple steps
                    # Adjust learning rate
                    adjust_learning_rate(self.hparams.model.optimizer.inner_loop_lr, latent_optimizer, iter, self.hparams.model.optimizer.inner_loop_steps, self.hparams.model.optimizer.decreased_by)

                    output_dict = self.forward(data_dict, latent_code)
                    _, value_loss = self._loss(data_dict, output_dict, latent_code, "Optimization")
                    latent_optimizer.zero_grad()
                    self.manual_backward(value_loss)         
                    # Step the latent optimizer and zero its gradients
                    latent_optimizer.step()
                    # Update the postfix of the progress bar with the current loss
                    if use_tqdm:
                        pbar.set_postfix({'udf_loss': value_loss.item()})

                udf_loss = value_loss
                # Print the loss for this step
                print(f"Scene: {scene_name}, Val UDF Loss: {value_loss.item()}")
                
                if self.hparams.model.network.use_modulation:
                    functa_modulations = self.functa_backbone(latent_code, data_dict['voxel_coords'], data_dict['voxel_indices']) # B, C
                    self.latent_codes[scene_name] = functa_modulations['voxel_features'].F.detach()
                else:
                    self.latent_codes[scene_name] = latent_code.detach()

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
            torch.set_grad_enabled(True)
            if self.current_epoch >= self.hparams.model.dense_generator.prepare_epochs:
                if self.hparams.model.inference.visualization:
                    self.udf_visualization(data_dict, output_dict, latent_code, self.current_epoch, udf_loss)



    def test_step(self, data_dict, idx):
        scene_name = data_dict['scene_names'][0]  # Assume scene_names is a list of strings
        voxel_num = data_dict["voxel_nums"][0]    # Assume voxel_nums is a list of ints

        # Inner loop
        if self.hparams.model.inference.term == "UDF":
           # Initialize the latent codes
            torch.set_grad_enabled(True)
            if self.hparams.model.loss.norm_initialization:
                std_dev = 0.01
                latent_code = torch.randn(voxel_num, self.latent_dim, device=self.device) * std_dev
                latent_code.requires_grad_()
            else:
                latent_code = torch.rand(voxel_num, self.latent_dim,requires_grad=True, device=self.device)
            latent_optimizer = torch.optim.Adam([latent_code], lr=self.hparams.model.optimizer.inner_loop_lr)

            pbar = tqdm(range(self.hparams.model.optimizer.inner_loop_steps), desc='Optimization Progress')
            for iter in pbar:  # Perform multiple steps
                # Adjust learning rate
                adjust_learning_rate(self.hparams.model.optimizer.inner_loop_lr, latent_optimizer, iter, self.hparams.model.optimizer.inner_loop_steps, self.hparams.model.optimizer.decreased_by)

                output_dict = self.forward(data_dict, latent_code)
                _, value_loss = self._loss(data_dict, output_dict, latent_code, "Optimization")
                
                latent_optimizer.zero_grad()
                self.manual_backward(value_loss)         
                # Step the latent optimizer and zero its gradients
                latent_optimizer.step()
                pbar.set_postfix({'udf_loss': value_loss.item()})

            udf_loss = value_loss
            # Print the loss for this step
            print(f"Scene: {scene_name}, Test UDF Loss: {value_loss.item()}")

            if self.current_epoch >= self.hparams.model.dense_generator.prepare_epochs:
                if self.hparams.model.inference.visualization:
                    self.udf_visualization(data_dict, output_dict, latent_code, self.current_epoch, udf_loss)
            # Calculating the metrics

    # def udf_visualization(self, data_dict, output_dict, latent_code, current_epoch, udf_loss):

    #     functa_modulations = self.functa_backbone(latent_code, data_dict['voxel_coords'], data_dict['voxel_indices']) # B, C
    #     voxel_num = functa_modulations['voxel_features'].F.shape[0]
    #     # voxel_id = torch.randint(0, voxel_num, (1,)).item()
    #     scene_name = data_dict['scene_names'][0]
    #     loss_type = self.hparams.model.loss.loss_type

    #     if self.hparams.model.dense_generator.type == 'voxel':
    #         original_points = data_dict["points"][data_dict['voxel_indices'] == voxel_id].cpu().numpy()
    #     else:
    #         original_points = data_dict["xyz"].cpu().numpy()

    #     # create point cloud for original points
    #     original_points_cloud = o3d.geometry.PointCloud()
    #     original_points_cloud.points = o3d.utility.Vector3dVector(original_points)
    #     # original_points_cloud.paint_uniform_color([1, 0, 0])  # red color

    #     if self.hparams.model.dense_generator.type == 'voxel': 
    #         dense_points, duration = self.dense_generator.generate_point_cloud(data_dict, functa_modulations['voxel_features'].F, voxel_id)
    #         # Create dense point cloud
    #         dense_points_cloud = o3d.geometry.PointCloud()
    #         dense_points_cloud.points = o3d.utility.Vector3dVector(dense_points)
    #         # Translate the dense point cloud along z-axis by 1 unit to avoid overlap

    #     else: 
    #         dense_points, duration = self.dense_generator.generate_all_voxel_point_clouds(data_dict, functa_modulations['voxel_features'].F)
    #         # Create dense point cloud
    #         dense_points_cloud = o3d.geometry.PointCloud()
    #         dense_points_cloud.points = o3d.utility.Vector3dVector(dense_points)
    #         # dense_points_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # gray color

    #     # visualize the point clouds
    #     if self.hparams.model.inference.show_visualizations:
    #         o3d.visualization.draw_geometries([dense_points_cloud])
    #         o3d.visualization.draw_geometries([original_points_cloud]) # show the original point cloud for scene visualization

    #     # Save the point clouds
    #     if self.hparams.model.inference.save_predictions:
    #         save_dir = os.path.join(
    #             self.hparams.exp_output_root_path, 'inference', self.hparams.model.inference.split,
    #             'udf_visualizations'
    #         )
    #         if self.hparams.model.dense_generator.type == 'voxel':
    #             o3d.io.write_point_cloud(os.path.join(save_dir, 'Single_voxel_' + str(self.hparams.data.voxel_size) + '_from_' + scene_name + '_' + loss_type + 'udf_loss_' + '{:.5f}'.format(udf_loss) + '_dense.ply'), dense_points_cloud)
    #             self.save_rotating_video_from_object(dense_points_cloud, os.path.join(save_dir, 'Single_voxel_' + str(self.hparams.data.voxel_size) + '_from_' + scene_name + '_' + loss_type + 'udf_loss_' + '{:.5f}'.format(udf_loss) + '_dense.mp4'))
    #             o3d.io.write_point_cloud(os.path.join(save_dir, 'Single_voxel_' + str(self.hparams.data.voxel_size) + '_from_' + scene_name + '_' + loss_type + '_udf_loss_' + '{:.5f}'.format(udf_loss) + '_origin.ply'), original_points_cloud)
    #             self.save_rotating_video_from_object(original_points_cloud, os.path.join(save_dir, 'Single_voxel_' + str(self.hparams.data.voxel_size) + '_from_' + scene_name + '_' + loss_type + '_udf_loss_' + '{:.5f}'.format(udf_loss) + '_origin.mp4'))
    #         else:
    #             o3d.io.write_point_cloud(os.path.join(save_dir, scene_name + '_voxel_' + str(self.hparams.data.voxel_size) + '_' + loss_type + '_udf_loss_' + '{:.5f}'.format(udf_loss) + '_dense.ply'), dense_points_cloud)
    #             o3d.io.write_point_cloud(os.path.join(save_dir, scene_name + '_voxel_' + str(self.hparams.data.voxel_size) + '_' +  loss_type + '_udf_loss_' + '{:.5f}'.format(udf_loss) + '_origin.ply'), original_points_cloud)            
    #         self.print(f"\nPredictions saved at {os.path.abspath(save_dir)}")

    def udf_visualization(self, data_dict, output_dict, latent_code, current_epoch, udf_loss):
        functa_modulations = self.functa_backbone(latent_code, data_dict['voxel_coords'], data_dict['voxel_indices']) # B, C
        voxel_num = functa_modulations['voxel_features'].F.shape[0]
        scene_name = data_dict['scene_names'][0]
        loss_type = self.hparams.model.loss.loss_type

        save_dir = os.path.join(
            self.hparams.exp_output_root_path, 'inference', self.hparams.model.inference.split,
            'udf_visualizations'
        )

        if self.hparams.model.dense_generator.type == 'voxel':
            for voxel_id in range(voxel_num):
                original_points = data_dict["points"][data_dict['voxel_indices'] == voxel_id].cpu().numpy()
                original_points_cloud = o3d.geometry.PointCloud()
                original_points_cloud.points = o3d.utility.Vector3dVector(original_points)

                dense_points, duration = self.dense_generator.generate_point_cloud(data_dict, functa_modulations['voxel_features'].F, voxel_id)
                dense_points_cloud = o3d.geometry.PointCloud()
                dense_points_cloud.points = o3d.utility.Vector3dVector(dense_points)

                if self.hparams.model.inference.show_visualizations:
                    o3d.visualization.draw_geometries([dense_points_cloud])
                    o3d.visualization.draw_geometries([original_points_cloud])

                if self.hparams.model.inference.save_predictions:
                    filename_base = f'Single_voxel_{self.hparams.data.voxel_size}_from_{scene_name}_voxelid_{voxel_id}_{loss_type}_udf_loss_{udf_loss:.5f}'
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
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{scene_name}_voxel_{self.hparams.data.voxel_size}_{loss_type}_udf_loss_{udf_loss:.5f}_dense.ply'), dense_points_cloud)
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{scene_name}_voxel_{self.hparams.data.voxel_size}_{loss_type}_udf_loss_{udf_loss:.5f}_origin.ply'), original_points_cloud)            

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
