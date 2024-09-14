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
import importlib
from pycg import exp, vis
from hybridpc.optimizer.optimizer import cosine_lr_decay, adjust_learning_rate
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
import MinkowskiEngine as ME
from hybridpc.evaluation import UnitMeshEvaluator
from hybridpc.model.module import Encoder, Generator, Dense_Generator, visualize_tool, MultiScaleInterpolatedDecoder, PointTransformerV3
from hybridpc.utils.samples import BatchedSampler
from lightning.pytorch.utilities import grad_norm
from pytorch3d.ops import knn_points
from torchviz import make_dot

from nksr.svh import SparseFeatureHierarchy


from hybridpc.model.general_model import GeneralModel
from torch.nn import functional as F

class PCS4ESR(GeneralModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters(cfg)
        # Shared latent code across both decoders
        # set automatic_optimization to False
        # self.automatic_optimization = False
        
        self.latent_dim = cfg.model.network.latent_dim
        self.decoder_type = cfg.model.network.udf_decoder.decoder_type
        self.eikonal = cfg.data.supervision.eikonal.loss
        self.surface_normal_supervision = cfg.data.supervision.on_surface.normal_loss
        self.flip_eikonal = cfg.data.supervision.eikonal.flip
        self.backbone = cfg.model.network.backbone
        if self.backbone == "PointTransformerV3":
            self.point_transformer = PointTransformerV3(
                backbone_cfg=cfg.model.network.point_transformerv3
            )
        # self.mask = cfg.model.network.mask_decoder.distance_mask
        else:
            self.encoder = Encoder(cfg)

        module = importlib.import_module('hybridpc.model.module')
        decoder_class = getattr(module, self.decoder_type)
        self.udf_decoder = decoder_class(
            decoder_cfg=cfg.model.network.udf_decoder,
            supervision = cfg.data.supervision,
            latent_dim=cfg.model.network.latent_dim,
            feature_dim=cfg.model.network.udf_decoder.feature_dim,
            hidden_dim=cfg.model.network.udf_decoder.hidden_dim,
            out_dim=1,
            voxel_size=cfg.data.voxel_size,
            activation=cfg.model.network.udf_decoder.activation
        )

        # if self.hparams.model.network.mask_decoder.distance_mask:
        self.mask_decoder = decoder_class(
            decoder_cfg=cfg.model.network.mask_decoder,
            supervision = 'Distance',
            latent_dim=cfg.model.network.latent_dim,
            feature_dim=cfg.model.network.mask_decoder.feature_dim,
            hidden_dim=cfg.model.network.mask_decoder.hidden_dim,
            out_dim=1,
            voxel_size=cfg.data.voxel_size,
            activation=cfg.model.network.mask_decoder.activation
        )

        self.dense_generator = Dense_Generator(
            self.udf_decoder,
            cfg.data.voxel_size,
            cfg.model.dense_generator.num_steps,
            cfg.model.dense_generator.num_points,
            cfg.model.dense_generator.threshold,
            cfg.model.dense_generator.filter_val,
            cfg.model.dense_generator.type
        )

        self.kernel_generator = KernelGenerator(kernel_size=cfg.model.network.udf_decoder.kernel_size,
                                                stride=1,
                                                dilation=1,
                                                dimension=3)
        self.batched_sampler = BatchedSampler(cfg)  # Instantiate the batched sampler with the configurations

    # def forward(self, data_dict):
    #     outputs = {}
    #     query_xyz, query_gt_sdf = self.batched_sampler.batch_sdf_sample(data_dict)
    #     voxel_centers = data_dict['voxel_coords'][:, 1:4] * self.hparams.data.voxel_size + self.hparams.data.voxel_size / 2.0
    #     knn_output = knn_points(query_xyz.unsqueeze(0).to(torch.device("cuda")),
    #                             voxel_centers.unsqueeze(0).to(torch.device("cuda")),
    #                             K=1)
    #     indices = knn_output.idx.squeeze(0)
    #         # else:
    #     outputs['gt_values'] = query_gt_sdf
    #     on_surface_xyz, _ = self.batched_sampler.batch_on_surface_sample(data_dict)
    #     voxel_centers = data_dict['voxel_coords'][:, 1:4] * self.hparams.data.voxel_size + self.hparams.data.voxel_size / 2.0
    #     knn_output = knn_points(on_surface_xyz.unsqueeze(0).to(torch.device("cuda")),
    #                             voxel_centers.unsqueeze(0).to(torch.device("cuda")),
    #                             K=1)
    #     on_surface_indices = knn_output.idx.squeeze(0)

    #     if self.mask:
    #         mask_query_xyz, mask_query_gt_udf = self.batched_sampler.batch_udf_sample(data_dict)
    #         outputs['gt_distances'] = mask_query_gt_udf
    #     encoder_output = self.encoder(data_dict)
    #     if "Interpolated" in self.decoder_type and self.hparams.model.network.udf_decoder.neighbor_type == "cm":
    #         query_indices = self.cm_neighbors(encoder_output, data_dict) # neighbor_indices
    #         # self.visualize_neighbors(in_data['absolute_coords'].cpu(), neighbor_indices[in_data['indices']].cpu(), in_data['indices'].cpu())   
    #         outputs['values'] = self.udf_decoder(encoder_output.F, data_dict['voxel_coords'], query_xyz, query_indices)  
    #     elif self.decoder_type == "InterpolatedDecoder":
    #         outputs['values'] = self.udf_decoder(encoder_output.F, data_dict['voxel_coords'], query_xyz, indices)
    #         outputs['surface_values'] = self.udf_decoder(encoder_output.F, data_dict['voxel_coords'], on_surface_xyz, on_surface_indices)

    #     elif self.decoder_type == "MultiScaleInterpolatedDecoder" or self.decoder_type == "LargeDecoder":
    #         outputs['values'] = self.udf_decoder(encoder_output, query_xyz)
    #         outputs['surface_values'] = self.udf_decoder(encoder_output, on_surface_xyz)

    #     else:
    #         outputs['values'] = self.udf_decoder(encoder_output.F, data_dict['query_relative_coords'], data_dict['query_indices'])

    #     if self.eikonal:
    #         if self.hparams.model.network.grad_type == "Numerical":
    #             interval = 0.01 * self.hparams.data.voxel_size
    #             grad_value = []
    #             for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
    #                 offset_tensor = torch.tensor(offset, device=self.device)[None, :]
    #                 res_p = self.udf_decoder(encoder_output, query_xyz + offset_tensor)
    #                 res_n = self.udf_decoder(encoder_output, query_xyz - offset_tensor)
    #                 grad_value.append((res_p - res_n) / (2 * interval))
    #             outputs['grad_value'] = torch.stack(grad_value, dim=1)
    #         else:
    #             xyz = torch.clone(query_xyz)
    #             xyz.requires_grad = True
    #             with torch.enable_grad():
    #                 res = self.udf_decoder(encoder_output, xyz)
    #                 # res = self.udf_decoder(encoder_output.F, data_dict['voxel_coords'], query_xyz, indices) 
    #                 outputs['grad_value'] = torch.autograd.grad(res, [xyz],
    #                                                     grad_outputs=torch.ones_like(res),
    #                                                     create_graph=self.udf_decoder.training, allow_unused=True)[0]
                    
    #     if self.mask:
    #         outputs['distances'] = self.mask_decoder(encoder_output, mask_query_xyz)
    #         # outputs['distances'] = outputs['gt_distances']
    #         # outputs['grad_value'] = outputs['gt_values']

    #     return outputs, encoder_output
    def forward(self, data_dict):
        outputs = {}
        query_xyz, query_gt_sdf = self.batched_sampler.batch_sdf_sample(data_dict)
        outputs['gt_values'] = query_gt_sdf
        on_surface_xyz, gt_on_surface_normal = self.batched_sampler.batch_on_surface_sample(data_dict)
        outputs['gt_on_surface_normal'] = gt_on_surface_normal

        # if self.mask:
        mask_query_xyz, mask_query_gt_udf = self.batched_sampler.batch_udf_sample(data_dict)
        outputs['gt_distances'] = mask_query_gt_udf

        if self.backbone == "PointTransformerV3":
            pt_data = {}
            # pt_data['grid_coord'] = data_dict['voxel_coords'][:, 1:4]
            if self.hparams.model.network.encoder.input_splat: 
                pt_data['coord'], pt_data['feat'], pt_data['batch'] = self.pt_input_splat(data_dict)
                pt_data['grid_size'] = 0.01
            else:
                pt_data['feat'] = data_dict['point_features']
                # batch_ids = torch.cat([torch.full((self.hparams.data.num_input_points, ), i) for i, n in enumerate(data_dict['voxel_nums'])]).to(data_dict['voxel_features'].device)
                pt_data['offset'] = torch.cumsum(data_dict['xyz_splits'], dim=0)
                pt_data['grid_size'] = 0.01
                pt_data['coord'] = data_dict['xyz']
            encoder_output = self.point_transformer(pt_data)
        else: 
            encoder_output = self.encoder(data_dict)
        outputs['values'], *_ = self.udf_decoder(encoder_output, query_xyz)
        outputs['surface_values'], *_ = self.udf_decoder(encoder_output, on_surface_xyz)

        if self.eikonal:
            if self.hparams.model.network.grad_type == "Numerical":
                interval = 0.01 * self.hparams.data.voxel_size
                grad_value = []
                for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
                    offset_tensor = torch.tensor(offset, device=self.device)[None, :]
                    res_p, *_ = self.udf_decoder(encoder_output, query_xyz + offset_tensor)
                    res_n, *_ = self.udf_decoder(encoder_output, query_xyz - offset_tensor)
                    grad_value.append((res_p - res_n) / (2 * interval))
                outputs['pd_grad'] = torch.stack(grad_value, dim=1)
            else:
                xyz = torch.clone(query_xyz)
                xyz.requires_grad = True
                with torch.enable_grad():
                    res, *_ = self.udf_decoder(encoder_output, xyz)
                    # res = self.udf_decoder(encoder_output.F, data_dict['voxel_coords'], query_xyz, indices) 
                    outputs['pd_grad'] = torch.autograd.grad(res, [xyz],
                                                        grad_outputs=torch.ones_like(res),
                                                        create_graph=self.udf_decoder.training, allow_unused=True)[0]
                    
        if self.surface_normal_supervision:
            if self.hparams.model.network.grad_type == "Numerical":
                interval = 0.01 * self.hparams.data.voxel_size
                grad_value = []
                for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
                    offset_tensor = torch.tensor(offset, device=self.device)[None, :]
                    res_p, *_ = self.udf_decoder(encoder_output, on_surface_xyz + offset_tensor)
                    res_n, *_ = self.udf_decoder(encoder_output, on_surface_xyz - offset_tensor)
                    grad_value.append((res_p - res_n) / (2 * interval))
                outputs['pd_surface_grad'] = torch.stack(grad_value, dim=1)
            else:
                xyz = torch.clone(on_surface_xyz)
                xyz.requires_grad = True
                with torch.enable_grad():
                    res, *_ = self.udf_decoder(encoder_output, xyz)
                    # res = self.udf_decoder(encoder_output.F, data_dict['voxel_coords'], query_xyz, indices) 
                    outputs['pd_surface_grad'] = torch.autograd.grad(res, [xyz],
                                                        grad_outputs=torch.ones_like(res),
                                                        create_graph=self.udf_decoder.training, allow_unused=True)[0]
                                        
        # if self.mask:
        outputs['distances'], *_ = self.mask_decoder(encoder_output, mask_query_xyz)

        return outputs, encoder_output
    
    def loss(self, data_dict, outputs, encoder_output):
        l1_loss = torch.nn.L1Loss(reduction='mean')(torch.clamp(outputs['values'], min = -self.hparams.data.supervision.sdf.max_dist, max=self.hparams.data.supervision.sdf.max_dist), torch.clamp(outputs['gt_values'], min = -self.hparams.data.supervision.sdf.max_dist, max=self.hparams.data.supervision.sdf.max_dist))
        on_surface_loss = torch.abs(outputs['surface_values']).mean()

        mask_loss = 0
        eikonal_loss = 0
        normal_loss = 0
        
        # Eikonal Loss computation
        if self.eikonal:
            norms = torch.norm(outputs['pd_grad'], dim=1)  # Compute the norm over the gradient vectors
            eikonal_loss = torch.mean((norms - 1) ** 2)  # Eikonal loss formula

        if self.surface_normal_supervision:
            normalized_pd_surface_grad = -outputs['pd_surface_grad'] / (torch.linalg.norm(outputs['pd_surface_grad'], dim=-1, keepdim=True) + 1.0e-6)
            normal_loss = 1.0 - torch.sum(normalized_pd_surface_grad * outputs['gt_on_surface_normal'], dim=-1).mean()

        # if self.mask:
        mask_loss = torch.nn.L1Loss(reduction='mean')(torch.clamp(outputs['distances'], max=self.hparams.data.supervision.udf.max_dist), torch.clamp(outputs['gt_distances'], max=self.hparams.data.supervision.udf.max_dist))

        return l1_loss, on_surface_loss, mask_loss, eikonal_loss, normal_loss

    def log_visualizations(self, batch, encodes_dict):
        if self.trainer.logger is None:
            return
        # with torch.no_grad():         
            # if not self.hparams.no_mesh_vis:
        mesh, voxel_centers = self.dense_generator.generate_mesh(batch, encodes_dict)
        self.log_geometry("pd_mesh", mesh)

    def pt_input_splat(self, data_dict):
        batch_size = len(data_dict['voxel_nums'])
        points_per_batch = self.hparams.data.num_input_points
        batch_voxel_coords = []
        batch_indices = []
        batch_splat_point_features = []
        for b in range(batch_size):
            batch_start_idx = b * points_per_batch
            batch_end_idx = (b + 1) * points_per_batch
            xyz = data_dict['xyz'][batch_start_idx:batch_end_idx]
            point_features = data_dict['point_features'][batch_start_idx:batch_end_idx]
            svh = SparseFeatureHierarchy(
                voxel_size=self.hparams.data.voxel_size,
                depth=4,
                device=xyz.device
            )
            svh.build_point_splatting(xyz)
            grid = svh.grids[0]
            # Get voxel idx
            xyz_grid = grid.world_to_grid(xyz)
            indices = grid.ijk_to_index(xyz_grid.round().int())
            voxel_coords =  grid.active_grid_coords()
            voxel_coords = voxel_coords[:torch.max(indices).item() + 1]
            batch_voxel_coords.append(voxel_coords)
            splat_point_features = torch.zeros(len(voxel_coords), point_features.shape[1], device=point_features.device)
            splat_point_features[indices] = point_features
            batch_splat_point_features.append(splat_point_features)


        batch_voxel_coords = ME.utils.batched_coordinates(batch_voxel_coords, dtype=data_dict['xyz'].dtype, device=data_dict['xyz'].device)
        batch_voxel_centers = batch_voxel_coords[:, 1:4] * self.hparams.data.voxel_size + self.hparams.data.voxel_size / 2.0
        batch_ids = batch_voxel_coords[:, 0].int()
        batch_splat_point_features = torch.cat(batch_splat_point_features, dim=0)

        return batch_voxel_centers, batch_splat_point_features, batch_ids

    def cm_neighbors(self, encoder_output, data_dict):
        """ compute neighbor indices by minkowski cooridnate manager"""
        cm = encoder_output.coordinate_manager
        in_key = encoder_output.coordinate_key
        out_key = cm.stride(in_key, self.kernel_generator.kernel_stride)
        region_type, region_offset, _ = self.kernel_generator.get_kernel(encoder_output.tensor_stride, False)
        kernel_map = cm.kernel_map(in_key,
                                   out_key,
                                   self.kernel_generator.kernel_stride,
                                   self.kernel_generator.kernel_size,
                                   self.kernel_generator.kernel_dilation,
                                   region_type=region_type,
                                   region_offset=region_offset) #kernel size 3, stride 1, dilation 1
        neighbors = torch.full((encoder_output.shape[0], 8), -1, dtype=torch.long).to(encoder_output.device)
        for key in kernel_map.keys():
            in_out = kernel_map[key].long()
            neighbors[in_out[0], key] = in_out[1]
        
        neighbor_indices = neighbors[data_dict['query_indices']] #N, K
        mask = data_dict['query_indices'] == -1
        neighbor_indices[mask] = -1
        
        return neighbor_indices


    # def on_train_start(self):
        # Assuming you have only one optimizer
        # self.trainer.optimizers[0].param_groups[0]['capturable'] = True
    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     udf_norms = grad_norm(self.udf_decoder, norm_type=2)
    #     mask_norms = grad_norm(self.mask_decoder, norm_type=2)
    #     encoder_norms = grad_norm(self.encoder, norm_type=2)
    #     self.log_dict("train/udf_norms", udf_norms)
    #     self.log_dict("train/mask_norms", mask_norms)
    #     self.log_dict('train/encoder_norms', encoder_norms)

    def training_step(self, data_dict):
        """ UDF auto-encoder training stage """
        # self.zero_grad()
        # opt = self.optimizers()
        # opt.zero_grad()
        batch_size = self.hparams.data.batch_size
        outputs, encoder_output = self.forward(data_dict)
        # outputs['values'] = outputs['values'].detach()
        # outputs['surface_values'] = outputs['surface_values'].detach()
        # outputs['grad_value'] = outputs['grad_value'].detach()

        l1_loss, on_surface_loss, mask_loss, eikonal_loss, normal_loss = self.loss(data_dict, outputs, encoder_output)
        self.log("train/l1_loss", l1_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/on_surface_loss", on_surface_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/mask_loss", mask_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/eikonal_loss", eikonal_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/normal_loss", normal_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

        total_loss = l1_loss*self.hparams.data.supervision.sdf.weight + on_surface_loss*self.hparams.data.supervision.on_surface.weight + eikonal_loss*self.hparams.data.supervision.eikonal.weight + mask_loss*self.hparams.data.supervision.udf.weight + normal_loss*self.hparams.data.supervision.on_surface.normal_weight
        # total_loss = l1_loss*0 + on_surface_loss*0 + eikonal_loss*0 + mask_loss
        # total_loss = 0

        # total_loss = mask_loss*self.hparams.data.supervision.udf.weight
        # self.manual_backward(udf_loss)
        # opt.step()
                # Visualize computation graph
        # if self.current_epoch == 0 and self.global_step == 0:
        #     dot = make_dot(total_loss, params=dict(self.named_parameters()))
        #     dot.format = 'png'
        #     dot.render('computation_graph_detach_unused')

        return total_loss

    def validation_step(self, data_dict, idx):
        # self.eval()
        # torch.set_grad_enabled(False)
        batch_size = 1
        outputs, encoder_output = self.forward(data_dict)
        l1_loss, on_surface_loss, mask_loss, eikonal_loss, normal_loss = self.loss(data_dict, outputs, encoder_output)

        self.log("val/l1_loss",  l1_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log("val/on_surface_loss", on_surface_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log("val/mask_loss", mask_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log("val/eikonal_loss", eikonal_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log("val/normal_loss", normal_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        if  self.current_epoch > self.hparams.model.network.prepare_epochs and self.current_epoch % self.hparams.model.network.eval_CD_frequency == 0:
            gt_pointcloud = data_dict['all_xyz'].cpu().numpy()
            input_pointcloud = data_dict['xyz'].cpu().numpy()        
            if self.hparams.model.network.eval_algorithm == 'DMC':
                # dense_pointcloud, duration = dense_generator.generate_point_cloud(batch, encoder_outputs, device)
                dmc_mesh = self.dense_generator.generate_dual_mc_mesh(data_dict, encoder_output, self.device)
                # Evaluate the reconstructed mesh
                evaluator = UnitMeshEvaluator(n_points=100000, metric_names=UnitMeshEvaluator.ESSENTIAL_METRICS)
                eval_dict, translation, scale = evaluator.eval_mesh(dmc_mesh, data_dict['all_xyz'], None, onet_samples=None)
                # pc_eval_dict = eval_pointcloud(dense_pointcloud, gt_pointcloud, input_pointcloud)
            elif self.hparams.model.network.eval_algorithm == 'MC':
                mesh, voxel_centers = self.dense_generator.generate_mesh(data_dict, encoder_output, self.device)
                dense_pointcloud, duration = self.dense_generator.generate_point_cloud(data_dict, encoder_output, self.device)
                eval_dict = eval_pointcloud(dense_pointcloud, gt_pointcloud, input_pointcloud)

            else:
                dense_pointcloud, duration = self.dense_generator.generate_point_cloud(data_dict, encoder_output, self.device)
                eval_dict = eval_pointcloud(dense_pointcloud, gt_pointcloud, input_pointcloud)
            
            # Log the metrics
            self.log("val/chamfer_L1", eval_dict['chamfer-L1'], on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/accuracy", eval_dict['accuracy'], on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/F-Score (1.0%)", eval_dict['f-score-10'], on_step=False, on_epoch=True, sync_dist=True)

            # if self.hparams.model.inference.visualization:
            #     visualize_tool(dense_pointcloud, gt_pointcloud, input_pointcloud, data_dict['scene_names'])
            # if self.hparams.model.inference.log_visualization:
            #     self.log_visualizations(data_dict, encoder_output)
            # return eval_dict
