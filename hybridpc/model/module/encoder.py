import functools
import torch.nn as nn
import pytorch_lightning as pl
import MinkowskiEngine as ME
import math
import torch
from hybridpc.model.module.common import ResidualBlock, UBlock, MultiScaleUBlock, ResNetBase, BasicBlock, Bottleneck, LocalPointNet, MultiScaleEncoderUBlock
import open3d as o3d
import numpy as np
from pytorch3d.ops import knn_points
from nksr.svh import SparseFeatureHierarchy


class Encoder(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()

        self.voxel_size = cfg.data.voxel_size
        self.input_splat = cfg.model.network.encoder.input_splat
        self.sp_norm = functools.partial(ME.MinkowskiBatchNorm)
        self.local_pointnet = LocalPointNet(3 + cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3, cfg.model.network.latent_dim, cfg.model.network.encoder.pn_hidden_dim, self.sp_norm, ResidualBlock, scatter_type='mean', n_blocks=cfg.model.network.encoder.pn_n_blocks)

        self.in_conv = ME.MinkowskiConvolution(in_channels=cfg.model.network.latent_dim, out_channels=cfg.model.network.latent_dim, kernel_size=3, dimension=3)
        if cfg.model.network.encoder.larger_unet:
            self.unet = MultiScaleUBlock([cfg.model.network.latent_dim * (2**(c-1)) for c in range(1, cfg.model.network.encoder.unet_blocks_num + 1)], self.sp_norm, cfg.model.network.encoder.unet_block_reps, ResidualBlock, cfg)
        else: 
            self.unet = MultiScaleUBlock([cfg.model.network.latent_dim * c for c in range(1, cfg.model.network.encoder.unet_blocks_num + 1)], self.sp_norm, cfg.model.network.encoder.unet_block_reps, ResidualBlock, cfg)

    def xyz_splat(self, data_dict):
        """ modify the data_dict to include splatted voxel_coords, relative_coords, and indices"""
        batch_size = len(data_dict['scene_names'])
        points_per_batch = len(data_dict['xyz']) // batch_size
        batch_voxel_coords = []
        batch_relative_coords = []
        batch_indices = []
        cumulative_voxel_coords_len = 0  # Keep track of the cumulative length
        cumulative_xyz_len = 0 
        for b in range(batch_size):
            if 'xyz_splits' in data_dict:
                batch_start_idx = cumulative_xyz_len
                batch_end_idx = cumulative_xyz_len + data_dict['xyz_splits'][b]
                cumulative_xyz_len += data_dict['xyz_splits'][b]
            else:
                batch_start_idx = b * points_per_batch
                batch_end_idx = (b + 1) * points_per_batch
            xyz = data_dict['xyz'][batch_start_idx:batch_end_idx]
            svh = SparseFeatureHierarchy(
                voxel_size=self.voxel_size,
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
            voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 
            relative_coords = xyz - voxel_center[indices]
            # relative_coords = xyz

            batch_indices.append(indices + cumulative_voxel_coords_len)
            cumulative_voxel_coords_len += len(voxel_coords)
            batch_voxel_coords.append(voxel_coords)
            batch_relative_coords.append(relative_coords)

        batch_voxel_coords = ME.utils.batched_coordinates(batch_voxel_coords, dtype=data_dict['xyz'].dtype, device=data_dict['xyz'].device)

        data_dict['relative_coords'] = torch.cat(batch_relative_coords, dim=0)
        data_dict['voxel_coords'] = batch_voxel_coords
        data_dict['indices'] = torch.cat(batch_indices, dim=0)


    def forward(self, data_dict):
        if self.input_splat:
            self.xyz_splat(data_dict)
        pn_feat = self.local_pointnet(torch.cat((data_dict['relative_coords'], data_dict['point_features']), dim=1), data_dict['indices'])
        x = ME.SparseTensor(pn_feat, coordinates=data_dict['voxel_coords'])
        x = self.in_conv(x)
        _, x = self.unet(x, data_dict)

        normalized_x = []
        for latent in x:
            latent = self.sp_norm(latent.F.shape[-1]).to(pn_feat.device)(latent)
            latent = ME.MinkowskiReLU(inplace=True)(latent)
            normalized_x.append(latent)

        return normalized_x
    