import functools
import torch.nn as nn
import pytorch_lightning as pl
import MinkowskiEngine as ME
import math
import torch
from hybridpc.model.module.common import ResidualBlock, UBlock, ResNetBase, BasicBlock, Bottleneck, LocalPointNet
from pycarus.geometry.pcd import knn
import open3d as o3d
import numpy as np



class Encoder(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()

        self.use_unet = cfg.model.network.encoder.use_unet
        self.local_pointnet = cfg.model.network.encoder.local_pointnet
        self.down_sampler = cfg.model.network.encoder.down_sampler
        self.voxel_size_out = cfg.model.network.encoder.voxel_size_out
        self.voxel_size_in = cfg.data.voxel_size
        # Compute the downsample_steps based on voxel sizes
        downsample_steps = int(math.log2(cfg.model.network.encoder.voxel_size_out / cfg.data.voxel_size))

        self.feature_in = cfg.model.network.encoder.feature_in # random_latent/ voxel_features
        self.training_stage = cfg.model.training_stage
        # if self.feature_in == "random_latent":
        #     input_channels = cfg.model.network.latent_dim
        # else: 
        in_channels = cfg.model.network.encoder.pn_hidden_dim if self.local_pointnet else cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3
        channels = [in_channels * (2 ** k) for k in range(0, downsample_steps+1)]
        out_channels = in_channels * (2 ** downsample_steps) # Compute the output channels dynamically   
        sp_norm = functools.partial(ME.MinkowskiBatchNorm)

        #1. LocalPointNet
        if self.down_sampler == 'Conv':
            self.local_pointnet = LocalPointNet(3 + cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3, cfg.model.network.encoder.pn_hidden_dim, cfg.model.network.encoder.pn_hidden_dim, sp_norm, ResidualBlock, scatter_type='mean', n_blocks=cfg.model.network.encoder.pn_n_blocks)
        else:
            self.local_pointnet = LocalPointNet(3 + cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3, cfg.model.network.latent_dim, cfg.model.network.encoder.pn_hidden_dim, sp_norm, ResidualBlock, scatter_type='mean', n_blocks=cfg.model.network.encoder.pn_n_blocks)

        #2. Downsampler
        if self.down_sampler == 'Conv':
            self.blocks = []
            for i in range(downsample_steps):
                in_c, out_c = channels[i], channels[i+1]
                # in_c, out_c = cfg.model.network.latent_dim, cfg.model.network.latent_dim
                
                # Replace ConvLayer with MinkowskiConvolution
                self.blocks.append( ME.MinkowskiConvolution(in_channels=in_c,  out_channels=out_c, kernel_size=2, stride=2, expand_coordinates=True, dimension=3) )
                self.blocks.append( ME.MinkowskiConvolution(in_channels=out_c, out_channels=out_c, kernel_size=1, stride=1, expand_coordinates=True, dimension=3) )
            self.downsample_layers = nn.Sequential(*self.blocks)
        
        if self.down_sampler == 'MaxPool':
            #3. Pooling
            self.pooling_layers = []
            for i in range(downsample_steps):         
                self.pooling_layers.append( ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3) )
            self.downsample_layers = nn.Sequential(*self.pooling_layers)

        # 4. Unet
        if self.use_unet:
            if self.down_sampler == 'Conv':
                self.unet = nn.Sequential(
                    ME.MinkowskiConvolution(in_channels=out_channels, out_channels=cfg.model.network.latent_dim, kernel_size=3, dimension=3),
                    UBlock([cfg.model.network.latent_dim * c for c in cfg.model.network.encoder.unet_blocks], sp_norm, cfg.model.network.encoder.unet_block_reps, ResidualBlock),
                    sp_norm(cfg.model.network.latent_dim),
                    ME.MinkowskiReLU(inplace=True)
                )
            else:
                self.unet = nn.Sequential(
                    ME.MinkowskiConvolution(in_channels=cfg.model.network.latent_dim, out_channels=cfg.model.network.latent_dim, kernel_size=3, dimension=3),
                    UBlock([cfg.model.network.latent_dim * c for c in cfg.model.network.encoder.unet_blocks], sp_norm, cfg.model.network.encoder.unet_block_reps, ResidualBlock),
                    sp_norm(cfg.model.network.latent_dim),
                    ME.MinkowskiReLU(inplace=True)
                )

    def recompute_udf_queries(self, voxel_coords, voxel_size_in, voxel_size_out, query_points, values):
        voxel_center = voxel_coords * voxel_size_in + voxel_size_in / 2.0 # compute voxel_center in original coordinate system
        # voxel_center = voxel_coords * voxel_size_in + voxel_size_out /2.0# compute voxel_center in original coordinate system
        voxel_center = voxel_center.to(self.device)
        query_indices, _, _ = knn(query_points, voxel_center, 1)
        query_indices = query_indices[:, 0]
        query_relative_coords = query_points - voxel_center[query_indices]
        # remove outliers
        lower_bound = -voxel_size_out / 2
        upper_bound = voxel_size_out / 2
        # Create a mask
        mask = (query_relative_coords >= lower_bound) & (query_relative_coords <= upper_bound)
        # Reduce across the last dimension to get a (N, ) mask
        mask = torch.all(mask,-1)
        query_indices = query_indices[mask]
        query_relative_coords = query_relative_coords[mask]
        values = values[mask]

        return query_relative_coords, query_indices, values
    
    def recompute_xyz_queries(self, voxel_coords, voxel_size_in, voxel_size_out, query_points):
        voxel_center = voxel_coords * voxel_size_in + voxel_size_in / 2.0 # compute voxel_center in original coordinate system
        voxel_center = voxel_center.to(self.device)
        query_indices, _, _ = knn(query_points, voxel_center, 1)
        query_indices = query_indices[:, 0]
        query_relative_coords = query_points - voxel_center[query_indices]
        
        return query_relative_coords, query_indices

    def visualize_voxel(self, relative_coords, indices, query_relative_coords, query_indices, voxel_id):
        # Convert tensors to numpy arrays if they are not (assuming they are on CPU for simplicity)
        relative_coords = relative_coords.cpu().numpy()
        indices = indices.cpu().numpy()
        query_relative_coords = query_relative_coords.cpu().numpy()
        query_indices = query_indices.cpu().numpy()
        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()

        # Create a PointCloud for the query points
        query_pcd = o3d.geometry.PointCloud()

        # Filter the points that belong to the given voxel_id
        voxel_points = relative_coords[indices == voxel_id]
        query_voxel_points = query_relative_coords[query_indices == voxel_id]

        # Add points to the PointCloud object
        pcd.points = o3d.utility.Vector3dVector(voxel_points)
        query_pcd.points = o3d.utility.Vector3dVector(query_voxel_points)

        # Color the points in red
        pcd.paint_uniform_color([1, 0, 0])

        # Color the query points in green
        query_pcd.paint_uniform_color([0, 1, 0])

        # Combine the two point clouds
        combined_pcd = pcd + query_pcd

        # Visualize the point cloud
        o3d.visualization.draw_geometries([combined_pcd])


    def forward(self, data_dict):
        output_dict = {}
        if self.local_pointnet:
            # use local pointnet to extract voxel features
            # pn_feat = self.local_pointnet(torch.cat((data_dict['points'], data_dict['point_features']), dim=1), data_dict['voxel_indices'])
            pn_feat = self.local_pointnet(torch.cat((data_dict['points'][:, 0, :], data_dict['point_features']), dim=1), data_dict['voxel_indices'][:, 0])
            x = ME.SparseTensor(pn_feat, coordinates=data_dict['voxel_coords'])
        else:
            x = ME.SparseTensor(features=data_dict['voxel_features'], coordinates=data_dict['voxel_coords'])
        
        if self.down_sampler != 'None':
            x = self.downsample_layers(x)

        if self.use_unet:
            x = self.unet(x)
        
        if self.down_sampler !='None':
            query_relative_coords, query_indices, values = self.recompute_udf_queries(x.C[:, 1:4], self.voxel_size_in, self.voxel_size_out, data_dict['absolute_query_points'], data_dict['unmasked_values'])
            output_dict['values'] = values
            # compute original points indices
            relative_coords, indices = self.recompute_xyz_queries(x.C[:, 1:4], self.voxel_size_in, self.voxel_size_out, data_dict['xyz'])
        else:
            output_dict['values'] = data_dict['values']
            query_relative_coords = data_dict['query_points']
            query_indices = data_dict['query_voxel_indices']
            relative_coords = data_dict['points']
            indices = data_dict['voxel_indices']
        output_dict['query_relative_coords'] = query_relative_coords
        output_dict['query_indices'] = query_indices
        output_dict['relative_coords'] = relative_coords
        output_dict['indices'] = indices
        output_dict['latent_codes'] = x.F
        if self.down_sampler == 'None':
            output_dict['mixed_latent_codes'] = torch.cat((x.F, data_dict['voxel_features']), dim=1)
        output_dict['voxel_coords'] = x.C
        # Visualize voxel with ID = some_voxel_id
        # some_voxel_id = 5  # Replace with the voxel ID you are interested in
        # self.visualize_voxel(relative_coords, indices, query_relative_coords, query_indices, some_voxel_id)

        return output_dict