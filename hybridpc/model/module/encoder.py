import functools
import torch.nn as nn
import pytorch_lightning as pl
import MinkowskiEngine as ME
import math
import torch
from hybridpc.model.module.common import ResidualBlock, UBlock, ResNetBase, BasicBlock, Bottleneck
from pycarus.geometry.pcd import knn



class Encoder(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()

        self.use_unet = cfg.model.network.encoder.use_unet
        self.voxel_size_out = cfg.model.network.encoder.voxel_size_out
        self.voxel_size_in = cfg.data.voxel_size
        # Compute the downsample_steps based on voxel sizes
        downsample_steps = int(math.log2(cfg.model.network.encoder.voxel_size_out / cfg.data.voxel_size))

        self.feature_in = cfg.model.network.encoder.feature_in # random_latent/ voxel_features
        self.training_stage = cfg.model.training_stage
        # if self.feature_in == "random_latent":
        #     input_channels = cfg.model.network.latent_dim
        # else: 
        in_channels = cfg.model.network.use_xyz * 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3
        channels = [in_channels * (2 ** k) for k in range(0, downsample_steps+1)]
        out_channels = in_channels * (2 ** downsample_steps) # Compute the output channels dynamically   

        sp_norm = functools.partial(ME.MinkowskiBatchNorm)
        #1. Downsampler
        self.blocks = []
        for i in range(downsample_steps):
            in_c, out_c = channels[i], channels[i+1]
            
            # Replace ConvLayer with MinkowskiConvolution
            self.blocks.append( ME.MinkowskiConvolution(in_channels=in_c,  out_channels=out_c, kernel_size=2, stride=2, dimension=3) )
            self.blocks.append( ME.MinkowskiConvolution(in_channels=out_c, out_channels=out_c, kernel_size=1, stride=1, dimension=3) )

        self.downsamplers = nn.Sequential(*self.blocks)

        if self.use_unet:
            # 2. Unet
            self.unet = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=out_channels, out_channels=cfg.model.network.latent_dim, kernel_size=3, dimension=3),
                UBlock([cfg.model.network.latent_dim * c for c in cfg.model.network.encoder.blocks], sp_norm, cfg.model.network.encoder.block_reps, ResidualBlock),
                sp_norm(cfg.model.network.latent_dim),
                ME.MinkowskiReLU(inplace=True)
            )

    def recompute_udf_queries(self, voxel_coords, voxel_size_in, voxel_size_out, query_points, values):
        voxel_center = voxel_coords * voxel_size_in + voxel_size_in / 2.0 # compute voxel_center in original coordinate system
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

    def forward(self, data_dict):

        # data = {
        #     "xyz": scene['xyz'],  # N, 3
        #     "points": scene['points'],  # N, 3
        #     "labels": scene['labels'],  # N,
        #     "voxel_indices": inds_reconstruct,  # N,
        #     "voxel_coords": voxel_coords,  # K, 3
        #     "voxel_features": feats,  # K, ?
        #     "query_points": scene['query_points'],  # M, 3
        #     "query_voxel_indices": scene['query_voxel_indices'],  # M,
        #     "values": scene['values'],  # M,
        #     "scene_name": scene['scene_name']
        # }
        output_dict = {}
        x = ME.SparseTensor(features=data_dict['voxel_features'], coordinates=data_dict['voxel_coords'])
        x = self.downsamplers(x)
        if self.use_unet:
            x = self.unet(x)
        
        # if self.training_stage == 1: # compute query points indices
        query_relative_coords, query_indices, values = self.recompute_udf_queries(x.C[:, 1:4], self.voxel_size_in, self.voxel_size_out, data_dict['absolute_query_points'], data_dict['unmasked_values'])
        output_dict['values'] = values
        # else: # compute original points indices
        relative_coords, indices = self.recompute_xyz_queries(x.C[:, 1:4], self.voxel_size_in, self.voxel_size_out, data_dict['xyz'])
        output_dict['query_relative_coords'] = query_relative_coords
        output_dict['query_indices'] = query_indices
        output_dict['relative_coords'] = relative_coords
        output_dict['indices'] = indices
        output_dict['latent_codes'] = x.F
        output_dict['voxel_coords'] = x.C

        return output_dict