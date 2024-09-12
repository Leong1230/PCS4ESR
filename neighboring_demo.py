import functools
import torch
import torch.nn as nn
import pytorch_lightning as pl
import MinkowskiEngine as ME
import numpy as np
from collections import OrderedDict
from hybridpc.data.dataset.voxelizer import Voxelizer
from hybridpc.model.module.common import ResidualBlock, UBlock, MultiScaleUBlock, ResNetBase, BasicBlock, Bottleneck, LocalPointNet

def init_weights_to_one(m):
    if isinstance(m, ME.MinkowskiConvolution):
        with torch.no_grad():
            m.kernel.data.fill_(1.0)
    
# Encoder definition
class Encoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.sp_norm = functools.partial(ME.MinkowskiBatchNorm)
        self.unet = nn.Sequential(
            MultiScaleUBlock([cfg.model.network.latent_dim * c for c in range(1, cfg.model.network.encoder.unet_blocks_num + 1)], self.sp_norm, cfg.model.network.encoder.unet_block_reps, ResidualBlock),
        )
        self.first_conv = ME.MinkowskiConvolution(
            in_channels=1, 
            out_channels=1, 
            kernel_size=2, 
            stride=2, 
            dimension=3
        )
        self.second_conv = ME.MinkowskiConvolution(
            in_channels=1, 
            out_channels=1, 
            kernel_size=2, 
            stride=2, 
            dimension=3
        )
        self.first_conv.apply(init_weights_to_one)
        self.second_conv.apply(init_weights_to_one)


    def forward(self, voxel_coords, voxel_feats):
        x = ME.SparseTensor(voxel_feats, coordinates=voxel_coords)
        y = self.first_conv(x)
        z = self.second_conv(y)
        return x, y, z

def main():
    # Example usage
    voxel_size = 0.5
    voxelizer = Voxelizer(voxel_size)

    # Create a simple grid of points
    # xyz = np.array([[x, y, z] for x in range(5) for y in range(5) for z in range(1)], dtype=np.float32)
    values = [0.4, 1.6, 2.7, 3.8, 4.9]
    x, y = np.meshgrid(values, values)
    z = np.zeros_like(x)
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    point_features = np.ones((xyz.shape[0], 1), dtype=np.float32)  # Feature tensor with all ones
    point_features = point_features.reshape(5, 5)
    point_features[1, 1] = 10
    point_features = point_features.reshape(-1, 1)
    labels = np.zeros((xyz.shape[0],), dtype=np.int32)  # Dummy labels

    # Voxelize the points
    voxel_coords, voxel_feats, voxel_labels, indices = voxelizer.voxelize(xyz, point_features, labels)
    voxel_coords = torch.tensor(voxel_coords, dtype=torch.int32)
    voxel_coords = ME.utils.batched_coordinates(
    [voxel_coords], dtype=voxel_coords.dtype, 
    device=voxel_coords.device)  
    voxel_feats=torch.tensor(voxel_feats, dtype=torch.float32)

    query_xyz = np.array([
        [0.1, 0.1, 0],
        [1.1, 1.1, 0],
        [2.7, 2.7, 0],
        [5.1, 5.1, 0]
    ], dtype=np.float32)

    # Define dummy point features and labels
    # point_features = np.ones((query_points.shape[0], 3), dtype=np.float32)  # Feature tensor with all ones
    # labels = np.zeros((query_points.shape[0],), dtype=np.int32)  # Dummy labels

    # # Assuming you have a voxelizer class or function
    # # Replace `voxelizer` with your actual voxelizer implementation
    # query_voxel_coords, _, _, query_indices = voxelizer.voxelize(query_points, point_features, labels)
    # query_voxel_coords = torch.tensor(query_voxel_coords, dtype=torch.int32)
    # query_voxel_coords = ME.utils.batched_coordinates(
    # [query_voxel_coords], dtype=query_voxel_coords.dtype,
    # device=query_voxel_coords.device)

    # Configuration
    cfg = type('', (), {})()  # Create an empty config object
    cfg.model = type('', (), {})()
    cfg.model.network = type('', (), {})()
    cfg.model.network.encoder = type('', (), {})()
    cfg.model.network.encoder.pn_hidden_dim = 3
    cfg.model.network.latent_dim = 3
    cfg.model.network.encoder.unet_blocks_num = 3
    cfg.model.network.encoder.unet_block_reps = 1

    # Create encoder
    encoder = Encoder(cfg)

    # Forward pass
    x, y, z = encoder(voxel_coords, voxel_feats)
    query_voxel_coords, index, inverse_index = ME.utils.sparse_quantize(coordinates=query_xyz, return_index=True, return_inverse=True, quantization_size=voxel_size)
    query_voxel_coords = ME.utils.batched_coordinates(
    [query_voxel_coords], dtype=query_voxel_coords.dtype, 
    device=query_voxel_coords.device)  
    query_sp = ME.SparseTensor(features = query_voxel_coords, coordinates=query_voxel_coords, tensor_stride = 4)
    sparse_tensor = z
    cm = sparse_tensor.coordinate_manager
    source_key = sparse_tensor.coordinate_map_key
    query_key = query_sp.coordinate_map_key
    kernel_map = cm.kernel_map(
        source_key,
        query_key,
        kernel_size=1,
        region_type=ME.RegionType.HYPER_CUBE, # default 0 hypercube
    )  
    neighbor_idx = torch.full((query_voxel_coords.shape[0], 1), -1, dtype=torch.long)
    for key in kernel_map.keys():
        in_out = kernel_map[key].long()
        neighbor_idx[in_out[0], key] = in_out[1]
    
    print(neighbor_idx)
    print(y)

if __name__ == "__main__":
    main()
