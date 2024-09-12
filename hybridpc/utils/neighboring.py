import MinkowskiEngine as ME
import torch

import torch.nn.functional as F

import torch_scatter
import numpy as np

from voxelization_utils import sparse_quantize 

#

def world2grid(pts_w, transform_voxel, batch_idx=None):
    '''
    '''
    pts_w = F.pad(pts_w, pad=(0, 1), mode='constant', value=1.0) # nx4
    pts_grid = pts_w @ transform_voxel.transpose(-2, -1) 
    return pts_grid[..., :3]    

def grid2world(pts_grid, transform_voxel):
    '''
    '''
    pts_grid = F.pad(pts_grid, pad=(0, 1), mode='constant', value=1.0) # nx4
    transform_voxel_inverse = torch.inverse(transform_voxel)
    pts_world = pts_grid @ transform_voxel_inverse.transpose(-2, -1) 
    return pts_world[..., :3]

def grid2world_batched(pts_grid_batched, transform_voxel):
    '''
    '''
    return torch.cat(
        [grid2world(pts_grid.float(), transform_voxel[i]) \
         for i, pts_grid in enumerate(pts_grid_batched)], 0)

def voxelize(coords, feats, voxel_size=0.02):
    assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]

    voxelization_matrix = np.eye(4)
    scale = 1 / voxel_size
    np.fill_diagonal(voxelization_matrix[:3, :3], scale)
    # Apply transformations
    rigid_transformation = voxelization_matrix  

    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()
    homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
    coords_aug = homo_coords @ rigid_transformation.T[:, :3]

    # Align all coordinates to the origin.
    min_coords = coords_aug.min(0)
    M_t = np.eye(4)
    M_t[:3, -1] = -min_coords
    rigid_transformation = M_t @ rigid_transformation
    coords_aug = coords_aug - min_coords
    coords_aug = np.floor(coords_aug)
    inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True)
    coords_aug, feats = coords_aug[inds], feats[inds]
    return coords_aug, feats, rigid_transformation

def mink_neighbor(sparse_tensor, query_loc_batched, kernel_size=3):
    query_loc_batched_int = torch.round(query_loc_batched).int()
    cm = sparse_tensor.coordinate_manager
    query_key, (unique_map, inverse_map)= cm.insert_and_map(
            query_loc_batched_int,
            tensor_stride=1,
            string_id='query',
        ) # be cautious when batch size is larger than one
    source_key = sparse_tensor.coordinate_map_key
    kernel_map = cm.kernel_map(
        source_key,
        query_key,
        kernel_size=kernel_size,
        region_type=ME.RegionType.HYPER_CUBE, # default 0 hypercube
    )   
    # Inverse
    idx = torch.full(
        [unique_map.shape[0], kernel_size**3], -1, 
        dtype=torch.int32, device=unique_map.device)
    idx_ori = idx[inverse_map.long()] if len(inverse_map) > 0 else idx # one-to-one correspondence.
    mask = idx_ori > 0
    kernel_map_ori = torch.stack([idx_ori[mask], torch.nonzero(mask)[:, 0]], dim=0)
    return kernel_map_ori
    

    
    

# 
def mink_interpolate():
    pass


def demo_interpolation():

    pts = torch.rand((20, 3)).to(torch.float32) # bxnx3 
    feats = pts 
    # 
    coords_vox, _, pts_transform = voxelize(pts, feats, voxel_size=0.01)
    coords_vox = torch.tensor(coords_vox, dtype=torch.float32)  # Use appropriate dtype
    pts_transform = torch.tensor(pts_transform, dtype=torch.float32)  # Use appropriate dtype

    # coords_vox = torch.from_numpy(coords_vox).int().cuda()
    coords = ME.utils.batched_coordinates(
        [coords_vox], dtype=coords_vox.dtype, 
        device=coords_vox.device)  

    # source sparse tensor and query point 
    sparse_tensor = ME.SparseTensor(
        features=coords.float(), coordinates=coords) 

    # transform query point into voxel coordinate system
    b = 1
    query_point = pts[:10].reshape(b, -1, 3)
    query_points_grid = world2grid(
        query_point, pts_transform)
    query_points_grid = query_points_grid.to(torch.float32)
    query_loc_batched = ME.utils.batched_coordinates(
        torch.unbind(query_points_grid, dim=0), dtype=query_points_grid.dtype, 
        device=query_points_grid.device) 
    query_loc_w_batched = ME.utils.batched_coordinates(
        torch.unbind(query_point, dim=0), dtype=query_points_grid.dtype, 
        device=query_points_grid.device) 
    kernel_map = mink_neighbor(sparse_tensor, query_loc_batched)    

    source_feat = sparse_tensor.features
    # source_feat, source_coords_w_offset = source_feat[:, :-3], source_feat[:, -3:] 
    pts_rel_feat = source_feat[kernel_map[0]]
    # source_coords_w_offset = torch.sigmoid(source_coords_w_offset) * scale_size * extra_inputs['voxel_size']
    # print(source_coords_w_offset.min(), source_coords_w_offset.max())
    # source_coords_w_ = source_coords_w_ + torch.zeros_like(source_coords_w_offset) 
    source_loc_w = grid2world_batched(sparse_tensor.decomposed_coordinates, pts_transform)
    pts_rel_loc = query_loc_w_batched[kernel_map[1]][:, 1:] - source_loc_w[kernel_map[0]] 
    
    # TODO aggregation module.


if __name__ == '__main__':
    demo_interpolation()
    