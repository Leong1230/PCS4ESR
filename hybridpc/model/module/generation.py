from pyexpat import features
import time
import os
import numpy as np
import copy
import math
import torchmetrics
# import nearest_neighbors
import torch
import torchviz
from pycg import vis, exp
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm
from typing import Callable, Tuple
from skimage.measure import marching_cubes
from sklearn.neighbors import NearestNeighbors
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
from pykdtree.kdtree import KDTree
from pytorch3d.ops import knn_points
import open3d as o3d
import pytorch_lightning as pl
from hybridpc.utils.samples import BatchedSampler


class MeshingResult:
    def __init__(self, v: torch.Tensor = None, f: torch.Tensor = None, c: torch.Tensor = None):
        self.v = v
        self.f = f
        self.c = c

class Dense_Generator(pl.LightningModule):
    def __init__(self, model, voxel_size, num_steps, num_points, threshold, filter_val, type='scene'):
        super().__init__()
        self.model = model # the model should be the UDF Decoder
        self.model.eval()
        self.voxel_size = voxel_size
        self.num_steps = num_steps
        self.num_points = num_points
        self.threshold = threshold
        self.filter_val = filter_val
        self.type = type # voxel or scene5
class Generator(pl.LightningModule):
    def __init__(self, model, mask_model, decoder_type, voxel_size, num_steps, num_points, threshold, filter_val, neighbor_type, k_neighbors, last_n_layers, reconstruction_cfg):
        super().__init__()
        self.model = model # the model should be the UDF Decoder
        self.mask_model = mask_model # the distance mask decoder
        self.rec_cfg = reconstruction_cfg
        # self.model.eval()
        self.decoder_type = decoder_type
        self.voxel_size = voxel_size
        self.threshold = threshold
        self.filter_val = filter_val
        self.neighbor_type = neighbor_type
        self.k_neighbors = k_neighbors
        self.last_n_layers = last_n_layers                                      
    

    def compute_gt_sdf_from_pts(self, gt_xyz, gt_normals, query_pos: torch.Tensor):
        k = 8  
        stdv = 0.02
        knn_output = knn_points(query_pos.unsqueeze(0).to(torch.device("cuda")), gt_xyz.unsqueeze(0).to(torch.device("cuda")), K=k)
        indices = knn_output.idx.squeeze(0)
        # dists, indices = self.kdtree.query(query_pos.detach().cpu().numpy(), k=k)
        indices = torch.tensor(indices, device=query_pos.device)
        closest_points = gt_xyz[indices]
        surface_to_queries_vec = query_pos.unsqueeze(1) - closest_points #N, K, 3

        dot_products = torch.einsum("ijk,ijk->ij", surface_to_queries_vec, gt_normals[indices]) #N, K
        vec_lengths = torch.norm(surface_to_queries_vec[:, 0, :], dim=-1) 
        use_dot_product = vec_lengths < stdv
        sdf = torch.where(use_dot_product, torch.abs(dot_products[:, 0]), vec_lengths)

        # Adjust the sign of the sdf values based on the majority of dot products
        num_pos = torch.sum(dot_products > 0, dim=1)
        inside = num_pos <= (k / 2)
        sdf[inside] *= -1
        
        return -sdf
    
    def generate_dual_mc_mesh(self, data_dict, encoder_outputs, device):
        from nksr.svh import SparseFeatureHierarchy, SparseIndexGrid
        from nksr.ext import meshing
        from nksr.meshing import MarchingCubes
        from nksr import utils

        max_depth = 100
        grid_upsample = 1
        max_points = -1
        mise_iter = 0
        knn_time = 0
        dmc_time = 0 
        attentive_time = 0
        interpolation_time = 0
        after_layers_time = 0
        decoder_time = 0
        grid_splat_time = 0
        mask_threshold = self.rec_cfg.mask_threshold

        self.build_on_splated_points = False
        # Initialize voxel center coordinates
        if self.build_on_splated_points:
            pts = data_dict['xyz'].detach()
        else:
            pts = data_dict['un_splats_xyz'].detach()
        self.last_n_layers = 4
        self.trim = self.rec_cfg.trim
        self.gt_mask = self.rec_cfg.gt_mask
        self.gt_sdf = self.rec_cfg.gt_sdf
        self.build_on_splated_points = self.rec_cfg.build_on_splated_points
        self.regular_grid = self.rec_cfg.regular_grid
        grid_splat_time -= time.time()

        # Generate nksr grid structure (single depth)
        nksr_svh = SparseFeatureHierarchy(
            voxel_size=self.voxel_size,
            depth=self.last_n_layers,
            device= pts.device
        )

        if self.regular_grid:
            resolution = 0.2  # Define the resolution of the grid
            distance_threshold = 1  # Define the distance threshold
            offset = 0.5

            min_xyz = torch.min(pts, dim=0).values
            max_xyz = torch.max(pts, dim=0).values
            # Apply the offset to the min and max coordinates
            min_xyz = min_xyz - offset
            max_xyz = max_xyz + offset
            # Step 2: Create a uniform grid of points within the bounding box
            x_range = torch.arange(min_xyz[0], max_xyz[0], resolution)
            y_range = torch.arange(min_xyz[1], max_xyz[1], resolution)
            z_range = torch.arange(min_xyz[2], max_xyz[2], resolution)
            xx, yy, zz = torch.meshgrid(x_range, y_range, z_range)
            grid_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(torch.device("cuda"))
            knn_output = knn_points(grid_points.unsqueeze(0).to(torch.device("cuda")),
                                    pts.unsqueeze(0).to(torch.device("cuda")),
                                    K=1)
            dist = knn_output.dists.squeeze(0).squeeze(-1)
            mask = dist <= distance_threshold
            masked_grid_points = grid_points[mask]
            nksr_svh.build_iterative_coarsening(masked_grid_points)
        else:
            nksr_svh.build_point_splatting(pts)


        flattened_grids = []
        for d in range(min(nksr_svh.depth, max_depth + 1)):
            f_grid = meshing.build_flattened_grid(
                nksr_svh.grids[d]._grid,
                nksr_svh.grids[d - 1]._grid if d > 0 else None,
                d != nksr_svh.depth - 1
            )
            if grid_upsample > 1:
                f_grid = f_grid.subdivided_grid(grid_upsample)
            flattened_grids.append(f_grid)

        dual_grid = meshing.build_joint_dual_grid(flattened_grids)
        dmc_graph = meshing.dual_cube_graph(flattened_grids, dual_grid)
        dmc_vertices = torch.cat([
            f_grid.grid_to_world(f_grid.active_grid_coords().float())
            for f_grid in flattened_grids if f_grid.num_voxels() > 0
        ], dim=0)
        del flattened_grids, dual_grid
        grid_splat_time += time.time()
        """ create a mask to trim spurious geometry """

        decoder_time -= time.time()
        dmc_value, sdf_knn_time, sdf_attentive_time, sdf_interpolation_time, sdf_after_layers_time, voxel_centers = self.model(encoder_outputs, dmc_vertices)
        decoder_time += time.time()
        knn_time += sdf_knn_time
        attentive_time += sdf_attentive_time
        interpolation_time += sdf_interpolation_time
        after_layers_time += sdf_after_layers_time
        if self.gt_sdf:
            if 'gt_geometry' in data_dict:
                ref_xyz, ref_normal, _ = data_dict['gt_geometry'][0].torch_attr()
            else:
                ref_xyz, ref_normal = data_dict['all_xyz'], data_dict['all_normals']
            dmc_value = self.compute_gt_sdf_from_pts(ref_xyz, ref_normal, dmc_vertices)

        colors = torch.zeros((dmc_value.shape[0], 3)).to(dmc_value.device)
        colors[dmc_value > 0] = torch.Tensor([1, 0, 0]).to(dmc_value.device)  # Red for positive values
        colors[dmc_value < 0] = torch.Tensor([0, 1, 0]).to(dmc_value.device)  # Green for negative values

        # Normalize the color intensity based on the magnitude of dmc_values
        max_abs_value = torch.max(torch.abs(dmc_value))
        colors *= torch.abs(dmc_value)[:, None] / max_abs_value

        for _ in range(mise_iter):
            cube_sign = dmc_value[dmc_graph] > 0
            cube_mask = ~torch.logical_or(torch.all(cube_sign, dim=1), torch.all(~cube_sign, dim=1))
            dmc_graph = dmc_graph[cube_mask]
            unq, dmc_graph = torch.unique(dmc_graph.view(-1), return_inverse=True)
            dmc_graph = dmc_graph.view(-1, 8)
            dmc_vertices = dmc_vertices[unq]
            dmc_graph, dmc_vertices = utils.subdivide_cube_indices(dmc_graph, dmc_vertices)
            dmc_value = torch.clamp(self.model(encoder_outputs, dmc_vertices.to(device)), max=self.threshold)

        dmc_time -= time.time()
        dual_v, dual_f = MarchingCubes().apply(dmc_graph, dmc_vertices, dmc_value)
        dmc_time += time.time()

        vert_mask = None
        if self.trim:
            if self.gt_mask: 
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(data_dict['all_xyz'].cpu().numpy())  # coords is an (N, 3) array
                dist, indx = nn.kneighbors(dual_v.detach().cpu().numpy())  # xyz is an (M, 3) array
                dist = torch.from_numpy(dist).to(dual_v.device).squeeze(-1)
                vert_mask = dist < mask_threshold
            else: 
                decoder_time -= time.time()
                dist, mask_knn_time, mask_attentive_time, mask_interpolation_time, mask_after_layers_time, _ = self.mask_model(encoder_outputs, dual_v.to(device))
                decoder_time += time.time()
                vert_mask = dist < mask_threshold
                knn_time += mask_knn_time
                attentive_time += mask_attentive_time
                interpolation_time += mask_interpolation_time
                after_layers_time += mask_after_layers_time
            dmc_time -= time.time()
            dual_v, dual_f = utils.apply_vertex_mask(dual_v, dual_f, vert_mask)
            dmc_time += time.time()

        dmc_time -= time.time()
        mesh_res =  MeshingResult(dual_v, dual_f, None)
        # del dual_v, dual_f
        mesh = vis.mesh(mesh_res.v, mesh_res.f)
        dmc_time += time.time()
        return mesh, knn_time, dmc_time, attentive_time, interpolation_time, after_layers_time, decoder_time, grid_splat_time, dmc_vertices.cpu().numpy(), dmc_value.cpu().numpy(), colors.cpu().numpy(), voxel_centers.cpu().numpy()