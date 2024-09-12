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

    def generate_point_cloud(self, data_dict, voxel_latents, voxel_id):
        if self.type == 'voxel':
            start = time.time()

            # freeze model parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # sample_num set to 200000
            sample_num = 200000

            # Initialize samples in CUDA device
            samples_cpu = np.zeros((0, 3))

            # Initialize samples and move to CUDA device
            samples = torch.rand(1, sample_num, 3).float().to(self.device) * self.voxel_size - self.voxel_size / 2 # make samples within voxel_size
            N = samples.shape[1]  # The number of samples
            index = torch.full((N,), voxel_id, dtype=torch.long, device=self.device)

            samples.requires_grad = True

            i = 0
            while len(samples_cpu) < self.num_points:
                # print('iteration', i)

                for j in range(self.num_steps):
                    # print('refinement', j)
                    df_pred = torch.clamp(self.model(voxel_latents.detach(), samples[0].unsqueeze(1), samples[0].unsqueeze(1), index.unsqueeze(1)), max=self.threshold).unsqueeze(0)
                    df_pred.sum().backward(retain_graph=True)
                    gradient = samples.grad.detach()
                    samples = samples.detach()
                    df_pred = df_pred.detach()
                    samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
                    samples = samples.detach()
                    samples.requires_grad = True

                # print('finished refinement')

                if not i == 0:
                    # Move samples to CPU, detach from computation graph, convert to numpy array, and stack to samples_cpu
                    samples_cpu = np.vstack((samples_cpu, samples[df_pred < self.filter_val].detach().cpu().numpy()))

                samples = samples[df_pred < 0.03].unsqueeze(0)
                indices = torch.randint(samples.shape[1], (1, sample_num))
                samples = samples[[[0, ] * sample_num], indices]
                samples += (self.threshold / 3) * torch.randn(samples.shape).to(self.device)  # 3 sigma rule
                samples = samples.detach()
                samples.requires_grad = True

                i += 1
                print(samples_cpu.shape)

            duration = time.time() - start
            return samples_cpu, duration

        if self.type == 'multiple_voxels':
            start = time.time()

            # freeze model parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # sample_num set to 200000
            sample_num = 200000

            # Initialize samples in CUDA device
            samples_cpu = np.zeros((0, 3))

            # Initialize samples and move to CUDA device
            samples = torch.rand(1, sample_num, 3).float().to(self.device) * self.voxel_size - self.voxel_size / 2 # make samples   # make samples within voxel_size
            N = samples.shape[1]  # The number of samples
            # index = torch.full((N,), voxel_id, dtype=torch.long, device=self.device)
            indices = voxel_id.unsqueeze(0).repeat(N, 1)
            voxel_center_transfer  = (data_dict['voxel_coords'][:, 1:4][voxel_id[0]].unsqueeze(0) - data_dict['voxel_coords'][:, 1:4][voxel_id]) * self.voxel_size # K, 3


            samples.requires_grad = True

            i = 0
            while len(samples_cpu) < self.num_points:
                # print('iteration', i)

                for j in range(self.num_steps):
                    print('refinement', j)
                    samples_relative = samples.unsqueeze(2) - voxel_center_transfer.unsqueeze(0).unsqueeze(0) 
                    N = samples.shape[1]  # The number of samples
                    indices = voxel_id.unsqueeze(0).repeat(N, 1)
                    df_pred = torch.clamp(self.model(voxel_latents.detach(), samples[0], samples_relative[0], indices), max=self.threshold).unsqueeze(0)
                    df_pred.sum().backward(retain_graph=True)
                    gradient = samples.grad.detach()
                    samples = samples.detach()
                    df_pred = df_pred.detach()
                    samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
                    samples = samples.detach()
                    samples.requires_grad = True

                # print('finished refinement')

                if not i == 0:
                    # Move samples to CPU, detach from computation graph, convert to numpy array, and stack to samples_cpu
                    samples_cpu = np.vstack((samples_cpu, samples[df_pred < self.filter_val].detach().cpu().numpy()))

                samples = samples[df_pred < 0.03].unsqueeze(0)
                indices = torch.randint(samples.shape[1], (1, sample_num))
                samples = samples[[[0, ] * sample_num], indices]
                samples += (self.threshold / 3) * torch.randn(samples.shape).to(self.device)  # 3 sigma rule
                samples = samples.detach()
                samples.requires_grad = True

                i += 1
                print(samples_cpu.shape)

            duration = time.time() - start
            return samples_cpu, duration
        
        else:
            device = torch.device("cuda")

            start = time.time()

            # conver voxel latents
            mink_voxel_latents = voxel_latents
            voxel_latents = voxel_latents.F

            # freeze model parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # sample_num set to 200000
            sample_num = 200000

            # Initialize samples in CUDA device
            samples_cpu = np.zeros((0, 3))

            # Initialize voxel center coordinates
            points = data_dict['xyz'].detach()
            voxel_coords = data_dict['voxel_coords'][:, 1:4]
            voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 # compute voxel_center in orginal coordinate system (torch.tensor)
            voxel_center = voxel_center.to(device)
        
            # Initialize samples and move to CUDA device
            min_range, _ = torch.min(points, axis=0)
            max_range, _ = torch.max(points, axis=0)
            samples = torch.rand(1, sample_num, 3).float().to(device)
            samples *= (max_range.to(device) - min_range.to(device))
            samples += min_range.to(device) # make samples within coords_range
            N = samples.shape[1]  # The number of samples

            samples.requires_grad = True

            i = 0
            while len(samples_cpu) < self.num_points:
                # print('iteration', i)

                for j in range(self.num_steps):
                    # print('refinement', j)
                    # find the voxel_id of each sample
                    query_indices, _, _ = knn(samples[0], voxel_center, 1)
                    query_indices = query_indices.squeeze(-1) # (N, )

                    # remove query_points outside the voxel
                    lower_bound = -self.voxel_size / 2
                    upper_bound = self.voxel_size / 2
                    sample_relative_coords = samples[0] - voxel_center[query_indices]
                    # mask = (sample_relative_coords >= lower_bound) & (sample_relative_coords <= upper_bound)
                    # # Reduce across the last dimension to get a (N, ) mask
                    # mask = torch.all(mask, dim=-1)
                    # query_indices = query_indices[mask]
                    # mask = mask.unsqueeze(0) #1, N
                    # samples = samples[mask].unsqueeze(0) # 1, M, 3
                    # sample_relative_coords = samples[0] - voxel_center[query_indices]
                    # sample_relative_coords.retain_grad()
                    # samples.retain_grad()

                    df_pred = torch.clamp(self.model(voxel_latents.detach(), sample_relative_coords, query_indices), max=self.threshold).unsqueeze(0)
                    df_pred.sum().backward(retain_graph=True)
                    # gradient = sample_relative_coords.grad.unsqueeze(0).detach()
                    gradient = samples.grad.detach()
                    samples = samples.detach()
                    df_pred = df_pred.detach()
                    samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
                    samples = samples.detach()
                    samples.requires_grad = True

                # print('finished refinement')

                if not i == 0:
                    # Move samples to CPU, detach from computation graph, convert to numpy array, and stack to samples_cpu
                    samples_cpu = np.vstack((samples_cpu, samples[df_pred < self.filter_val].detach().cpu().numpy()))

                samples = samples[df_pred < 0.03].unsqueeze(0)
                indices = torch.randint(samples.shape[1], (1, sample_num))
                samples = samples[[[0, ] * sample_num], indices]
                samples += (self.threshold / 3) * torch.randn(samples.shape).to(device)  # 3 sigma rule
                samples = samples.detach()
                samples.requires_grad = True

                i += 1
                # print(samples_cpu.shape)
            for param in self.model.parameters():
                param.requires_grad = True
            duration = time.time() - start
            return samples_cpu, duration
        
    def generate_mesh(self, data_dict, voxel_latents):
        device = torch.device("cuda")
        start = time.time()
        # conver voxel latents
        mink_voxel_latents = voxel_latents
        voxel_latents = voxel_latents.F

        # Freeze model parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Grid dimensions
        grid_dim = 196
        total_voxels = grid_dim ** 3  # 128x128x128

        # Initialize voxel center coordinates
        points = data_dict['xyz'].detach()
        voxel_coords = data_dict['voxel_coords'][:, 1:4]  # M, 3
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0  # compute voxel_center in original coordinate system (torch.tensor)
        voxel_center = voxel_center.to(device)

        # Compute grid range
        min_range, _ = torch.min(points, axis=0)
        max_range, _ = torch.max(points, axis=0)
        grid_range = torch.linspace(min_range.min(), max_range.max(), steps=grid_dim)
        print(min_range, max_range)

        # Create uniform grid samples
        grid_x, grid_y, grid_z = torch.meshgrid(grid_range, grid_range, grid_range)
        samples = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, total_voxels, 3).float().to(device)

        # samples.requires_grad = True
        query_indices, _, _ = knn(samples[0], voxel_center, 1)
        query_indices = query_indices.squeeze(-1) # (N, )
        sample_relative_coords = samples[0] - voxel_center[query_indices]
        # df_pred = torch.clamp(self.model(voxel_latents.detach(), sample_relative_coords, query_indices), max=self.threshold).unsqueeze(0)

        sample_relative_parts = torch.chunk(sample_relative_coords, 32, dim=0)
        query_indices_parts = torch.chunk(query_indices, 32, dim=0)
        df_pred_parts = []
        for sample_relative_part, query_indices_part in tqdm(zip(sample_relative_parts, query_indices_parts), total=len(sample_relative_parts)):
            part_pred = torch.clamp(self.model(voxel_latents.detach(), sample_relative_part, query_indices_part), max=self.threshold)
            df_pred_parts.append(part_pred.detach().cpu().numpy()[np.newaxis, ...])
        df_pred = np.concatenate(df_pred_parts, 1)
        
        # Reshape the output to grid shape
        df_pred = df_pred.reshape(grid_dim, grid_dim, grid_dim)
        vertices, faces, normals, values = marching_cubes(df_pred, level=0.024, spacing=[1.0/255] * 3)
        # Create an Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

        return mesh, voxel_center.cpu().numpy()

class Interpolated_Dense_Generator(pl.LightningModule):
    def __init__(self, model, decoder_type, voxel_size, num_steps, num_points, threshold, filter_val, neighbor_type, k_neighbors):
        super().__init__()
        self.model = model # the model should be the UDF Decoder
        # self.model.eval()
        self.decoder_type = decoder_type
        self.voxel_size = voxel_size
        self.num_steps = num_steps
        self.num_points = num_points
        self.threshold = threshold
        self.filter_val = filter_val
        self.neighbor_type = neighbor_type
        self.k_neighbors = k_neighbors
        self.kernel_generator = KernelGenerator(kernel_size=2,
                                            stride=1,
                                            dilation=1,
                                            dimension=3)

    def cm_neighbors(self, encodes_tensor, query_indices):
        """ compute neighbor indices by minkowski cooridnate manager"""
        cm = encodes_tensor.coordinate_manager
        in_key = encodes_tensor.coordinate_key
        out_key = cm.stride(in_key, self.kernel_generator.kernel_stride)
        region_type, region_offset, _ = self.kernel_generator.get_kernel(encodes_tensor.tensor_stride, False)
        kernel_map = cm.kernel_map(in_key,
                                   out_key,
                                   self.kernel_generator.kernel_stride,
                                   self.kernel_generator.kernel_size,
                                   self.kernel_generator.kernel_dilation,
                                   region_type=region_type,
                                   region_offset=region_offset) #kernel size 3, stride 1, dilation 1
        neighbors = torch.full((encodes_tensor.shape[0], 8), -1, dtype=torch.long).to(encodes_tensor.device)
        for key in kernel_map.keys():
            in_out = kernel_map[key].long()
            neighbors[in_out[0], key] = in_out[1]
        
        neighbor_indices = neighbors[query_indices] #N, K
        mask = query_indices == -1
        neighbor_indices[mask] = -1
        
        return neighbor_indices
    
    def generate_point_cloud(self, data_dict, voxel_latents, device):
        # self.model.eval()
        start = time.time()
        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # sample_num set to 200000
        sample_num = 200000

        # Initialize samples in CUDA device
        samples_cpu = np.zeros((0, 3))

        # Initialize voxel center coordinates
        points = data_dict['xyz'].detach()
        voxel_coords = data_dict['voxel_coords'][:, 1:4] #M, 3
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 # compute voxel_center in orginal coordinate system (torch.tensor)
        voxel_center = voxel_center.to(device)

        # # conver voxel latents
        mink_voxel_latents = voxel_latents
        voxel_latents = voxel_latents.F
    
        # Initialize samples and move to CUDA device
        min_range, _ = torch.min(points, axis=0)
        max_range, _ = torch.max(points, axis=0)
        samples = torch.rand(1, sample_num, 3).float().to(device)
        samples *= (max_range.to(device) - min_range.to(device))
        voxel_coords = voxel_coords.to(device)
        samples += min_range.to(device) # make samples within coords_range
        N = samples.shape[1]  # The number of samples
        samples.requires_grad = True

        i = 0
        while len(samples_cpu) < self.num_points:
            # print('iteration', i)

            for j in range(self.num_steps):
                # print('refinement', j)
                # if self.neighbor_type == "cm":
                #     query_indices, _, _ = knn(samples[0], voxel_center, 1)
                #     query_indices = query_indices.squeeze(-1) # (N, )

                #     # remove query_points outside the range
                #     sample_relative_coords = samples[0] - voxel_center[query_indices]
                #     distances = torch.sqrt(torch.sum(sample_relative_coords**2, dim=1))
                #     mask = distances > 2 * self.voxel_size
                #     query_indices[mask] = -1
                #     query_indices = self.cm_neighbors(mink_voxel_latents, query_indices) # neighbor_indices

                # Reduce across the last dimension to get a (N, ) mask
                # mask = torch.all(mask, dim=-1)
                # query_indices = query_indices[mask]
                # mask = mask.unsqueeze(0) #1, N
                # samples = samples[mask].unsqueeze(0) # 1, M, 3
                # sample_relative_coords = samples[0][:, None, :] - voxel_center[query_indices]
                # sample_relative_coords.retain_grad()
                # samples.retain_grad()  
                 
                if self.neighbor_type == 'ball_query':
                    # query_indices = nearest_neighbors.knn(voxel_center, samples[0], self.k_neighbors)
                    query_indices, _, _ = knn(samples[0], voxel_center, self.k_neighbors)
                    sample_relative_coords = samples[0][:, None, :] - voxel_center[query_indices]
                    # remove query_points outside the ball range
                    distances = torch.sqrt(torch.sum(sample_relative_coords**2, dim=2))
                    mask = distances > 2 * self.voxel_size
                    query_indices[mask] = -1

                elif self.decoder_type == "MultiScaleInterpolatedDecoder":
                    df_pred = torch.clamp(torch.abs(self.model(encoder_outputs, samples[0])), max=self.threshold).unsqueeze(0)

                if self.decoder_type == "InterpolatedDecoder":
                    """ the abs is for SDF version """
                    query_indices, _, _ = knn(samples[0], voxel_center, self.k_neighbors)
                    df_pred = torch.clamp(torch.abs(self.model(voxel_latents.detach(), data_dict['voxel_coords'], samples[0], query_indices)), max=self.threshold).unsqueeze(0) 

                df_pred.sum().backward(retain_graph=True)
                # gradient = samples.grad.unsqueeze(0).detach()
                gradient = samples.grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
                samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
                samples = samples.detach()
                samples.requires_grad = True

            # print('finished refinement')

            if not i == 0:
                # Move samples to CPU, detach from computation graph, convert to numpy array, and stack to samples_cpu
                samples_cpu = np.vstack((samples_cpu, samples[df_pred < self.filter_val].detach().cpu().numpy()))

            samples = samples[df_pred < 0.03].unsqueeze(0)
            indices = torch.randint(samples.shape[1], (1, sample_num))
            samples = samples[[[0, ] * sample_num], indices]
            samples += (self.threshold / 3) * torch.randn(samples.shape).to(device)  # 3 sigma rule
            samples = samples.detach()
            samples.requires_grad = True

            i += 1
            # print(samples_cpu.shape)

        for param in self.model.parameters():
            param.requires_grad = True
        duration = time.time() - start

        return samples_cpu, duration

    def distance_p2p(self, points_src, normals_src, points_tgt, normals_tgt):
        kdtree = KDTree(points_tgt)
        dist, idx = kdtree.query(points_src)

        if normals_src is not None and normals_tgt is not None:
            normals_src = \
                normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
            normals_tgt = \
                normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

            normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
            # Handle normals that point into wrong direction gracefully
            # (mostly due to method not caring about this in generation)
            normals_dot_product = np.abs(normals_dot_product)
        else:
            normals_dot_product = np.array(
                [np.nan] * points_src.shape[0], dtype=np.float32)
        return dist, normals_dot_product

    def generate_mesh(self, data_dict, voxel_latents, device):
        device = torch.device("cuda")
        start = time.time()
        # Freeze model parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Grid dimensions
        # grid_dim = 256
        # total_voxels = grid_dim ** 3  # 128x128x128

        # conver voxel latents
        mink_voxel_latents = voxel_latents
        voxel_latents = voxel_latents.F

        # Initialize voxel center coordinates
        points = data_dict['xyz'].detach()
        voxel_coords = data_dict['voxel_coords'][:, 1:4]  # M, 3
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0  # compute voxel_center in original coordinate system (torch.tensor)
        voxel_center = voxel_center.to(device)

        # Compute grid range for each dimension separately
        min_range, _ = torch.min(points, axis=0)
        max_range, _ = torch.max(points, axis=0)
        # min_range -= 0.2
        # max_range += 0.2
        # Create separate grid ranges for x, y, and z
        resolution = 0.04  # 0.02 meters (2 cm)

        # Calculate physical size in each dimension
        physical_size_x = max_range[0] - min_range[0]
        physical_size_y = max_range[1] - min_range[1]
        physical_size_z = max_range[2] - min_range[2]

        # Calculate the number of steps in each dimension
        grid_dim_x = int(torch.round(physical_size_x / resolution))
        grid_dim_y = int(torch.round(physical_size_y / resolution))
        grid_dim_z = int(torch.round(physical_size_z / resolution))
        total_voxels = grid_dim_x * grid_dim_y * grid_dim_z

        grid_range_x = torch.linspace(min_range[0], max_range[0], steps=grid_dim_x)
        grid_range_y = torch.linspace(min_range[1], max_range[1], steps=grid_dim_y)
        grid_range_z = torch.linspace(min_range[2], max_range[2], steps=grid_dim_z)
        print(min_range, max_range)

        # Create uniform grid samples for the cuboid
        grid_x, grid_y, grid_z = torch.meshgrid(grid_range_x, grid_range_y, grid_range_z)
        samples = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, total_voxels, 3).float().to(device)

        # samples.requires_grad = True

        i = 0

        if self.neighbor_type == "cm":
            query_indices, _, _ = knn(samples[0], voxel_center, 1)
            query_indices = query_indices.squeeze(-1) # (N, )

            # remove query_points outside the range
            sample_relative_coords = samples[0] - voxel_center[query_indices]
            distances = np.sqrt(np.sum(sample_relative_coords**2, axis=1))
            mask = distances > 2 * self.voxel_size
            query_indices[mask] = -1
            query_indices = self.cm_neighbors(mink_voxel_latent, query_indices) # neighbor_indices
        
        else:
            query_indices, _, _ = knn(samples[0], voxel_center, self.k_neighbors)
            
        if self.neighbor_type == 'ball_query':
            # remove query_points outside the ball range
            sample_relative_coords = samples[0][:, None, :] - voxel_center[query_indices]
            distances = torch.sqrt(torch.sum(sample_relative_coords**2, dim=2))
            mask = distances > 2 * self.voxel_size
            query_indices[mask] = -1

        if self.decoder_type == "InterpolatedDecoder":
            sample_parts = torch.chunk(samples[0], 128, dim=0)
            query_indices_parts = torch.chunk(query_indices, 128, dim=0)
            df_pred_parts = []
            for sample_part, query_indices_part in tqdm(zip(sample_parts, query_indices_parts), total=len(sample_parts)):
                part_pred = torch.clamp(self.model(voxel_latents.detach(), data_dict['voxel_coords'], sample_part, query_indices_part), min=-1.0, max=1.0)
                df_pred_parts.append(part_pred.detach().cpu().numpy()[np.newaxis, ...])
            df_pred = np.concatenate(df_pred_parts, 1)
            
        else:
            df_pred = torch.clamp(self.model(voxel_latents.detach(), sample_relative_coords, query_indices), max=self.threshold).unsqueeze(0)

        # Reshape the output to grid shape
        # vertices, faces, normals, values = marching_cubes(df_pred, level=0.0, spacing=[1.0/255] * 3)
        distance, _ = self.distance_p2p(samples[0].cpu().numpy(), None, voxel_center.cpu().numpy(), None)
        mask = distance > 0.3
        df_pred[0][mask] = 10
        df_pred = df_pred.reshape(grid_dim_x, grid_dim_y, grid_dim_z)

        vertices, faces, normals, values = marching_cubes(df_pred, level=0.00, spacing=(resolution, resolution, resolution)) # for SDF

        # Create an Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

        # for param in self.model.parameters():
        #     param.requires_grad = True

        return mesh

    def get_grid_voxel_size_origin(self, depth: int):
            return self.voxel_size * (2 ** depth), 0.5 * self.voxel_size * (2 ** depth)

    def generate_dual_mc_mesh(self, data_dict, voxel_latents, device):
        from nksr.svh import SparseFeatureHierarchy, SparseIndexGrid
        from nksr.ext import meshing
        from nksr.meshing import MarchingCubes
        from nksr import utils

        max_depth = 100
        grid_upsample = 1
        max_points = -1
        mise_iter = 0
        # Initialize voxel center coordinates
        self.last_n_layers = 4
        
        device = torch.device("cuda")
        start = time.time()
        # conver voxel latents
        mink_voxel_latents = voxel_latents
        voxel_latents = voxel_latents.F

        # Initialize voxel center coordinates
        pts = data_dict['xyz'].detach()
        voxel_coords = data_dict['voxel_coords'][:, 1:4]  # M, 3
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0  # compute voxel_center in original coordinate system (torch.tensor)
        voxel_center = voxel_center.to(device)

        nksr_svh = SparseFeatureHierarchy(
            voxel_size=self.voxel_size,
            depth=self.last_n_layers,
            device= pts.device
        )
        nksr_svh.build_point_splatting(pts)

        # flattened_grids = []
        # single_dense_grid = SparseIndexGrid(*self.get_grid_voxel_size_origin(0), device=self.device)
        # single_dense_grid.build_from_pointcloud(torch.from_numpy(dense_points))
        # single_dense_grid  = meshing.build_flattened_grid(
        #     single_dense_grid._grid,
        #     None, # self.svh.grids[d - 1]._grid if d > 0 else None,
        #     True # d != self.svh.depth - 1
        # )
        # flattened_grids.append(single_dense_grid)

        flattened_grids = []
        for d in range(min(nksr_svh.depth, max_depth + 1)):
            f_grid = meshing.build_flattened_grid(
                nksr_svh.grids[d]._grid,
                nksr_svh.grids[d - 1]._grid if d > 0 else None,
                d != nksr_svh.depth - 1
            )
            flattened_grids.append(f_grid)

        dual_grid = meshing.build_joint_dual_grid(flattened_grids)
        dmc_graph = meshing.dual_cube_graph(flattened_grids, dual_grid)
        dmc_vertices = torch.cat([
            f_grid.grid_to_world(f_grid.active_grid_coords().float())
            for f_grid in flattened_grids if f_grid.num_voxels() > 0
        ], dim=0)
    
        del flattened_grids, dual_grid

        # query_indices, _, _ = knn(dmc_vertices.to(device), voxel_center, self.k_neighbors)
        nn = NearestNeighbors(n_neighbors=self.k_neighbors)
        nn.fit(voxel_center.cpu().numpy())  # coords is an (N, 3) array
        dist, indx = nn.kneighbors(dmc_vertices.cpu().numpy())  # xyz is an (M, 3) array
        query_indices = torch.from_numpy(indx).to(dmc_vertices.device)
        dist = torch.from_numpy(dist).to(dmc_vertices.device)
        
        dmc_value = torch.clamp(self.model(voxel_latents.detach(), data_dict['voxel_coords'], dmc_vertices.to(device), query_indices), max=self.threshold)
        
        # for _ in range(mise_iter):
        #     cube_sign = dmc_value[dmc_graph] > 0
        #     cube_mask = ~torch.logical_or(torch.all(cube_sign, dim=1), torch.all(~cube_sign, dim=1))
        #     dmc_graph = dmc_graph[cube_mask]
        #     unq, dmc_graph = torch.unique(dmc_graph.view(-1), return_inverse=True)
        #     dmc_graph = dmc_graph.view(-1, 8)
        #     dmc_vertices = dmc_vertices[unq]
        #     dmc_graph, dmc_vertices = utils.subdivide_cube_indices(dmc_graph, dmc_vertices)
        #     dmc_value = self.evaluate_f_bar(dmc_vertices, max_points=max_points)

        dual_v, dual_f = MarchingCubes().apply(dmc_graph, dmc_vertices, dmc_value)

        # if self.mask_field is not None and trim:
        #     vert_mask = self.mask_field.evaluate_f_bar(dual_v, max_points=max_points) < 0.0
        #     dual_v, dual_f = utils.apply_vertex_mask(dual_v, dual_f, vert_mask)

        # if self.texture_field is not None:
        #     dual_c = self.texture_field.evaluate_f_bar(dual_v, max_points=max_points)
        # else:
        #     dual_c = None

        mesh_res =  MeshingResult(dual_v, dual_f, None)
        mesh = vis.mesh(mesh_res.v, mesh_res.f)
        return mesh
    
class MultiScale_Interpolated_Dense_Generator(pl.LightningModule):
    def __init__(self, model, mask_model, decoder_type, voxel_size, num_steps, num_points, threshold, filter_val, neighbor_type, k_neighbors, last_n_layers, reconstruction_cfg):
        super().__init__()
        self.model = model # the model should be the UDF Decoder
        self.mask_model = mask_model # the distance mask decoder
        self.rec_cfg = reconstruction_cfg
        # self.model.eval()
        self.decoder_type = decoder_type
        self.voxel_size = voxel_size
        self.num_steps = num_steps
        self.num_points = num_points
        self.threshold = threshold
        self.filter_val = filter_val
        self.neighbor_type = neighbor_type
        self.k_neighbors = k_neighbors
        self.last_n_layers = last_n_layers
        self.kernel_generator = KernelGenerator(kernel_size=2,
                                            stride=1,
                                            dilation=1,
                                            dimension=3)

    def cm_neighbors(self, encodes_tensor, query_indices):
        """ compute neighbor indices by minkowski cooridnate manager"""
        cm = encodes_tensor.coordinate_manager
        in_key = encodes_tensor.coordinate_key
        out_key = cm.stride(in_key, self.kernel_generator.kernel_stride)
        region_type, region_offset, _ = self.kernel_generator.get_kernel(encodes_tensor.tensor_stride, False)
        kernel_map = cm.kernel_map(in_key,
                                   out_key,
                                   self.kernel_generator.kernel_stride,
                                   self.kernel_generator.kernel_size,
                                   self.kernel_generator.kernel_dilation,
                                   region_type=region_type,
                                   region_offset=region_offset) #kernel size 3, stride 1, dilation 1
        neighbors = torch.full((encodes_tensor.shape[0], 8), -1, dtype=torch.long).to(encodes_tensor.device)
        for key in kernel_map.keys():
            in_out = kernel_map[key].long()
            neighbors[in_out[0], key] = in_out[1]
        
        neighbor_indices = neighbors[query_indices] #N, K
        mask = query_indices == -1
        neighbor_indices[mask] = -1
        
        return neighbor_indices
    
    def generate_point_cloud(self, data_dict, encoder_outputs, device):
        # self.model.eval()
        # Visualize the sign of the distance field
        self.visualize_sign = True
        start = time.time()
        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # sample_num set to 200000
        sample_num = 200000

        # Initialize samples in CUDA device
        samples_cpu = np.zeros((0, 3))
        sign_cpu = np.zeros((0, 1))

        # Initialize voxel center coordinates
        points = data_dict['xyz'].detach()
        voxel_coords = data_dict['voxel_coords'][:, 1:4] #M, 3
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 # compute voxel_center in orginal coordinate system (torch.tensor)
        voxel_center = voxel_center.to(device)

        # # conver voxel latents
        # mink_voxel_latents = voxel_latents
        # voxel_latents = voxel_latents.F
    
        # Initialize samples and move to CUDA device
        min_range, _ = torch.min(points, axis=0)
        max_range, _ = torch.max(points, axis=0)
        samples = torch.rand(1, sample_num, 3).float().to(device)
        samples *= (max_range.to(device) - min_range.to(device))
        voxel_coords = voxel_coords.to(device)
        samples += min_range.to(device) # make samples within coords_range
        N = samples.shape[1]  # The number of samples
        samples.requires_grad = True

        i = 0
        while len(samples_cpu) < self.num_points:
            # print('iteration', i)

            for j in range(self.num_steps):
                # print('refinement', j)
                # if self.neighbor_type == "cm":
                #     query_indices, _, _ = knn(samples[0], voxel_center, 1)
                #     query_indices = query_indices.squeeze(-1) # (N, )

                #     # remove query_points outside the range
                #     sample_relative_coords = samples[0] - voxel_center[query_indices]
                #     distances = torch.sqrt(torch.sum(sample_relative_coords**2, dim=1))
                #     mask = distances > 2 * self.voxel_size
                #     query_indices[mask] = -1
                #     query_indices = self.cm_neighbors(mink_voxel_latents, query_indices) # neighbor_indices

                # Reduce across the last dimension to get a (N, ) mask
                # mask = torch.all(mask, dim=-1)
                # query_indices = query_indices[mask]
                # mask = mask.unsqueeze(0) #1, N
                # samples = samples[mask].unsqueeze(0) # 1, M, 3
                # sample_relative_coords = samples[0][:, None, :] - voxel_center[query_indices]
                # sample_relative_coords.retain_grad()
                # samples.retain_grad()  
                 
                if self.neighbor_type == 'ball_query':
                    # query_indices = nearest_neighbors.knn(voxel_center, samples[0], self.k_neighbors)
                    query_indices, _, _ = knn(samples[0], voxel_center, self.k_neighbors)
                    sample_relative_coords = samples[0][:, None, :] - voxel_center[query_indices]
                    # remove query_points outside the ball range
                    distances = torch.sqrt(torch.sum(sample_relative_coords**2, dim=2))
                    mask = distances > 2 * self.voxel_size
                    query_indices[mask] = -1

                elif self.decoder_type == "MultiScaleInterpolatedDecoder" or self.decoder_type == "LargeDecoder":
                    # df_pred = torch.clamp(torch.abs(self.model(encoder_outputs, samples[0])), max=self.threshold).unsqueeze(0)
                    sdf_pred = self.model(encoder_outputs, samples[0]).unsqueeze(0)
                    df_pred = torch.clamp(torch.abs(sdf_pred), max=self.threshold)

                if self.decoder_type == "InterpolatedDecoder":
                    """ the abs is for SDF version """
                    query_indices, _, _ = knn(samples[0], voxel_center, self.k_neighbors)
                    # df_pred = torch.clamp(torch.abs(self.model(voxel_latents.detach(), data_dict['voxel_coords'], samples[0], query_indices)), max=self.threshold).unsqueeze(0)
                    sdf_pred = self.model(voxel_latents.detach(), data_dict['voxel_coords'], samples[0], query_indices).unsqueeze(0)
                    df_pred = torch.clamp(torch.abs(sdf_pred), max=self.threshold)

                df_pred.sum().backward(retain_graph=True)
                # gradient = samples.grad.unsqueeze(0).detach()
                gradient = samples.grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
                samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
                samples = samples.detach()
                samples.requires_grad = True

            # print('finished refinement')

            if not i == 0:
                # Move samples to CPU, detach from computation graph, convert to numpy array, and stack to samples_cpu
                samples_cpu = np.vstack((samples_cpu, samples[df_pred < self.filter_val].detach().cpu().numpy()))
                if self.visualize_sign:
                    sign_cpu = np.vstack((sign_cpu, torch.sign(sdf_pred[df_pred < self.filter_val]).unsqueeze(-1).detach().cpu().numpy()))
            samples = samples[df_pred < 0.03].unsqueeze(0)
            indices = torch.randint(samples.shape[1], (1, sample_num))
            samples = samples[[[0, ] * sample_num], indices]
            samples += (self.threshold / 3) * torch.randn(samples.shape).to(device)  # 3 sigma rule
            samples = samples.detach()
            samples.requires_grad = True

            i += 1
            # print(samples_cpu.shape)

        for param in self.model.parameters():
            param.requires_grad = True
        duration = time.time() - start

        return samples_cpu, duration

    def distance_p2p(self, points_src, normals_src, points_tgt, normals_tgt):
        kdtree = KDTree(points_tgt)
        dist, idx = kdtree.query(points_src)

        if normals_src is not None and normals_tgt is not None:
            normals_src = \
                normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
            normals_tgt = \
                normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

            normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
            # Handle normals that point into wrong direction gracefully
            # (mostly due to method not caring about this in generation)
            normals_dot_product = np.abs(normals_dot_product)
        else:
            normals_dot_product = np.array(
                [np.nan] * points_src.shape[0], dtype=np.float32)
        return dist, normals_dot_product

    def generate_mesh(self, data_dict, encoder_outputs, device):
        start = time.time()
        # Freeze model parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Grid dimensions
        # grid_dim = 256
        # total_voxels = grid_dim ** 3  # 128x128x128

        # # conver voxel latents
        # mink_voxel_latents = voxel_latents
        # voxel_latents = voxel_latents.F

        # Initialize voxel center coordinates
        points = data_dict['xyz'].detach()
        voxel_coords = data_dict['voxel_coords'][:, 1:4]  # M, 3
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0  # compute voxel_center in original coordinate system (torch.tensor)
        voxel_center = voxel_center.to(device)

        # Compute grid range for each dimension separately
        min_range, _ = torch.min(points, axis=0)
        max_range, _ = torch.max(points, axis=0)
        # min_range -= 0.2
        # max_range += 0.2
        # Create separate grid ranges for x, y, and z
        resolution = 0.04  # 0.02 meters (2 cm)

        # Calculate physical size in each dimension
        physical_size_x = max_range[0] - min_range[0]
        physical_size_y = max_range[1] - min_range[1]
        physical_size_z = max_range[2] - min_range[2]

        # Calculate the number of steps in each dimension
        grid_dim_x = int(torch.round(physical_size_x / resolution))
        grid_dim_y = int(torch.round(physical_size_y / resolution))
        grid_dim_z = int(torch.round(physical_size_z / resolution))
        total_voxels = grid_dim_x * grid_dim_y * grid_dim_z

        grid_range_x = torch.linspace(min_range[0], max_range[0], steps=grid_dim_x)
        grid_range_y = torch.linspace(min_range[1], max_range[1], steps=grid_dim_y)
        grid_range_z = torch.linspace(min_range[2], max_range[2], steps=grid_dim_z)
        print(min_range, max_range)

        # Create uniform grid samples for the cuboid
        grid_x, grid_y, grid_z = torch.meshgrid(grid_range_x, grid_range_y, grid_range_z)
        samples = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, total_voxels, 3).float().to(device)

        # samples.requires_grad = True

        i = 0

        if self.neighbor_type == "cm":
            query_indices, _, _ = knn(samples[0], voxel_center, 1)
            query_indices = query_indices.squeeze(-1) # (N, )

            # remove query_points outside the range
            sample_relative_coords = samples[0] - voxel_center[query_indices]
            distances = np.sqrt(np.sum(sample_relative_coords**2, axis=1))
            mask = distances > 2 * self.voxel_size
            query_indices[mask] = -1
            query_indices = self.cm_neighbors(mink_voxel_latent, query_indices) # neighbor_indices
        
        else:
            query_indices, _, _ = knn(samples[0], voxel_center, self.k_neighbors)
            
        if self.neighbor_type == 'ball_query':
            # remove query_points outside the ball range
            sample_relative_coords = samples[0][:, None, :] - voxel_center[query_indices]
            distances = torch.sqrt(torch.sum(sample_relative_coords**2, dim=2))
            mask = distances > 2 * self.voxel_size
            query_indices[mask] = -1

        if self.decoder_type == "MultiScaleInterpolatedDecoder":
            sample_parts = torch.chunk(samples[0], 128, dim=0)
            query_indices_parts = torch.chunk(query_indices, 128, dim=0)
            df_pred_parts = []
            for sample_part, query_indices_part in tqdm(zip(sample_parts, query_indices_parts), total=len(sample_parts)):
                part_pred = torch.clamp(self.model(encoder_outputs, sample_part), min=-1.0, max=1.0)
                df_pred_parts.append(part_pred.detach().cpu().numpy()[np.newaxis, ...])
            df_pred = np.concatenate(df_pred_parts, 1)
            
        else:
            df_pred = torch.clamp(self.model(voxel_latents.detach(), sample_relative_coords, query_indices), max=self.threshold).unsqueeze(0)

        # Reshape the output to grid shape
        # vertices, faces, normals, values = marching_cubes(df_pred, level=0.0, spacing=[1.0/255] * 3)
        distance, _ = self.distance_p2p(samples[0].cpu().numpy(), None, voxel_center.cpu().numpy(), None)
        mask = distance > 0.3
        df_pred[0][mask] = 10
        df_pred = df_pred.reshape(grid_dim_x, grid_dim_y, grid_dim_z)

        vertices, faces, normals, values = marching_cubes(df_pred, level=0.00, spacing=(resolution, resolution, resolution)) # for SDF

        # Create an Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

        # for param in self.model.parameters():
        #     param.requires_grad = True

        return mesh, voxel_center.cpu().numpy()

    def get_grid_voxel_size_origin(self, depth: int):
            return self.voxel_size * (2 ** depth), 0.5 * self.voxel_size * (2 ** depth)

    def compute_gt_sdf_from_pts(self, gt_xyz, gt_normals, query_pos: torch.Tensor):
        k = 8  
        stdv = 0.02
        # knn_output = knn_points(query_pos.unsqueeze(0), self.ref_xyz.unsqueeze(0), K)
        # indices = knn_output.idx.squeeze(0)
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
            # nksr_svh.build_point_splatting(masked_grid_points)
            # nksr_svh.build_from_grid_coords(4, masked_grid_points)
        else:
            nksr_svh.build_point_splatting(pts)
        # flattened_grids = []
        # single_dense_grid = SparseIndexGrid(*self.get_grid_voxel_size_origin(0), device=self.device)
        # single_dense_grid.build_from_pointcloud(torch.from_numpy(dense_points))
        # single_dense_grid  = meshing.build_flattened_grid(
        #     single_dense_grid._grid,
        #     None, # self.svh.grids[d - 1]._grid if d > 0 else None,
        #     True # d != self.svh.depth - 1
        # )
        # flattened_grids.append(single_dense_grid)

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
        # query_indices, _, _ = knn(dmc_vertices.to(device), voxel_center, self.k_neighbors)
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

        # batch_size = 2000  # You can adjust this based on your memory constraints
        # N = dmc_vertices.shape[0]
        # dmc_values = []

        # for i in tqdm(range(0, N, batch_size), desc="Processing Batches"):
        #     batch = dmc_vertices[i:i + batch_size].to(device)
        #     dmc_value_batch = self.model(encoder_outputs, batch)
        #     dmc_values.append(dmc_value_batch)
        #     # Delete the batch and value batch to free memory
        #     del batch, dmc_value_batch
        #     torch.cuda.empty_cache()  # Clear cache to free memory

        # # # Concatenate all the batch results to form the final output
        # dmc_values = torch.cat(dmc_values, dim=0)
        
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
            # vert_mask = self.mask_field.evaluate_f_bar(dual_v, max_points=max_points) < 0.0 # type: ignore
            # vert_mask = self.mask_model(encoder_outputs, dual_v.to(device)) < mask_threshold
            if self.gt_mask: 
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(data_dict['all_xyz'].cpu().numpy())  # coords is an (N, 3) array
                dist, indx = nn.kneighbors(dual_v.detach().cpu().numpy())  # xyz is an (M, 3) array
                dist = torch.from_numpy(dist).to(dual_v.device).squeeze(-1)
                vert_mask = dist < mask_threshold
            else: 
                # nn = NearestNeighbors(n_neighbors=1)
                # nn.fit(data_dict['all_xyz'].cpu().numpy())  # coords is an (N, 3) array
                # dist, indx = nn.kneighbors(dual_v.detach().cpu().numpy())  # xyz is an (M, 3) array
                # gt_dist = torch.from_numpy(dist).to(dual_v.device).squeeze(-1)
                # vert_mask = dist < mask_threshold
                decoder_time -= time.time()
                dist, mask_knn_time, mask_attentive_time, mask_interpolation_time, mask_after_layers_time, _ = self.mask_model(encoder_outputs, dual_v.to(device))
                decoder_time += time.time()
                vert_mask = dist < mask_threshold
                knn_time += mask_knn_time
                attentive_time += mask_attentive_time
                interpolation_time += mask_interpolation_time
                after_layers_time += mask_after_layers_time
                # vert_mask = torch.abs(torch.clamp(self.model(encoder_outputs, dual_v.to(device)), max=self.threshold)) < mask_threshold
            dmc_time -= time.time()
            dual_v, dual_f = utils.apply_vertex_mask(dual_v, dual_f, vert_mask)
            dmc_time += time.time()

        # if self.texture_field is not None:
        #     dual_c = self.texture_field.evaluate_f_bar(dual_v, max_points=max_points)
        # else:
        #     dual_c = None
        # if vert_mask is not None:
        #     del vert_mask
        dmc_time -= time.time()
        mesh_res =  MeshingResult(dual_v, dual_f, None)
        # del dual_v, dual_f
        mesh = vis.mesh(mesh_res.v, mesh_res.f)
        dmc_time += time.time()
        return mesh, knn_time, dmc_time, attentive_time, interpolation_time, after_layers_time, decoder_time, grid_splat_time, dmc_vertices.cpu().numpy(), dmc_value.cpu().numpy(), colors.cpu().numpy(), voxel_centers.cpu().numpy()
    
    def compute_objective_function(self, cfg, data_dict, encoder_outputs, device):
        self.normals = 'Analytical' # 'Analytical' or 'Numerical'
        xyz = data_dict['xyz']
        gt_normals = data_dict['normals']
        self.batched_sampler = BatchedSampler(cfg)
        query_xyz, query_gt_sdf = self.batched_sampler.batch_sdf_sample(data_dict)
        if self.normals == 'Analytical':
            xyz.requires_grad = True
            with torch.enable_grad():
                pd_sdf, *_ = self.model(encoder_outputs, xyz)
                pd_normals = torch.autograd.grad(pd_sdf, [xyz],
                                                    grad_outputs=torch.ones_like(pd_sdf),
                                                    create_graph=self.model.training, allow_unused=True)[0]
                norms = torch.norm(pd_normals, dim=1)
                eikonal_loss = torch.mean((norms - 1) ** 2)
            
        else: 
            interval = 0.01 * self.voxel_size
            grad_value = []
            for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
                offset_tensor = torch.tensor(offset, device=device)[None, :]
                res_p, *_ = self.model(encoder_outputs, xyz + offset_tensor)
                res_n, *_ = self.model(encoder_outputs, xyz - offset_tensor)
                grad_value.append((res_p - res_n) / (2 * interval))
            pd_normals = torch.stack(grad_value, dim=-1)
            pd_sdf, *_ = self.model(encoder_outputs, xyz)

        query_xyz.requires_grad = True
        with torch.enable_grad():
            pd_sdf, *_ = self.model(encoder_outputs, query_xyz)
            query_normals = torch.autograd.grad(pd_sdf, [query_xyz],
                                                grad_outputs=torch.ones_like(pd_sdf),
                                                create_graph=self.model.training, allow_unused=True)[0]
        
        pd_normals = -pd_normals / (torch.linalg.norm(pd_normals, dim=-1, keepdim=True) + 1.0e-6)
        sdf_error = torch.mean(torch.abs(pd_sdf))
        normal_error = torch.mean(torch.norm(pd_normals - gt_normals, dim=1))


        
        return sdf_error, normal_error
    
    # def generate_simple_mc_mesh(self, data_dict, encoder_outputs, device):
    #     from nksr.svh import SparseFeatureHierarchy, SparseIndexGrid
    #     from nksr.ext import meshing
    #     from nksr.meshing import MarchingCubes
    #     from nksr import utils

    #     max_depth = 100
    #     grid_upsample = 1
    #     max_points = -1
    #     mise_iter = 0
    #     knn_time = 0
    #     sdf_time = 0 
    #     attentive_time = 0
    #     interpolation_time = 0
    #     after_layers_time = 0
    #     decoder_time = 0
    #     grid_splat_time = 0
    #     mask_threshold = 0.02
    #     self.build_on_splated_points = False
    #     # Initialize voxel center coordinates
    #     if self.build_on_splated_points:
    #         pts = data_dict['xyz'].detach()
    #     else:
    #         pts = data_dict['un_splats_xyz'].detach()
    #     self.last_n_layers = 4
    #     self.trim = True
    #     self.gt_mask = False
    #     self.gt_sdf = False
    #     self.build_on_splated_points = False
    #     self.regular_grid = True

    #     grid_splat_time -= time.time()

    #     min_coords = pts.min(dim=0)[0].cpu().numpy()
    #     max_coords = pts.max(dim=0)[0].cpu().numpy()

    #     resolution = 64
    #     x = np.linspace(min_coords[0], max_coords[0], resolution)
    #     y = np.linspace(min_coords[1], max_coords[1], resolution)
    #     z = np.linspace(min_coords[2], max_coords[2], resolution)

    #     grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    #     grids = np.stack([grid_x, grid_y, grid_z], axis=-1)
    #     grids = torch.tensor(grids, dtype=torch.float32).reshape(-1, 3)

    #     grid_splat_time += time.time()
    #     # query_indices, _, _ = knn(dmc_vertices.to(device), voxel_center, self.k_neighbors)
    #     """ create a mask to trim spurious geometry """

    #     decoder_time -= time.time()
    #     sdf_value, sdf_knn_time, sdf_attentive_time, sdf_interpolation_time, sdf_after_layers_time, voxel_centers = self.model(encoder_outputs, grids)
    #     decoder_time += time.time()
    #     knn_time += sdf_knn_time
    #     attentive_time += sdf_attentive_time
    #     interpolation_time += sdf_interpolation_time
    #     after_layers_time += sdf_after_layers_time
    #     if self.gt_sdf:
    #         if 'gt_geometry' in data_dict:
    #             ref_xyz, ref_normal, _ = data_dict['gt_geometry'][0].torch_attr()
    #         else:
    #             ref_xyz, ref_normal = data_dict['all_xyz'], data_dict['all_normals']
    #         sdf_value = self.compute_gt_sdf_from_pts(ref_xyz, ref_normal, grids)

    #     colors = torch.zeros((sdf_value.shape[0], 3)).to(sdf_value.device)
    #     colors[sdf_value > 0] = torch.Tensor([1, 0, 0]).to(sdf_value.device)  # Red for positive values
    #     colors[sdf_value < 0] = torch.Tensor([0, 1, 0]).to(sdf_value.device)  # Green for negative values

    #     # Normalize the color intensity based on the magnitude of sdf_values
    #     max_abs_value = torch.max(torch.abs(sdf_value))
    #     colors *= torch.abs(sdf_value)[:, None] / max_abs_value


    #     sdf_time -= time.time()
    #     dual_v, dual_f = MarchingCubes().apply(sdf_graph, sdf_vertices, sdf_value)
    #     sdf_time += time.time()

    #     vert_mask = None
    #     if self.trim:
    #         # vert_mask = self.mask_field.evaluate_f_bar(dual_v, max_points=max_points) < 0.0 # type: ignore
    #         # vert_mask = self.mask_model(encoder_outputs, dual_v.to(device)) < mask_threshold
    #         if self.gt_mask: 
    #             nn = NearestNeighbors(n_neighbors=1)
    #             nn.fit(data_dict['all_xyz'].cpu().numpy())  # coords is an (N, 3) array
    #             dist, indx = nn.kneighbors(dual_v.detach().cpu().numpy())  # xyz is an (M, 3) array
    #             dist = torch.from_numpy(dist).to(dual_v.device).squeeze(-1)
    #             vert_mask = dist < mask_threshold
    #         else: 
    #             # nn = NearestNeighbors(n_neighbors=1)
    #             # nn.fit(data_dict['all_xyz'].cpu().numpy())  # coords is an (N, 3) array
    #             # dist, indx = nn.kneighbors(dual_v.detach().cpu().numpy())  # xyz is an (M, 3) array
    #             # gt_dist = torch.from_numpy(dist).to(dual_v.device).squeeze(-1)
    #             # vert_mask = dist < mask_threshold
    #             decoder_time -= time.time()
    #             dist, mask_knn_time, mask_attentive_time, mask_interpolation_time, mask_after_layers_time, _ = self.mask_model(encoder_outputs, dual_v.to(device))
    #             decoder_time += time.time()
    #             vert_mask = dist < mask_threshold
    #             knn_time += mask_knn_time
    #             attentive_time += mask_attentive_time
    #             interpolation_time += mask_interpolation_time
    #             after_layers_time += mask_after_layers_time
    #             # vert_mask = torch.abs(torch.clamp(self.model(encoder_outputs, dual_v.to(device)), max=self.threshold)) < mask_threshold
    #         sdf_time -= time.time()
    #         dual_v, dual_f = utils.apply_vertex_mask(dual_v, dual_f, vert_mask)
    #         sdf_time += time.time()

    #     # if self.texture_field is not None:
    #     #     dual_c = self.texture_field.evaluate_f_bar(dual_v, max_points=max_points)
    #     # else:
    #     #     dual_c = None
    #     # if vert_mask is not None:
    #     #     del vert_mask
    #     sdf_time -= time.time()
    #     mesh_res =  MeshingResult(dual_v, dual_f, None)
    #     # del dual_v, dual_f
    #     mesh = vis.mesh(mesh_res.v, mesh_res.f)
    #     sdf_time += time.time()
    #     return mesh, knn_time, sdf_time, attentive_time, interpolation_time, after_layers_time, decoder_time, grid_splat_time, sdf_vertices.cpu().numpy(), sdf_value.cpu().numpy(), colors.cpu().numpy(), voxel_centers.cpu().numpy()