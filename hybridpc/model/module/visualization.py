from pyexpat import features
import time
import os
import numpy as np
import math
import torchmetrics
import torch
from einops import repeat
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Callable, Tuple
from pycarus.geometry.pcd import knn
import open3d as o3d
import pytorch_lightning as pl

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
        self.type = type # voxel or scene

    def generate_point_cloud(self, data_dict, encodes_dict, voxel_latents, voxel_id):
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
                print('iteration', i)

                for j in range(self.num_steps):
                    print('refinement', j)
                    df_pred = torch.clamp(self.model(voxel_latents.detach(), samples[0], index), max=self.threshold).unsqueeze(0)
                    df_pred.sum().backward(retain_graph=True)
                    gradient = samples.grad.detach()
                    samples = samples.detach()
                    df_pred = df_pred.detach()
                    samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
                    samples = samples.detach()
                    samples.requires_grad = True

                print('finished refinement')

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
            samples = torch.rand(1, sample_num, 3).float().to(self.device)  # make samples within voxel_size
            N = samples.shape[1]  # The number of samples
            # index = torch.full((N,), voxel_id, dtype=torch.long, device=self.device)
            indices = voxel_id.unsqueeze(0).repeat(N, 1)
            voxel_center_transfer  = encodes_dict['voxel_coords'][:, 1:4][voxel_id[0]].unsqueeze(0) - encodes_dict['voxel_coords'][:, 1:4][voxel_id] # K, 3


            samples.requires_grad = True

            i = 0
            while len(samples_cpu) < self.num_points:
                print('iteration', i)

                for j in range(self.num_steps):
                    print('refinement', j)
                    samples_relative = samples.unsqueeze(2) - voxel_center_transfer.unsqueeze(0).unsqueeze(0) 
                    N = samples.shape[1]  # The number of samples
                    indices = voxel_id.unsqueeze(0).repeat(N, 1)
                    df_pred = torch.clamp(self.model(voxel_latents.detach(), samples_relative[0], indices), max=self.threshold).unsqueeze(0)
                    df_pred.sum().backward(retain_graph=True)
                    gradient = samples.grad.detach()
                    samples = samples.detach()
                    df_pred = df_pred.detach()
                    samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
                    samples = samples.detach()
                    samples.requires_grad = True

                print('finished refinement')

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
            start = time.time()

            # freeze model parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # sample_num set to 200000
            sample_num = 400000

            # Initialize samples in CUDA device
            samples_cpu = np.zeros((0, 3))

            # Initialize voxel center coordinates
            points = data_dict['xyz'].detach()
            voxel_coords = data_dict['voxel_coords'][:, 1:4]
            voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 # compute voxel_center in orginal coordinate system (torch.tensor)
            voxel_center = voxel_center.to(self.device)
        
            # Initialize samples and move to CUDA device
            min_range, _ = torch.min(points, axis=0)
            max_range, _ = torch.max(points, axis=0)
            samples = torch.rand(1, sample_num, 3).float().to(self.device)
            samples *= (max_range.to(self.device) - min_range.to(self.device))
            samples += min_range.to(self.device) # make samples within coords_range
            N = samples.shape[1]  # The number of samples

            samples.requires_grad = True

            i = 0
            while len(samples_cpu) < self.num_points:
                print('iteration', i)

                for j in range(self.num_steps):
                    print('refinement', j)
                    # find the voxel_id of each sample
                    query_indices, _, _ = knn(samples[0], voxel_center, 1)
                    query_indices = query_indices.squeeze(-1) # (N, )

                    # remove query_points outside the voxel
                    lower_bound = -self.voxel_size / 2
                    upper_bound = self.voxel_size / 2
                    sample_relative_coords = samples[0] - voxel_center[query_indices]
                    mask = (sample_relative_coords >= lower_bound) & (sample_relative_coords <= upper_bound)
                    # Reduce across the last dimension to get a (N, ) mask
                    mask = torch.all(mask, dim=-1)
                    query_indices = query_indices[mask]
                    mask = mask.unsqueeze(0) #1, N
                    samples = samples[mask].unsqueeze(0) # 1, M, 3
                    sample_relative_coords = samples[0] - voxel_center[query_indices]
                    sample_relative_coords.retain_grad()
                    samples.retain_grad()

                    # sample_relative_coords = sample_relative_coords.unsqueeze(0) # 1, M, 3
                    # samples = samples.detach()
                    # samples.requires_grad = True
                    # samples.retains_grad = True
                    # # Create a mask
                    # mask = (sample_relative_coords >= lower_bound) & (sample_relative_coords <= upper_bound)
                    # # Reduce across the last dimension to get a (N, ) mask
                    # mask = torch.all(mask, dim=-1)
                    # query_indices = query_indices[mask]
                    # sample_relative_coords = sample_relative_coords[mask]
                    df_pred = torch.clamp(self.model(modulations.detach(), sample_relative_coords, query_indices), max=self.threshold).unsqueeze(0)
                    df_pred.sum().backward(retain_graph=True)
                    gradient = sample_relative_coords.grad.unsqueeze(0).detach()
                    # gradient = samples.grad.detach()
                    samples = samples.detach()
                    df_pred = df_pred.detach()
                    samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
                    samples = samples.detach()
                    samples.requires_grad = True

                print('finished refinement')

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

