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
from hybridpc.model.module.common import ResnetBlockFC
from typing import Callable, Tuple
from pycarus.geometry.pcd import knn
import open3d as o3d
import pytorch_lightning as pl

class CoordsEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dims: int = 3,
        include_input: bool = True,
        max_freq_log2: int = 9,
        num_freqs: int = 10,
        log_sampling: bool = True,
        periodic_fns: Tuple[Callable, Callable] = (torch.sin, torch.cos)
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: Tensor) -> Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class ImplicitDecoder(pl.LightningModule):
    def __init__(
        self,
        type: str,
        cfg,
        embed_dim: int,
        voxel_size: float,
        out_dim: int
    ) -> None:
        super().__init__()

        in_dim = cfg.input_dim
        embed_dim = embed_dim
        hidden_dim = cfg.hidden_dim
        out_dim = out_dim
        self.local_coords = cfg.local_coords
        self.decoder_type = cfg.decoder_type
        self.k_neighbors = cfg.k_neighbors # 1 for no interpolation
        self.interpolation_mode = cfg.interpolation_mode
        self.normalize_encoding = cfg.normalize_encoding
        self.voxel_size = voxel_size
        self.num_hidden_layers_before_skip = cfg.num_hidden_layers_before_skip
        self.num_hidden_layers_after_skip = cfg.num_hidden_layers_after_skip

        self.coords_enc = CoordsEncoder(in_dim)
        enc_dim = self.coords_enc.out_dim

        if self.k_neighbors > 1:
            self.interpolation_layer = ResnetBlockFC(embed_dim+in_dim, embed_dim)

        # coords_dim = 0
        if self.decoder_type == 'ConvONet':
            self.fc_c = nn.ModuleList([
                nn.Linear(embed_dim, hidden_dim) for i in range(self.num_hidden_layers_before_skip)
            ])
            self.blocks = nn.ModuleList([
                ResnetBlockFC(hidden_dim) for i in range(self.num_hidden_layers_before_skip)
            ])
            self.fc_p = nn.Linear(enc_dim, hidden_dim)
        else:
            self.in_layer = nn.Sequential(nn.Linear(embed_dim + enc_dim, hidden_dim), nn.ReLU())
            self.skip_proj = nn.Sequential(nn.Linear(embed_dim + enc_dim, hidden_dim), nn.ReLU())
            before_skip = []
            for _ in range(self.num_hidden_layers_before_skip):
                before_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
            self.before_skip = nn.Sequential(*before_skip)

        after_skip = []
        for _ in range(self.num_hidden_layers_after_skip):
            after_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        after_skip.append(nn.Linear(hidden_dim, out_dim))
        if self.type == 'functa':
            after_skip.append(nn.ReLU())
        self.after_skip = nn.Sequential(*after_skip)
        
    def interpolation(self, voxel_latents: Tensor, coords: Tensor, index: Tensor):
        """Interpolates voxel features for a given set of points.

        The function calculates interpolated features based on the voxel latent codes and the indices 
        of the nearest voxels for each point. The interpolation takes into account the spatial 
        proximity of the nearest voxels to each point.

        Args:
            voxel_latents (Tensor): A tensor containing voxel latent codes. 
                It has the shape (M, D), where M is the number of voxels and D is the dimension of the latent space.

            coords (Tensor): A tensor containing the coordinates of sampled points.
                It has the shape (N, K, 3), where N is the number of sampled points and each point is represented by its relative coordinates to multiple voxel centers.

            index (Tensor): A tensor containing the indices of the K nearest voxels for each sampled point.
                It has the shape (N, K), where N is the number of sampled points and K is the number of nearest voxels considered for each point.

        Returns:
            Tensor: A tensor containing the interpolated features for all sampled points.
                The output tensor has the shape (N, D), where N is the number of sampled points and D is the dimension of the latent space.
        """

        # Gather the corresponding voxel latents based on index
        gathered_latents = voxel_latents[index]
        if self.interpolation_mode == 'trilinear':
            gathered_features = gathered_latents
        else:
            gathered_features = self.interpolation_layer(torch.cat((gathered_latents, coords), 2))

        # Calculate the weights for interpolation based on distances
        # Here, we simply use the inverse of the distance. Closer voxels have higher weights.
        if self.interpolation_mode =='inverse_distance':
            distances = torch.norm(coords, dim=2)
            weights = 1.0 / (distances + 1e-8) # Adding a small constant to avoid division by zero
            normalized_weights = weights / torch.sum(weights, dim=1, keepdim=True) # Normalize the weights to sum up to 1
            
            # Compute the interpolated features
            interpolated_features = torch.sum(gathered_features * normalized_weights.unsqueeze(-1), dim=1) #N, D
        elif self.interpolation_mode == 'max':
            interpolated_features, _ = torch.max(gathered_features, dim=1) #N, D
        elif self.interpolation_mode == 'trilinear':
            # Calculate bounding box of the neighbors for each point
            max_abs_value, _ = torch.max(torch.abs(coords), dim=1, keepdim=True)  # Shape: N, 1, 3
            
            # Normalize coords based on the maximum absolute value
            norm_coords = coords / max_abs_value
            # Split the normalized coords for easier calculation
            x, y, z = norm_coords[..., 0], norm_coords[..., 1], norm_coords[..., 2]

            # Calculate weights for trilinear interpolation
            wx = 1 - x
            wy = 1 - y
            wz = 1 - z

            # Weights for each corner (assuming K=8)
            w000 = wx * wy * wz
            w001 = wx * wy * z
            w010 = wx * y * wz
            w011 = wx * y * z
            w100 = x * wy * wz
            w101 = x * wy * z
            w110 = x * y * wz
            w111 = x * y * z

            # Gather features for each corner (assuming gathered_features has shape N, K, C with K=8)
            f000, f001, f010, f011, f100, f101, f110, f111 = torch.split(gathered_features, 1, dim=1)

            # Trilinear interpolation
            interpolated_features = (w000 * f000 + w001 * f001 + w010 * f010 + w011 * f011 +
                                    w100 * f100 + w101 * f101 + w110 * f110 + w111 * f111).squeeze(1)
            
        else:
            raise ValueError(f"Unsupported interpolation mode: {self.interpolation_mode}")

        return interpolated_features
        


    # def forward(self, embeddings: Tensor, absolute_coords: Tensor, coords: Tensor, index: Tensor) -> Tensor:
    #     # embeddings (M, C)
    #     # absolute_coords (N, K, 3)
    #     # coords (N, 3) or (N, K, 3)
    #     # index (N, ) or (N, K)
    #     if self.k_neighbors == 1:
    #         embeded_coords = self.coords_enc.embed(coords[:, 0, :])
    #         selected_embeddings = embeddings[index[:, 0]]
    #         emb_and_coords = torch.cat([selected_embeddings, embeded_coords], dim=-1)

    #     else:
    #         embeded_coords = self.coords_enc.embed(absolute_coords) # embed the nearest relative coords
    #         interpolated_embeddings = self.interpolation(embeddings, coords, index) # N, C
    #         emb_and_coords = torch.cat([interpolated_embeddings, embeded_coords], dim=-1)
    #         # emb_and_coords = interpolated_embeddings

    #     x = self.in_layer(emb_and_coords)
    #     x = self.before_skip(x)

    #     inp_proj = self.skip_proj(emb_and_coords)
    #     x = x + inp_proj

    #     x = self.after_skip(x)

    #     return x.squeeze(-1)
    
    def forward(self, embeddings: Tensor, absolute_coords: Tensor, coords: Tensor, index: Tensor) -> Tensor:
        # embeddings (M, C)
        # absolute_coords (N, K, 3)
        # coords (N, 3) or (N, K, 3)
        # index (N, ) or (N, K)

        # compute positional encoding
        if self.local_coords:
            if self.normalize_encoding:
                enc_coords = self.coords_enc.embed(coords[:, 0, :] / self.voxel_size) # encode the nearest relative coords
            else:
                enc_coords = self.coords_enc.embed(coords[:, 0, :])
        else:
            enc_coords = self.coords_enc.embed(absolute_coords) # encode the nearest relative coords
        
        # nearest query
        if self.k_neighbors == 1:
            interpolated_embeddings = embeddings[index[:, 0]]
        else:
            interpolated_embeddings = self.interpolation(embeddings, coords, index) # N, C

        if self.decoder_type == 'ConvONet':
            net = self.fc_p(enc_coords) # N, hidden_dim
            for i in range(self.num_hidden_layers_before_skip):
                net = net + self.fc_c[i](interpolated_embeddings)
                net = self.blocks[i](net)
        
        else:
            emb_and_coords = torch.cat([interpolated_embeddings, enc_coords], dim=-1)
            net = self.in_layer(emb_and_coords)
            net = self.before_skip(net)
            inp_proj = self.skip_proj(emb_and_coords)
            net = net + inp_proj

        out = self.after_skip(net)

        return out.squeeze(-1)
    
# class UDF_Decoder(pl.LightningModule):
#     def __init__(
#         self,
#         embed_dim: int,
#         in_dim: int,
#         hidden_dim: int,
#         num_hidden_layes_before_skip: int,
#         num_hidden_layes_after_skip: int,
#         out_dim: int,
#     ) -> None:
#         super().__init__()

#         self.coords_enc = CoordsEncoder(in_dim)
#         coords_dim = self.coords_enc.out_dim

#         self.in_layer = nn.Sequential(nn.Linear(embed_dim + coords_dim, hidden_dim))

#         self.skip_proj = nn.Linear(embed_dim + coords_dim, hidden_dim)

#         before_skip = []
#         for _ in range(num_hidden_layes_before_skip):
#             before_skip.append(nn.Linear(hidden_dim, hidden_dim))
#         self.before_skip = nn.Sequential(*before_skip)

#         after_skip = []
#         for _ in range(num_hidden_layes_after_skip):
#             after_skip.append(nn.Linear(hidden_dim, hidden_dim))
#         after_skip.append(nn.Linear(hidden_dim, out_dim))
#         after_skip.append(nn.Tanh())
#         # after_skip.append(nn.ReLU())
#         self.after_skip = nn.Sequential(*after_skip)

#     def forward(self, embeddings: Tensor, coords: Tensor, index: Tensor) -> Tensor:
#         # embeddings (B, D1)
#         # coords (N, D2)
#         # index (N, )
#         coords = self.coords_enc.embed(coords)
#         selected_embeddings = embeddings[index]

#         # selected_embeddings = embeddings[index]
#         # Concatenate the selected embeddings and the encoded coordinates
#         emb_and_coords = torch.cat([selected_embeddings, coords], dim=-1)

#         x = self.in_layer(emb_and_coords)
#         x = self.before_skip(x)

#         inp_proj = self.skip_proj(emb_and_coords)
#         x = x + inp_proj

#         x = self.after_skip(x)

#         return x.squeeze(-1)



class Dense_Generator(pl.LightningModule):
    def __init__(self, model, voxel_size, num_steps, num_points, threshold, filter_val, type='scene'):
        super().__init__()
        self.model = model
        self.model.eval()
        self.voxel_size = voxel_size
        self.num_steps = num_steps
        self.num_points = num_points
        self.threshold = threshold
        self.filter_val = filter_val
        self.type = type # voxel or scene

    def _generate_point_cloud_for_voxel(self, modulations, voxel_id, voxel_center):
        """
        Generate dense point cloud for a single voxel
        """
        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # sample_num set to 200000
        sample_num = 20000

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
                df_pred = torch.clamp(self.model(modulations.detach(), samples[0], index), max=self.threshold).unsqueeze(0)
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

            filtered_samples = samples[df_pred < 0.03]
            # Check if filtered_samples is empty and return the accumulated samples if true
            if filtered_samples.shape[0] == 0:
                return samples_cpu + voxel_center.cpu().numpy()
            samples = filtered_samples.unsqueeze(0)
            indices = torch.randint(samples.shape[1], (1, sample_num))
            samples = samples[[[0, ] * sample_num], indices]
            samples += (self.threshold / 3) * torch.randn(samples.shape).to(self.device)  # 3 sigma rule
            samples = samples.detach()
            samples.requires_grad = True

            i += 1
            # print(samples_cpu.shape)

        filtered_samples_cpu = samples_cpu[
            (samples_cpu[:, 0] >= -self.voxel_size / 2) & (samples_cpu[:, 0] <= self.voxel_size / 2) &
            (samples_cpu[:, 1] >= -self.voxel_size / 2) & (samples_cpu[:, 1] <= self.voxel_size / 2) &
            (samples_cpu[:, 2] >= -self.voxel_size / 2) & (samples_cpu[:, 2] <= self.voxel_size / 2)
        ]
        samples_cpu = filtered_samples_cpu
        samples_cpu += voxel_center.cpu().numpy()
        return samples_cpu

    def generate_all_voxel_point_clouds(self, data_dict, modulations):
        """
        Generate dense point clouds for all voxels and concatenate to make a large scene
        """
        start = time.time()

        all_point_clouds = []
        voxel_coords = data_dict['voxel_coords'][:, 1:4]
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 # compute voxel_center in original coordinate system
        voxel_center = voxel_center.to(self.device)

        print ('voxel num: ', len(voxel_center))
        for voxel_id in range(len(voxel_center)):
            print ('start for voxel: ', voxel_id)
            voxel_point_cloud = self._generate_point_cloud_for_voxel(modulations, voxel_id, voxel_center[voxel_id])
            all_point_clouds.append(voxel_point_cloud)
            print ('finished for voxel: ', voxel_id)
        
        duration = time.time() - start
        return np.vstack(all_point_clouds), duration

    def generate_point_cloud(self, data_dict, modulations, voxel_id):
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
                    df_pred = torch.clamp(self.model(modulations.detach(), samples[0], index), max=self.threshold).unsqueeze(0)
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


    def generate_df(self, data):
        # Move the inputs and points to the appropriate device
        inputs = data['inputs'].to(self.device)
        points = data['point_cloud'].to(self.device)
        scale = data['scale']
        # Compute the encoding of the input
        encoding = self.model.encoder(inputs)

        # Compute the predicted distance field for the points using the decoder
        df_pred = self.model.decoder(points, *encoding).squeeze(0)
        # Scale distance field back w.r.t. original point cloud scale
        # The predicted distance field is returned to the cpu
        df_pred_cpu = df_pred.detach().cpu().numpy()
        df_pred_cpu *= scale.detach().cpu().numpy()

        return df_pred_cpu