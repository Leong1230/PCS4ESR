from pyexpat import features
import time
import os
import numpy as np
import math
import torchmetrics
import torch
# from einops import repeat
from torch import Tensor, nn
from torch.nn import functional as F
from hybridpc.model.module.common import ResnetBlockFC, ActivationConv1d
from typing import Callable, Tuple
import open3d as o3d
from pytorch3d.ops import knn_points
import pytorch_lightning as pl
import MinkowskiEngine as ME
from hybridpc.data.dataset.voxelizer import Voxelizer
from torch_scatter import scatter_mean, scatter_max, scatter_add
from hybridpc.utils.serialization import encode

    
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
        self.padding = False # pad to 64 dimensions
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
        if self.padding:
            out_dim += 1
        self.out_dim = out_dim

    def embed(self, inputs: Tensor) -> Tensor:
        embedded = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        if self.padding:
            padding = torch.zeros(embedded.shape[0], embedded.shape[1], 1, device=embedded.device)
            embedded = torch.cat([embedded, padding], dim=-1)
        return embedded

class ResMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation_fn, num_hidden_layers_before, num_hidden_layers_after):
        super(ResMLPBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim if num_hidden_layers_after > 0 else out_dim
        self.activation_fn = activation_fn
        self.num_hidden_layers_before = num_hidden_layers_before # No skip connection when is 0
        self.num_hidden_layers_after = num_hidden_layers_after

        # Initial transformation layer
        self.in_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            self.activation_fn
        )

        if self.num_hidden_layers_before > 0:
            self.skip_proj = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                self.activation_fn
            )

            before_skip = [
                nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn)
                for _ in range(self.num_hidden_layers_before)
            ]
            self.before_skip = nn.Sequential(*before_skip)

        if self.num_hidden_layers_after > 0:
            after_skip = [
                nn.Sequential(nn.Linear(self.hidden_dim, self.out_dim), self.activation_fn)
                for _ in range(self.num_hidden_layers_after)
            ]
            self.after_skip = nn.Sequential(*after_skip)

    def forward(self, x_in):
        # Apply initial transformation
        x = self.in_layer(x_in)
        
        if self.num_hidden_layers_before > 0:
            x = self.before_skip(x)
            inp_proj = self.skip_proj(x_in)
            x = x + inp_proj

        if self.num_hidden_layers_after > 0:
            x = self.after_skip(x)

        return x
    
class BaseDecoder(pl.LightningModule):
    def __init__(
        self,
        decoder_cfg,
        supervision,
        latent_dim: int,
        feature_dim: list,
        hidden_dim: int,
        out_dim: int,
        voxel_size: float,
        activation: str = 'ReLU'
    ) -> None:
        super().__init__()
        
        self.in_dim = 3
        self.ENC_DIM = 32
        self.supervision = supervision
        self.decoder_cfg = decoder_cfg
        self.decoder_type = decoder_cfg.decoder_type
        self.architecture = decoder_cfg.architecture
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.voxel_size = voxel_size
        self.coords_enc = CoordsEncoder(self.in_dim)  
        self.enc_dim = self.coords_enc.out_dim 
        self.activation_fn = self.get_activation(activation, self.decoder_cfg.negative_slope)
        self.use_bn = decoder_cfg.use_bn
        
        # Assuming other common components can be initialized here
        
    def get_activation(self, activation_str: str, negative_slope=0.01):
        """Return the desired activation function based on the string."""
        if activation_str == "ReLU":
            act = nn.ReLU()
        elif activation_str == "LeakyReLU":
            act = nn.LeakyReLU(negative_slope, inplace=True)
        elif activation_str == "Softplus":
            act = nn.Softplus()
        elif activation_str == "ShiftedSoftplus":
            def shifted_softplus(input_tensor):
                shifted = input_tensor - 1
                return nn.Softplus()(shifted)
            act = shifted_softplus
        else:
            raise ValueError(f"Activation {activation_str} not supported!")
        return act
    
class Decoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.interpolation_mode = kwargs['decoder_cfg'].interpolation_mode
        self.inter = kwargs['decoder_cfg'].inter
        self.learnable_alpha = kwargs['decoder_cfg'].learnable_alpha
        self.dist_mask = kwargs['decoder_cfg'].dist_mask
        self.no_pos_enc_noramalization = kwargs['decoder_cfg'].no_pos_enc_noramalization
        self.correct_pos_enc_noramalization = kwargs['decoder_cfg'].correct_pos_enc_noramalization
        self.pos_enc_dist_factor = kwargs['decoder_cfg'].pos_enc_dist_factor
        self.dist_factor = kwargs['decoder_cfg'].dist_factor
        self.neighboring = kwargs['decoder_cfg'].neighboring
        self.serial_neighbor_layers = kwargs['decoder_cfg'].serial_neighbor_layers
        self.filter_by_mask = kwargs['decoder_cfg'].filter_by_mask

        self.last_n_layers = kwargs['decoder_cfg'].last_n_layers
        self.k_neighbors = kwargs['decoder_cfg'].k_neighbors
        self.num_hidden_layers_after = kwargs['decoder_cfg'].num_hidden_layers_after
        self.num_hidden_layers_before = kwargs['decoder_cfg'].num_hidden_layers_before
        self.multi_scale_aggregation = kwargs['decoder_cfg'].multi_scale_aggregation
        self.transformer_layers = 2
        self.backbone = kwargs['decoder_cfg'].backbone
        self.larger_unet = kwargs['decoder_cfg'].larger_unet
        self.serial_orders = kwargs['decoder_cfg'].serial_orders
        self.dec_channels = kwargs['decoder_cfg'].decoder_channels
        self.stride = kwargs['decoder_cfg'].stride
        self.scale_visualization = kwargs['decoder_cfg'].scale_visualization
        self.larger_point_nerf = kwargs['decoder_cfg'].larger_point_nerf
        self.per_scale_in = kwargs['decoder_cfg'].per_scale_in
        self.point_nerf_hidden_dim = kwargs['decoder_cfg'].point_nerf_hidden_dim
        self.point_nerf_before_skip = kwargs['decoder_cfg'].point_nerf_before_skip  
        self.point_nerf_after_skip = kwargs['decoder_cfg'].point_nerf_after_skip

        if self.backbone == 'PointTransformerV3':
            in_channels = [self.dec_channels[0]] + self.dec_channels[0:-1]
        else:
            # Minkowski Backbone
            if self.larger_unet:
                in_channels = [self.latent_dim * (2**scale) for scale in range(self.last_n_layers)]
            else:
                in_channels = [self.latent_dim * (scale + 1) for scale in range(self.last_n_layers)]  
        self.per_scale_in = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels[scale], self.feature_dim[scale]), nn.ReLU())
            for scale in range(self.last_n_layers)
        ])

        self.point_nerf_blocks = nn.ModuleList([
            ResMLPBlock(
                self.enc_dim+self.feature_dim[i], self.point_nerf_hidden_dim, 32, nn.ReLU(), self.point_nerf_before_skip, self.point_nerf_after_skip
            ) 
            for i in range(self.last_n_layers)
        ])
        
        multi_scale_dim = 32
            
        self.all_scale_out = ResMLPBlock(
            multi_scale_dim,  self.hidden_dim, self.hidden_dim, self.activation_fn, self.num_hidden_layers_before, self.num_hidden_layers_after
        ) 

        if self.supervision == 'UDF':
            self.out = nn.Sequential(
                nn.Linear(self.hidden_dim, 1), 
                self.activation_fn
            )
        if self.supervision == 'Distance':
            self.out = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                # self.activation_fn
            )
        else: 
            self.out = nn.Sequential(
                nn.Linear(self.hidden_dim, 1), 
                nn.Tanh()
            )
        
    def interpolation(self, latents: Tensor, voxel_centers: Tensor, query_xyz: Tensor, index: Tensor, scale: int, layer: int):
        if self.filter_by_mask:
            N = index.shape[0]
            K = self.k_neighbors
            if self.neighboring == 'Mink':
                valid_mask = index >= 0
            else: 
                """ mask by radius """
                if index.shape[-1] > self.k_neighbors: # when using mixture neighbor method
                    index, _ = torch.sort(index, dim=-1)
                    _, indices = torch.unique_consecutive(index, return_inverse=True)
                    indices -= indices.min(dim=1, keepdims=True)[0]
                    result = -torch.ones_like(index).to(query_xyz.device)
                    index = result.scatter_(1, indices, index)
                    all_centers = voxel_centers[index] # M, K, 3
                    all_query_xyz = query_xyz.unsqueeze(1).repeat(1, index.shape[-1], 1) # M, K, 3
                    all_relative_coords = all_query_xyz - all_centers # M, K, 3
                    all_dist = torch.norm(all_relative_coords, dim=-1, keepdim=False) # M, K

                    sorted_idx = torch.argsort(all_dist, dim=-1)
                    sorted_indices = torch.gather(index, 1, sorted_idx)
                    sorted_dist = torch.gather(all_dist, 1, sorted_idx)

                    index = sorted_indices[:, :self.k_neighbors]  # Select the first k_neighbors
                    valid_mask = sorted_dist[:, :self.k_neighbors] <= self.dist_factor[layer] * (self.voxel_size * 2**scale)

                else:
                    """ mask by radius """
                    all_centers = voxel_centers[index] # M, K, 3
                    all_query_xyz = query_xyz.unsqueeze(1).repeat(1, index.shape[-1], 1) # M, K, 3
                    all_relative_coords = all_query_xyz - all_centers # M, K, 3
                    all_dist = torch.norm(all_relative_coords, dim=-1, keepdim=False) # M, K
                    valid_mask = all_dist <= self.dist_factor[layer] * (self.voxel_size * 2**scale)


            gathered_centers = voxel_centers[index[valid_mask]] # M, 3
            gathered_query_xyz = query_xyz.repeat_interleave(valid_mask.sum(dim=1), dim=0) # M, 3
            gathered_relative_coords = gathered_query_xyz - gathered_centers # M, 3
            gathered_dist = torch.norm(gathered_relative_coords, dim=-1, keepdim=False) # M, 
            gathered_latents = latents[index[valid_mask]] # M, C
            attentive_pooling_time = 0

            gathered_coords = self.coords_enc.embed(gathered_relative_coords/ (self.voxel_size * 2**scale))
            gathered_emb_and_coords = torch.cat([gathered_latents, gathered_coords], dim=-1) # M, C + enc_dim

            attentive_pooling_time -= time.time()                     
            valid_indices = torch.arange(N).unsqueeze(1).expand(N, K).flatten().to(query_xyz.device)  # Shape (N*K,)
            valid_indices = valid_indices[valid_mask.flatten()]  # Shape (M,)
            valid_features = self.point_nerf_blocks[layer](gathered_emb_and_coords)
            weights = torch.full((N, K), 1e-8).to(query_xyz.device)
            weights[valid_mask] = 1.0 / (gathered_dist + 1e-8) # Adding a small constant to avoid division by zero
            normalized_weights = weights / torch.sum(weights, dim=1, keepdim=True) # Normalize the weights to 
            weighted_features = valid_features * normalized_weights[valid_mask].unsqueeze(-1) # M, C
                
            interpolated_features = torch.zeros(N, weighted_features.shape[-1], device=weighted_features.device)
            interpolated_features = interpolated_features.scatter_add(0, valid_indices.unsqueeze(-1).expand_as(weighted_features), weighted_features)

            attentive_pooling_time += time.time()

            return interpolated_features, attentive_pooling_time

    def compute_average_recall(self, ref_indices, all_neighbor_idx):
        N, K = ref_indices.shape
        all_neighbor_idx = torch.stack(all_neighbor_idx)  # Shape: (num_orders, N, K)
        ref_indices_expanded = ref_indices.unsqueeze(0).expand_as(all_neighbor_idx)  # Shape: (num_orders, N, K)
        matches = (ref_indices_expanded.unsqueeze(3) == all_neighbor_idx.unsqueeze(2)).any(dim=3)
        correct_counts = matches.any(dim=0).sum(dim=1).float()
        average_recall = correct_counts.mean().item() / K

        return average_recall

    def serial_neighbor(self, source_voxel_coords, query_xyz, tensor_stride, knn_indices):
        all_neighbor_idx = []
        for serial_order in self.serial_orders:
            k_neighbors = self.k_neighbors
            ref_indices = knn_indices
            query_coords, index, inverse_index = ME.utils.sparse_quantize(coordinates=query_xyz, return_index=True, return_inverse=True, quantization_size=(self.voxel_size*tensor_stride), device=query_xyz.device)
            source_coords = (source_voxel_coords /tensor_stride).to(torch.int)
            batched_query_coords = ME.utils.batched_coordinates(
            [query_coords], dtype=query_coords.dtype, 
            device=query_coords.device)  

            depth = int(query_coords.max()).bit_length()
            query_codes = encode(batched_query_coords[:, 1:4], batched_query_coords[:, 0], depth, order=serial_order)
            source_codes = encode(source_coords[:, 1:4], torch.zeros(source_coords.shape[0], dtype=torch.int64, device=source_coords.device), depth, order=serial_order)

            sorted_source_codes, sorted_source_indices = torch.sort(source_codes)
            sorted_query_codes, sorted_query_indices = torch.sort(query_codes)
            nearest_right_positions = torch.searchsorted(sorted_source_codes, sorted_query_codes, right=True)

            k = int(k_neighbors/2)  # Number of neighbors in each direction
            front_indices = nearest_right_positions.unsqueeze(1) - torch.arange(1, k+1).to(nearest_right_positions.device).unsqueeze(0)
            back_indices = nearest_right_positions.unsqueeze(1) + torch.arange(0, k).to(nearest_right_positions.device).unsqueeze(0)

            # Combine front and back indices
            neighbor_indices = torch.cat((front_indices, back_indices), dim=1)

            # Pad indices that are out of range by -1
            neighbor_indices = torch.where((neighbor_indices >= 0) & (neighbor_indices < len(sorted_source_codes)), neighbor_indices, torch.tensor(-1))

            # Map the indices back to the original unsorted source codes
            neighbor_source_indices = torch.where(neighbor_indices != -1, sorted_source_indices[neighbor_indices], torch.tensor(-1))

            # Reorder the neighbors to match the original order of the query codes
            neighbor_idx = neighbor_source_indices[torch.argsort(sorted_query_indices)]

            if len(inverse_index) > 0:
                neighbor_idx = neighbor_idx[inverse_index]
        
            all_neighbor_idx.append(neighbor_idx)
        
        # average_recall = self.compute_average_recall(ref_indices, all_neighbor_idx)
        # print(f"Average Recall: {average_recall}")
        return torch.cat(all_neighbor_idx, dim=-1)

    def pt_serial_neighbor(self, points, query_xyz):
        
        all_neighbor_idx = []
        grid_size = 0.01
        sort_queries = False
        for order in self.serial_orders:
            k_neighbors = self.k_neighbors
            query_coords, index, inverse_index = ME.utils.sparse_quantize(coordinates=query_xyz, return_index=True, return_inverse=True, quantization_size=(grid_size), device=query_xyz.device)

            source_coords = torch.floor(points/grid_size).to(torch.int)

            depth = int(query_coords.max()).bit_length()
            query_codes = encode(query_coords, torch.zeros(query_coords.shape[0], dtype=torch.int64, device=query_coords.device), depth, order=order)

            source_codes = encode(source_coords, torch.zeros(source_coords.shape[0], dtype=torch.int64, device=source_coords.device), depth, order=order)

            sorted_source_codes, sorted_source_indices = torch.sort(source_codes)
            if sort_queries:
                sorted_query_codes, sorted_query_indices = torch.sort(query_codes)
                nearest_right_positions = torch.searchsorted(sorted_source_codes, sorted_query_codes, right=True)
            else:
                nearest_right_positions = torch.searchsorted(sorted_source_codes, query_codes, right=True)

            k = int(k_neighbors/2)  # Number of neighbors in each direction
            front_indices = nearest_right_positions.unsqueeze(1) - torch.arange(1, k+1).to(nearest_right_positions.device).unsqueeze(0)
            back_indices = nearest_right_positions.unsqueeze(1) + torch.arange(0, k).to(nearest_right_positions.device).unsqueeze(0)

            # Combine front and back indices
            neighbor_indices = torch.cat((front_indices, back_indices), dim=1)

            # Pad indices that are out of range by -1
            neighbor_indices = torch.where((neighbor_indices >= 0) & (neighbor_indices < len(sorted_source_codes)), neighbor_indices, torch.tensor(-1))

            # Map the indices back to the original unsorted source codes
            neighbor_idx = torch.where(neighbor_indices != -1, sorted_source_indices[neighbor_indices], torch.tensor(-1))

            if len(inverse_index) > 0:
                neighbor_idx = neighbor_idx[inverse_index]

            all_neighbor_idx.append(neighbor_idx)
        
        # average_recall = self.compute_average_recall(ref_indices, all_neighbor_idx)
        # print(f"Average Recall: {average_recall}")
        return torch.cat(all_neighbor_idx, dim=-1)

    def mink_neighbor(self, sparse_tensor, sparse_coords, query_xyz, tensor_stride, knn_indices):
        ref_indices = knn_indices
        voxel_coords, index, inverse_index = ME.utils.sparse_quantize(coordinates=query_xyz, return_index=True, return_inverse=True, quantization_size=(self.voxel_size*tensor_stride), device=query_xyz.device)
        voxel_coords = ME.utils.batched_coordinates(
        [voxel_coords], dtype=voxel_coords.dtype, 
        device=voxel_coords.device)  
        # query_sp = ME.SparseTensor(features = voxel_coords, coordinates=voxel_coords, tensor_stride = tensor_stride)
        cm = sparse_tensor.coordinate_manager
        cm2 = sparse_tensor.coordinate_manager

        query_key, (unique_map, inverse_map)= cm.insert_and_map(
            voxel_coords,
            tensor_stride=1,
            string_id='query',
        ) 
        strided_coords = (sparse_coords /tensor_stride).to(torch.int)
        # source_key = sparse_tensor.coordinate_map_key
        source_key, (unique_map, inverse_map) = cm2.insert_and_map(
            strided_coords,
            tensor_stride=1,
            string_id='source',
        )
        kernel_map = cm.kernel_map(
            source_key,
            query_key,
            kernel_size=2,
            region_type=ME.RegionType.HYPER_CUBE, # default 0 hypercube
        )  
        neighbor_idx = torch.full((voxel_coords.shape[0], 8), -1, dtype=torch.long).to(query_xyz.device)
        for key in kernel_map.keys():
            in_out = kernel_map[key].long()
            neighbor_idx[in_out[1], key] = in_out[0]
        
        if len(inverse_index) > 0:
            neighbor_idx = neighbor_idx[inverse_index]

        # average_recall = self.compute_average_recall(ref_indices, [neighbor_idx])
        # print(f"Average Recall: {average_recall} for tensor stride {tensor_stride}")
        return neighbor_idx

    def pt_mink_neighbor(self, sparse_voxel_centers, query_xyz, tensor_stride, knn_indices):
        ref_indices = knn_indices
        voxel_coords, index, inverse_index = ME.utils.sparse_quantize(coordinates=query_xyz, return_index=True, return_inverse=True, quantization_size=(self.voxel_size*tensor_stride), device=query_xyz.device)
        voxel_coords = ME.utils.batched_coordinates(
        [voxel_coords], dtype=voxel_coords.dtype, 
        device=voxel_coords.device)  

        query_sp = ME.SparseTensor(features = voxel_coords, coordinates=voxel_coords, tensor_stride = tensor_stride)
        cm = query_sp.coordinate_manager
        cm2 = query_sp.coordinate_manager

        source_voxel_coords, source_index, source_inverse_index = ME.utils.sparse_quantize(coordinates=sparse_voxel_centers, return_index=True, return_inverse=True, quantization_size=(self.voxel_size*tensor_stride), device=sparse_voxel_centers.device)
        source_voxel_coords = ME.utils.batched_coordinates(
        [source_voxel_coords], dtype=source_voxel_coords.dtype,
        device=source_voxel_coords.device)

        # source_voxel_coords = torch.floor(sparse_voxel_centers / (self.voxel_size*tensor_stride)).to(torch.int)
        # source_voxel_batch_id = torch.zeros(source_voxel_coords.shape[0], dtype=torch.int, device=source_voxel_coords.device)
        # source_voxel_coords = torch.cat([source_voxel_batch_id.unsqueeze(1), source_voxel_coords], dim=1)

        query_key, (unique_map, inverse_map)= cm.insert_and_map(
            voxel_coords,
            tensor_stride=1,
            string_id='query',
        ) 

        # source_key = sparse_tensor.coordinate_map_key
        source_key, (unique_map, inverse_map) = cm2.insert_and_map(
            source_voxel_coords,
            tensor_stride=1,
            string_id='source',
        )
        kernel_map = cm.kernel_map(
            source_key,
            query_key,
            kernel_size=2,
            region_type=ME.RegionType.HYPER_CUBE, # default 0 hypercube
        )  
        neighbor_idx = torch.full((voxel_coords.shape[0], 8), -1, dtype=torch.long).to(query_xyz.device)
        for key in kernel_map.keys():
            in_out = kernel_map[key].long()
            neighbor_idx[in_out[1], key] = in_out[0]
        
        if len(inverse_index) > 0:
            neighbor_idx = neighbor_idx[inverse_index]

        source_index[-1] = -1
        neighbor_idx = source_index[neighbor_idx]
        # average_recall = self.compute_average_recall(ref_indices, [neighbor_idx])
        # print(f"Average Recall: {average_recall} for tensor stride {tensor_stride}")
        return neighbor_idx
    
    def forward(self, encoder_outputs: list, query_xyz: Tensor) -> Tensor:
        # encoder_outputs: list of multi-scale Minkowski Tensors
        # voxel_centers (N, 3)
        # query_xyz (M, 3)
        # index (M, K), padding with negative values

        s_features = []
        visual_centers = None
        if self.backbone == 'PointTransformerV3':
            batch_size = encoder_outputs.batch.max().item() + 1
            query_points_per_batch = query_xyz.shape[0] // batch_size  # Points per batch assuming equal distribution
            s_features = []
            scale_mask = []
            knn_time = 0
            attentive_pooling_time = 0
            interpolation_time = 0
            after_layers_time = 0
            voxel_num = 0
            current_point = encoder_outputs
            batched_latents = []
            scale = 0
            strides = self.stride

            for layer in range(self.last_n_layers):
                if layer > 0:
                    scale += int(strides[layer-1] / 2)
                point_centers = current_point.coord
                point_features = current_point.feat
                point_batches = current_point.batch
                current_point = current_point['unpooling_parent']                                  
                latents = self.per_scale_in[layer](point_features) if self.per_scale_in else point_features

                if layer == self.scale_visualization:
                    visual_centers = point_centers

                offset = 0
                all_indices = []
                # torch.cuda.empty_cache()
                knn_start = time.time()

                for b in range(batch_size):
                    batch_mask = (point_batches == b)
                    batch_point_centers = point_centers[batch_mask]
                    batch_latents = latents[batch_mask]

                    batch_start_idx = b * query_points_per_batch
                    batch_end_idx = (b + 1) * query_points_per_batch
                    batch_query_xyz = query_xyz[batch_start_idx:batch_end_idx]

                    if self.neighboring == 'KNN':
                        knn_output = knn_points(batch_query_xyz.unsqueeze(0).to(torch.device("cuda")),
                                                batch_point_centers.unsqueeze(0).to(torch.device("cuda")),
                                                K=self.k_neighbors)
                        indices = knn_output.idx.squeeze(0)
                        indices = indices.to(latents.device)
                    
                    elif self.neighboring == 'Serial':
                        indices = self.pt_serial_neighbor(batch_point_centers, batch_query_xyz)

                    elif self.neighboring == 'Mink':

                        indices = self.pt_mink_neighbor(batch_point_centers, batch_query_xyz, 2**layer, None)

                    elif self.neighboring == 'Mixture':
                        if layer < self.serial_neighbor_layers:
                            indices = self.pt_serial_neighbor(batch_point_centers, batch_query_xyz)
                        else:
                            knn_output = knn_points(batch_query_xyz.unsqueeze(0).to(torch.device("cuda")),
                                                    batch_point_centers.unsqueeze(0).to(torch.device("cuda")),
                                                    K=self.k_neighbors)
                            indices = knn_output.idx.squeeze(0)
                            indices = indices.to(latents.device)
                    # Adjust indices to global index space
                    if offset > 0:
                        indices[indices!=-1] += offset
                    offset += batch_point_centers.shape[0]
                    all_indices.append(indices)

                knn_end = time.time()
                knn_time += knn_end - knn_start
                # torch.cuda.empty_cache()

                batched_indices = torch.cat(all_indices, dim=0)

                # Compute interpolated features for all batches at once
                interpolation_time -= time.time()
                if self.multi_scale_aggregation == 'average':
                    interpolated_features, scale_attentive_time, mask = self.interpolation(latents, point_centers, query_xyz, batched_indices, scale, layer)
                    scale_mask.append(mask)
                else:
                    interpolated_features, scale_attentive_time = self.interpolation(latents, point_centers, query_xyz, batched_indices, scale, layer)
                attentive_pooling_time += scale_attentive_time
                interpolation_time += time.time()

                s_features.append(interpolated_features)

            after_layers_time -= time.time()
            if self.multi_scale_aggregation == 'cat':
                features = torch.cat(s_features, dim=1)
                out = self.all_scale_out(features)
            elif self.multi_scale_aggregation == 'attention':
                features = torch.stack(s_features, dim=0)  # Shape: (num_scales, N, C)
                features = features.transpose(0, 1)  # Shape: (N, num_scales, C)   
                features = self.scales_transformer(features)  # Shape: (N, num_scales, C)
                features = features.view(features.size(0), -1)  # Shape: (N, num_scales * C)
                out = self.all_scale_out(features)  # Shape: (N, hidden_dim

            elif self.multi_scale_aggregation == 'sum':
                features = sum(s_features)
                out = self.all_scale_out(features)

            elif self.multi_scale_aggregation == 'average':
                scale_mask = torch.stack(scale_mask, dim=0)
                valid_scale_count = scale_mask.sum(dim=0)
                features = sum(s_features) / (valid_scale_count.unsqueeze(-1) + 1e-8)
                features *= self.last_n_layers
                out = self.all_scale_out(features)

            # Gating mechanism forward pass
            elif self.multi_scale_aggregation == 'gate':
                features = torch.stack(s_features, dim=1)  # Shape: (N, num_scales, C)
                gate_weights = self.sigmoid(self.gates)  # Shape: (1, num_scales, 1)
                features = features * gate_weights  # Shape: (N, num_scales, C)
                features = features.sum(dim=1)  # Shape: (N, C)
                out = self.all_scale_out(features)
            
            # Attentive pooling forward pass
            elif self.multi_scale_aggregation == 'attentive_pooling':
                features = torch.stack(s_features, dim=1)
                features = features.view((features.size(0), -1))  # Shape: (N, num_scales, C)
                attn_weights = self.all_scale_attentive_mlp(features)  # Shape: (N, num_scales, C)
                attn_weights = attn_weights.view(features.size(0), self.last_n_layers, -1)  # Shape: (N, num_scales, hidden_dim)
                attn_weights = self.attention_softmax(attn_weights)  # Shape: (N, num_scales, hidden_dim)
                features = features.view(features.size(0), self.last_n_layers, -1)  # Shape: (N, num_scales, C)
                features = features * attn_weights  # Shape: (N, num_scales, C)
                features = features.sum(dim=1)  # Shape: (N, C)
                out = self.all_scale_out(features)

            out = self.out(out)
            after_layers_time += time.time()

            scale = 2
            voxel_size = self.voxel_size * 2 ** scale 
            return out.squeeze(-1), knn_time, attentive_pooling_time, interpolation_time, after_layers_time, visual_centers
        
        else:
            batch_size = int((encoder_outputs[0].C)[:, 0].max().item() + 1)  # Assuming query_xyz includes batch index in column 0
            points_per_batch = query_xyz.shape[0] // batch_size  # Points per batch assuming equal distribution
            s_features = []
            scale_mask = []
            knn_time = 0
            attentive_pooling_time = 0
            interpolation_time = 0
            after_layers_time = 0
            voxel_num = 0
            current_point = encoder_outputs

            for scale in range(self.last_n_layers):
                voxel_size = self.voxel_size * 2 ** scale
                voxel_coords = encoder_outputs[-scale - 1].C
                voxel_features = encoder_outputs[-scale - 1].F
                voxel_centers = voxel_coords[:, 1:4] * self.voxel_size + voxel_size / 2.0

                latents = self.per_scale_in[scale](voxel_features)
                voxel_num += len(voxel_coords)

                batched_voxel_centers = []
                batched_latents = []
                batched_query_xyz = []
                offset = 0
                all_indices = []
                # torch.cuda.empty_cache()
                knn_start = time.time()

                for b in range(batch_size):
                    batch_mask = voxel_coords[:, 0] == b
                    batch_voxel_centers = voxel_centers[batch_mask]
                    batch_latents = latents[batch_mask]
                    batch_voxel_coords = voxel_coords[batch_mask]

                    batch_start_idx = b * points_per_batch
                    batch_end_idx = (b + 1) * points_per_batch
                    batch_query_xyz = query_xyz[batch_start_idx:batch_end_idx]

                    if self.neighboring == 'KNN':
                        knn_output = knn_points(batch_query_xyz.unsqueeze(0).to(torch.device("cuda")),
                                                batch_voxel_centers.unsqueeze(0).to(torch.device("cuda")),
                                                K=self.k_neighbors)
                        indices = knn_output.idx.squeeze(0)
                        indices = indices.to(latents.device)
                        # indices = torch.ones((batch_query_xyz.shape[0], 8), dtype=torch.long).to(latents.device)
                        
                    elif self.neighboring == 'Mink':
                        # knn_output = knn_points(batch_query_xyz.unsqueeze(0).to(torch.device("cuda")),
                        #                         batch_voxel_centers.unsqueeze(0).to(torch.device("cuda")),
                        #                         K=self.k_neighbors)
                        # knn_indices = knn_output.idx.squeeze(0)
                        # knn_indices = knn_indices.to(latents.device)
                        indices = self.mink_neighbor(encoder_outputs[-scale - 1], batch_voxel_coords, batch_query_xyz, 2**scale, None)
                        # indices = torch.ones((batch_query_xyz.shape[0], 8), dtype=torch.long).to(latents.device)
                        # knn_output = knn_points(batch_query_xyz.unsqueeze(0).to(torch.device("cuda")),
                        # batch_voxel_centers.unsqueeze(0).to(torch.device("cuda")),
                        # K=1)
                        # indices = knn_output.idx.squeeze(0)
                    
                    elif self.neighboring == 'Serial':
                        knn_output = knn_points(batch_query_xyz.unsqueeze(0).to(torch.device("cuda")),
                                                batch_voxel_centers.unsqueeze(0).to(torch.device("cuda")),
                                                K=self.k_neighbors)
                        knn_indices = knn_output.idx.squeeze(0)
                        knn_indices = knn_indices.to(latents.device)
                        indices = self.serial_neighbor(batch_voxel_coords, batch_query_xyz, 2**scale, knn_indices)
                    
                    # Adjust indices to global index space
                    if offset > 0:
                        indices[indices!=-1] += offset
                    offset += batch_voxel_centers.shape[0]

                    batched_voxel_centers.append(batch_voxel_centers)
                    batched_latents.append(batch_latents)
                    batched_query_xyz.append(batch_query_xyz)
                    all_indices.append(indices)
                knn_end = time.time()
                knn_time += knn_end - knn_start
                # torch.cuda.empty_cache()

                # Concatenate all batched data
                batched_voxel_centers = torch.cat(batched_voxel_centers, dim=0)
                batched_latents = torch.cat(batched_latents, dim=0)
                batched_query_xyz = torch.cat(batched_query_xyz, dim=0)
                batched_indices = torch.cat(all_indices, dim=0)

                # Compute interpolated features for all batches at once
                interpolation_time -= time.time()
                if self.multi_scale_aggregation == 'average':
                    interpolated_features, scale_attentive_time, mask = self.interpolation(batched_latents, batched_voxel_centers, batched_query_xyz, batched_indices, scale, scale)
                    scale_mask.append(mask)
                else:
                    interpolated_features, scale_attentive_time = self.interpolation(batched_latents, batched_voxel_centers, batched_query_xyz, batched_indices, scale, scale)
                attentive_pooling_time += scale_attentive_time
                interpolation_time += time.time()
                
                s_features.append(interpolated_features)

            after_layers_time -= time.time()
            if self.multi_scale_aggregation == 'cat':
                features = torch.cat(s_features, dim=1)
                out = self.all_scale_out(features)
            elif self.multi_scale_aggregation == 'attention':
                features = torch.stack(s_features, dim=0)  # Shape: (num_scales, N, C)
                features = features.transpose(0, 1)  # Shape: (N, num_scales, C)   
                features = self.scales_transformer(features)  # Shape: (N, num_scales, C)
                features = features.view(features.size(0), -1)  # Shape: (N, num_scales * C)
                out = self.all_scale_out(features)  # Shape: (N, hidden_dim

            elif self.multi_scale_aggregation == 'sum':
                features = sum(s_features)
                out = self.all_scale_out(features)

            elif self.multi_scale_aggregation == 'average':
                scale_mask = torch.stack(scale_mask, dim=0)
                valid_scale_count = scale_mask.sum(dim=0)
                features = sum(s_features) / (valid_scale_count.unsqueeze(-1) + 1e-8)
                features *= self.last_n_layers
                out = self.all_scale_out(features)

            out = self.out(out)
            after_layers_time += time.time()

            scale = 0
            if self.scale_visualization != -1:
                scale = self.scale_visualization
                
            voxel_size = self.voxel_size * 2 ** scale 
            # voxel_coords = encoder_outputs[-scale - 1].C
            # latents = self.per_scale_in[scale](encoder_outputs[-scale - 1].F)
            
            # Calculate voxel centers considering batch dimension
            # if scale == 0:
            voxel_centers = voxel_coords[:, 1:4] * self.voxel_size + voxel_size / 2.0
            # return out.squeeze(-1), knn_time, voxel_centers
            # print(f'num_voxels: {voxel_num}')
            return out.squeeze(-1), knn_time, attentive_pooling_time, interpolation_time, after_layers_time, voxel_centers