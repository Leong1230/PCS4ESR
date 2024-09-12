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
from torch.nn.parameter import Parameter
from hybridpc.utils.kernel_points import load_kernels

    
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

class ResMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, activation_fn, num_hidden_layers_before, num_hidden_layers_after):
        super(ResMLPBlock, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self.num_hidden_layers_before = num_hidden_layers_before
        self.num_hidden_layers_after = num_hidden_layers_after

        # Initial transformation layer
        self.in_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            self.activation_fn
        )

        self.skip_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            self.activation_fn
        )

        before_skip = [
            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn)
            for _ in range(self.num_hidden_layers_before)
        ]
        self.before_skip = nn.Sequential(*before_skip)

        after_skip = [
            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn)
            for _ in range(self.num_hidden_layers_after)
        ]
        self.after_skip = nn.Sequential(*after_skip)

    def forward(self, x_in):
        # Apply initial transformation
        x = self.in_layer(x_in)
        x = self.before_skip(x)
        inp_proj = self.skip_proj(x_in)
        x = x + inp_proj
        x = self.after_skip(x)

        return x
    
class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out, activation_fn):
        super().__init__()
        self.fc = nn.Linear(d_in, d_in)
        self.mlp = ActivationConv1d(d_in, d_out, kernel_size=1,bn=True, activation=activation_fn)

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=1) # M, K, hidden_dim+feature_dim
        # att_scores = alpha * att_scores
        att_scores = att_scores / (torch.sum(att_scores, dim=1, keepdim=True) + 1e-5)
        f_agg = feature_set * att_scores #M, K, hidden_dim + feature_dim
        f_agg = torch.sum(f_agg, dim=1, keepdim=True) # M, 1, hidden_dim + feature_dim
        f_agg = self.mlp(f_agg) #M, 1, hidden_dim + feature_dim
        return f_agg.squeeze(1) #M, 1, hidden_dim
    
class BaseDecoder(pl.LightningModule):
    def __init__(
        self,
        decoder_cfg,
        supervision,
        latent_dim: int,
        feature_dim: int,
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
    
class KPDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.last_n_layers = kwargs['decoder_cfg'].last_n_layers
        self.k_neighbors = kwargs['decoder_cfg'].k_neighbors
        self.num_hidden_layers_after = kwargs['decoder_cfg'].num_hidden_layers_after
        self.num_hidden_layers_before = kwargs['decoder_cfg'].num_hidden_layers_before

        self.KP_influence = 'linear'
        self.K = 30
        self.p_dim = 3
        self.KP_extent = 1.2
        self.radius = self.voxel_size * 2 ** self.last_n_layers
        self.fixed_kernel_points = 'center'
        self.aggregation_mode = 'sum'
        self.modulated = True   
        self.kernel_points = self.init_KP()


        self.center_feature_dim = self.latent_dim * self.last_n_layers
        self.offset_feature_dim = self.K * (self.p_dim + 1) # offset and modulation


        self.weights = Parameter(torch.zeros((self.K, self.latent_dim, self.feature_dim), dtype=torch.float32),
                                 requires_grad=True)


        self.center_feature_mlp = ResMLPBlock(
            self.center_feature_dim,  self.offset_feature_dim, self.activation_fn, self.num_hidden_layers_before, 2
        )

        self.att_pooling = Att_pooling(self.enc_dim+self.feature_dim, self.enc_dim+self.feature_dim, self.activation_fn)

        self.mlp = ResMLPBlock(
            self.enc_dim+self.feature_dim,  self.hidden_dim, self.activation_fn, self.num_hidden_layers_before, self.num_hidden_layers_after
        ) 

        if self.supervision == 'UDF':
            self.out = nn.Sequential(
                nn.Linear(self.hidden_dim, 1), 
                self.activation_fn
            )
        if self.supervision == 'Distance':
            self.out = nn.Sequential(
                nn.Linear(self.hidden_dim, 1), 
                self.activation_fn
            )
        else: 
            self.out = nn.Sequential(
                nn.Linear(self.hidden_dim, 1), 
                nn.Tanh()
            )
        

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)
    

    def forward(self, encoder_outputs: list, query_xyz: Tensor) -> Tensor:
        # encoder_outputs: list of multi-scale Minkowski Tensors
        # voxel_centers (N, 3)
        # query_xyz (M, 3)
        # index (M, K), padding with negative values

        # get center position
        scale = self.last_n_layers - 1
        voxel_size = self.voxel_size * 2 ** scale
        center_voxel_coords = encoder_outputs[-scale - 1].C
        center_latents = encoder_outputs[-scale - 1].F
        centers = center_voxel_coords[:, 1:4] * self.voxel_size + voxel_size / 2.0 # N, 3


        # get input positions
        voxel_coords = encoder_outputs[-1].C
        input_latents = encoder_outputs[- 1].F # M, C
        input = voxel_coords[:, 1:4] * self.voxel_size + self.voxel_size / 2.0 # M, 3
        knn_output = knn_points(centers.unsqueeze(0).to(torch.device("cuda")),
                                input.unsqueeze(0).to(torch.device("cuda")),
                                K=8)
        input_center_indices = knn_output.idx.squeeze(0) # N, K
        neighbors = input[input_center_indices] # N, K, 3
        neighb_x= input_latents[input_center_indices] # N, K, C
        K_points = centers.unsqueeze(1) + self.kernel_points.unsqueeze(0)  # N, P, 3
        K_points =K_points.unsqueeze(1) # N, 1, P, 3
        neighbors.unsqueeze_(2)
        differences = neighbors - K_points # N, K, P, 3
        sq_distances = torch.sum(differences ** 2, dim=3) # N, K, P 

        
        if self.modulated:
            # Get offset (in normalized scale) from features
            self.offset_features = self.center_feature_mlp(center_latents)
            unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
            unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)

            # Get modulations
            modulations = 2 * torch.sigmoid(self.offset_features[:, self.p_dim * self.K:])

            # Rescale offset for this layer
            offsets = unscaled_offsets * self.KP_extent

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # gather features in neighbor points
        weighted_features = torch.matmul(all_weights, neighb_x)
        if self.modulated: 
            weighted_features *= modulations.unsqueeze(2) 
        weighted_features = weighted_features.permute((1, 0, 2)) # N, P, C

        kernel_outputs = torch.matmul(weighted_features, self.weights).permute(1, 0, 2) #N, P, C

        # align query_xyz to voxel_centers
        knn_output = knn_points(query_xyz.unsqueeze(0).to(torch.device("cuda")),
                                centers.unsqueeze(0).to(torch.device("cuda")),
                                K=1)
        query_voxel_indices = knn_output.idx.squeeze(0).squeeze(-1) # N1
        
        query_centers = centers[query_voxel_indices] # N1, 3
        query_kp = K_points.squeeze(1)[query_voxel_indices] # N1, P, 3
        query_kernel_features = kernel_outputs[query_voxel_indices] # N1, P, C

        # find 4 nearest kernel points w.r.t each query_xyz
        dists = torch.sum((query_xyz.unsqueeze(1) - query_kp)**2, dim=2)
        _, query_kp_indices = torch.topk(dists, k=4, dim=1, largest=False)
        
        gathered_query_xyz = query_xyz.unsqueeze(1).repeat(1, query_kp_indices.shape[-1], 1) # N1, 4, 3
        gathered_kp_xyz = query_kp[torch.arange(query_kp.size(0)).unsqueeze(1), query_kp_indices] # N1, 4, 3
        gathered_relative_coords = gathered_query_xyz - gathered_kp_xyz # N1, 4, 3
        gathered_latents = query_kernel_features[torch.arange(query_kernel_features.size(0)).unsqueeze(1), query_kp_indices] # N1, 4, C
        gathered_coords = self.coords_enc.embed(gathered_relative_coords/ (self.voxel_size * 2**(self.last_n_layers)))
        gathered_emb_and_coords = torch.cat([gathered_latents, gathered_coords], dim=-1) # M, 4, C + enc_dim
        interpolated_features = self.att_pooling(gathered_emb_and_coords)

        out = self.mlp(interpolated_features)
        out = self.out(out)

        scale = self.last_n_layers-1
        voxel_size = self.voxel_size * 2 ** scale 
        voxel_coords = encoder_outputs[-scale - 1].C
        voxel_centers = voxel_coords[:, 1:4] * self.voxel_size + voxel_size / 2.0
        # return out.squeeze(-1), knn_time, voxel_centers
        return out.squeeze(-1), 0, voxel_centers