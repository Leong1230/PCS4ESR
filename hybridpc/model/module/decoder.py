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
        out_dim: int = 1,
        activation: str = 'ReLU',
    ) -> None:
        super().__init__()

        in_dim = cfg.input_dim
        embed_dim = embed_dim
        hidden_dim = cfg.hidden_dim
        out_dim = out_dim
        num_heads = 4
        self.type = type
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

        elif self.decoder_type == 'CrossAttention':
            self.fc_c = nn.Linear(embed_dim, hidden_dim)
            self.fc_p = nn.Linear(enc_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_hidden_layers_before_skip)
            # self.attention_layers = nn.ModuleList([
            #     nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads) 
            #     for _ in range(self.num_hidden_layers_before_skip)
            # ])
        else:
            self.in_layer = nn.Sequential(nn.Linear(embed_dim + enc_dim, hidden_dim), self.get_activation(activation))
            self.skip_proj = nn.Sequential(nn.Linear(embed_dim + enc_dim, hidden_dim), self.get_activation(activation))
            before_skip = []
            for _ in range(self.num_hidden_layers_before_skip):
                before_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self.get_activation(activation)))
            self.before_skip = nn.Sequential(*before_skip)

        after_skip = []
        for _ in range(self.num_hidden_layers_after_skip):
            after_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self.get_activation(activation)))
        after_skip.append(nn.Linear(hidden_dim, out_dim))
        if self.type == 'functa':
            after_skip.append(nn.LeakyReLU())
        self.after_skip = nn.Sequential(*after_skip)

    def get_activation(self, activation_str: str):
        """Return the desired activation function based on the string."""
        if activation_str == "ReLU":
            act = nn.ReLU()
        elif activation_str == "LeakyReLU":
            act = nn.LeakyReLU()
        elif activation_str == "Softplus":
            act = nn.Softplus()
        elif activation_str == "ShiftedSoftplus":
            def shifted_softplus(input_tensor):
                shifted = input_tensor - 1
                return nn.Softplus()(shifted)
            act = shifted_softplus
    
        return act

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

    def forward(self, embeddings: Tensor, absolute_coords: Tensor, coords: Tensor, index: Tensor) -> Tensor:
        # embeddings (M, C)
        # absolute_coords (N, K, 3)
        # coords (N, 3) or (N, K, 3)
        # index (N, ) or (N, K)

        # compute positional encoding
        if self.decoder_type == 'CrossAttention':
            if self.normalize_encoding:
                enc_coords = self.coords_enc.embed(coords/ self.voxel_size)
            else:
                enc_coords = self.coords_enc.embed(coords) #N, K, C
            gathered_latents = embeddings[index] # N, K, C
            net = self.fc_c(gathered_latents) + self.fc_p(enc_coords)
            net = self.transformer_encoder(net.transpose(0, 1))
            # for attention_layer in self.attention_layers:
            #     attn_output, _ = attention_layer(net, net, net)
            #     net = net + attn_output[:, 0, :]
            out = self.after_skip(net.transpose(0, 1)[:, 0, :])

            return out.squeeze(-1)

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
    
