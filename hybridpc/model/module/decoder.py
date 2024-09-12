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

class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out, activation_fn):
        super().__init__()
        self.fc = nn.Linear(d_in, d_in)
        self.mlp = ActivationConv1d(d_in, d_out, kernel_size=1,bn=True, activation=activation_fn)

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=1) # M, K, hidden_dim+feature_dim
        f_agg = feature_set * att_scores #M, K, hidden_dim + feature_dim
        f_agg = torch.sum(f_agg, dim=1, keepdim=True) # M, 1, hidden_dim + feature_dim
        f_agg = self.mlp(f_agg) #M, 1, hidden_dim + feature_dim
        return f_agg #M, 1, hidden_dim

# class Self_attention(nn.Module):
#     def __init__(self, d_in, num_heads, num_layers):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_in, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         # x has shape (N, K, C)
#         N, K, C = x.shape
        
#         x = x.permute(1, 0, 2)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)
        
#         # Pool across the K dimension to get a single feature per query point
#         # Here we use mean pooling as an example
#         x = x.mean(dim=1, keepdim=True)  # Shape: (N, 1, C)
        
#         return x
    
class EncMLP(nn.Module):
    def __init__(self, ENC_DIM, k_neighbors):
        super(EncMLP, self).__init__()
        self.ENC_DIM = ENC_DIM
        self.linear1 = nn.Linear(3, self.ENC_DIM, bias=False)
        self.bn1 = nn.BatchNorm1d(self.ENC_DIM*k_neighbors)
        self.linear2 = nn.Linear(self.ENC_DIM, self.ENC_DIM, bias=False)
        self.bn2 = nn.BatchNorm1d(self.ENC_DIM*k_neighbors)
        self.tanh = nn.Tanh()

    def forward(self, x):
        N = x.shape[0]
        x = self.linear1(x)
        # Transpose N and K so that batch normalization is applied correctly
        x = self.bn1(x.view(N, -1)).view(N, -1, self.ENC_DIM)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.bn1(x.view(N, -1)).view(N, -1, self.ENC_DIM)

        return self.tanh(x)

class BaseDecoder(pl.LightningModule):
    def __init__(
        self,
        decoder_cfg,
        supervision,
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
        self.in_layer = nn.Sequential(nn.Linear(self.feature_dim + self.enc_dim, self.hidden_dim), self.activation_fn)

        self.skip_proj = nn.Sequential(nn.Linear(self.feature_dim + self.enc_dim, self.hidden_dim), self.activation_fn)

        before_skip = []
        self.num_hidden_layers_before = kwargs['decoder_cfg'].num_hidden_layers_before
        self.num_hidden_layers_after = kwargs['decoder_cfg'].num_hidden_layers_after
        for _ in range(self.num_hidden_layers_before):
            before_skip.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn))
        self.before_skip = nn.Sequential(*before_skip)

        after_skip = []
        for _ in range(self.num_hidden_layers_after):
            after_skip.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn))
        after_skip.append(nn.Linear(self.hidden_dim, self.out_dim))
        after_skip.append(self.activation_fn)
        self.after_skip = nn.Sequential(*after_skip)

    def gradient_hook(self, module, grad_input, grad_output):
        """ used for debugging gradient """
        module.stored_gradients = grad_output
    def forward(self, latents: Tensor, coords: Tensor, index: Tensor) -> Tensor:
        # latents (M, C)
        # coords (N, D2)
        # index (N, )
        coords = self.coords_enc.embed(coords/ self.voxel_size)
        selected_latents = latents[index]
        emb_and_coords = torch.cat([selected_latents, coords], dim=-1)

        x = self.in_layer(emb_and_coords)
        x = self.before_skip(x)

        inp_proj = self.skip_proj(emb_and_coords)
        x = x + inp_proj

        x = self.after_skip(x)

        return x.squeeze(-1)
    
class SimpleDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ENC_DIM = 32
        self.enc_mlp = nn.Sequential(
            nn.Linear(3, self.ENC_DIM, bias=False),
            nn.BatchNorm1d(self.ENC_DIM),
            nn.Tanh(),
            nn.Linear(self.ENC_DIM, self.ENC_DIM, bias=False),
            nn.BatchNorm1d(self.ENC_DIM),
            nn.Tanh()
        )
        self.final = nn.Sequential(
            nn.Linear(self.feature_dim + self.ENC_DIM, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            self.activation_fn,
            nn.Linear(self.feature_dim, self.out_dim),
            self.activation_fn
        )
    
    def forward(self, latents: Tensor, coords: Tensor, index: Tensor) -> Tensor:
        # latents (M, C)
        # coords (norm_points)
        pos_embs = self.enc_mlp(coords)
        selected_latents = latents[index]

        emb_and_coords = torch.cat([selected_latents, pos_embs], dim=-1)
        out = self.final(emb_and_coords)

        return out.squeeze(-1)

class SimpleInterpolatedDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_heads = 4
        self.enc_mlp = EncMLP(self.ENC_DIM, kwargs['decoder_cfg'].k_neighbors)
        self.architecture = kwargs['decoder_cfg'].architecture
        self.interpolation_mode = kwargs['decoder_cfg'].interpolation_mode
        self.num_hidden_layers_before = kwargs['decoder_cfg'].num_hidden_layers_before
        self.num_hidden_layers_after = kwargs['decoder_cfg'].num_hidden_layers_after
        # self.interpolation_layer = ResnetBlockFC(self.feature_dim+self.ENC_DIM, self.hidden_dim)
        self.interpolation_layer = nn.Sequential(
            nn.Linear(self.feature_dim + self.ENC_DIM, self.hidden_dim),
            nn.ReLU()
        )
    
        if self.architecture == 'ConvONet': 
            """ to be modified """

            self.fc_c = nn.ModuleList([
                nn.Linear(self.feature_dim, self.hidden_dim) for i in range(self.num_hidden_layers_before)
            ])
            self.blocks = nn.ModuleList([
                ResnetBlockFC(self.hidden_dim) for i in range(self.num_hidden_layers_before)
            ])
            self.fc_p = nn.Linear(self.enc_dim, self.hidden_dim)
            after_skip = []
            for _ in range(self.num_hidden_layers_after):
                after_skip.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn))
            after_skip.append(nn.Linear(self.hidden_dim, self.out_dim))
            after_skip.append(self.activation_fn)
            self.after_skip = nn.Sequential(*after_skip)

        elif self.architecture == 'CrossAttention':
            """ to be modified """
            self.fc_c = nn.Linear(self.feature_dim, self.hidden_dim)
            self.fc_p = nn.Linear(self.enc_dim, self.hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=num_heads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_hidden_layers_before)
            after_skip = []
            for _ in range(self.num_hidden_layers_after):
                after_skip.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn))
            after_skip.append(nn.Linear(self.hidden_dim, self.out_dim))
            after_skip.append(self.activation_fn)
            self.after_skip = nn.Sequential(*after_skip)

        elif self.architecture == 'SimpleInterpolated':
            self.final = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                nn.BatchNorm1d(self.hidden_dim),
                self.activation_fn,
                nn.Linear(self.hidden_dim, self.out_dim),
                self.activation_fn

            )

        else:
            print("Decoder type not supported")




    def interpolation(self, voxel_latents: Tensor, coords_embed: Tensor, coords: Tensor, index: Tensor):
        """Interpolates voxel features for a given set of points.

        The function calculates interpolated features based on the voxel latent codes and the indices 
        of the nearest voxels for each point. The interpolation takes into account the spatial 
        proximity of the nearest voxels to each point.

        Args:
            voxel_latents (Tensor): A tensor containing voxel latent codes. 
                It has the shape (M, D), where M is the number of voxels and D is the dimension of the latent space.

            coords (Tensor): A normalized tensor containing the neighbor coordinates of sampled points.
                It has the shape (N, K, 3), where N is the number of sampled points and each point is represented by its relative coordinates to multiple voxel center

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
            gathered_features = self.interpolation_layer(torch.cat((gathered_latents, coords_embed), 2))

        # Calculate the weights for interpolation based on distances
        # Here, we simply use the inverse of the distance. Closer voxels have higher weights.
        if self.interpolation_mode =='inverse_distance':
            distances = torch.norm(coords, dim=2)
            weights = 1.0 / (distances + 1e-8) # Adding a small constant to avoid division by zero
            padding_mask = (index < 0)  # shape (N, K), with True for padded indices
            weights[padding_mask] = 1e-10  # or another small number close to zero
            normalized_weights = weights / torch.sum(weights, dim=1, keepdim=True) # Normalize the weights to sum up to 1
            
            # Compute the interpolated features
            interpolated_features = torch.sum(gathered_features * normalized_weights.unsqueeze(-1), dim=1) #N, D

        elif self.interpolation_mode == 'max':
            interpolated_features, _ = torch.max(gathered_features, dim=1) #N, D

        return interpolated_features

    def forward(self, latents: Tensor, neighbor_coords: Tensor, index: Tensor) -> Tensor:
        # latents (M, C)
        # neighbor_coords (N, K, 3)
        # index (N, K), padding with negative values

        # compute positional encoding
        # enc_coords = self.coords_enc.embed(neighbor_coords/ self.voxel_size)
        enc_coords = self.enc_mlp(neighbor_coords / self.voxel_size)
        if self.architecture == 'Attention':
            interpolated_features = self.interpolation(latents, enc_coords, neighbor_coords, index) # N, C


        elif self.architecture == 'ConvONet':
            """ to be modified """
            interpolated_features = self.interpolation(latents, enc_coords, neighbor_coords, index) # N, C
            net = self.fc_p(enc_coords) # N, hidden_dim
            for i in range(self.num_hidden_layers_before):
                net = net + self.fc_c[i](interpolated_features)
                net = self.blocks[i](net)
        
        elif self.architecture == 'SimpleInterpolated':
            interpolated_features = self.interpolation(latents, enc_coords, neighbor_coords, index) # N, C
            out = self.final(interpolated_features)
            return out.squeeze(-1)
        
        else:
            print("Decoder type not supported")
            
        out = self.after_skip(net)
        return out.squeeze(-1)

        
class InterpolatedDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.interpolation_mode = kwargs['decoder_cfg'].interpolation_mode
        self.enc_mlp = EncMLP(self.ENC_DIM, kwargs['decoder_cfg'].k_neighbors)

        self.k_neighbors = kwargs['decoder_cfg'].k_neighbors
        self.num_hidden_layers_after = kwargs['decoder_cfg'].num_hidden_layers_after
        self.num_hidden_layers_before = kwargs['decoder_cfg'].num_hidden_layers_before

        
        if self.interpolation_mode == 'inverse_distance':
            self.fc = nn.Linear(self.feature_dim+self.hidden_dim, self.feature_dim+self.hidden_dim)
            self.mlp = ActivationConv1d(self.feature_dim+self.hidden_dim, self.hidden_dim, kernel_size=1, bn=self.use_bn, activation=self.activation_fn)
        else:
            if self.architecture == 'Concatenate':
                self.att_pooling = Att_pooling(self.enc_dim+self.feature_dim, self.enc_dim+self.feature_dim, self.activation_fn)
            else:
                self.att_pooling = Att_pooling(self.hidden_dim+self.feature_dim, self.hidden_dim, self.activation_fn)

        # self.interpolation_layer = ResnetBlockFC(self.feature_dim+self.ENC_DIM, self.hidden_dim)
        if self.architecture == 'Concatenate':
            self.in_layer = nn.Sequential(nn.Linear(self.feature_dim + self.enc_dim, self.hidden_dim), self.activation_fn)
            self.skip_proj = nn.Sequential(nn.Linear(self.feature_dim + self.enc_dim, self.hidden_dim), self.activation_fn)

            before_skip = []
            for _ in range(self.num_hidden_layers_before):
                before_skip.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn))
            self.before_skip = nn.Sequential(*before_skip)

            after_skip = []
            for _ in range(self.num_hidden_layers_after):
                after_skip.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn))
            after_skip.append(nn.Linear(self.hidden_dim, self.out_dim))
            if self.supervision == 'UDF':
                after_skip.append(self.activation_fn)
            else:
                after_skip.append(nn.Tanh())
            self.after_skip = nn.Sequential(*after_skip)
        else:
            self.interpolation_layer = ActivationConv1d(10, self.hidden_dim, kernel_size=1,bn=self.use_bn, activation=self.activation_fn)
            finals=[]
            in_channel=self.hidden_dim + 3
            for i in range(self.decoder_cfg.num_hidden_layers_after):
                out_channel=self.decoder_cfg.after_dims[i]
                finals.append(ActivationConv1d(in_channel, out_channel, kernel_size=1, bn=self.use_bn, activation=self.activation_fn))
                in_channel=out_channel

            if self.supervision == 'UDF':
                end=ActivationConv1d(in_channel,1,kernel_size=1, bn=False, activation=nn.ReLU())
            else:
                end=ActivationConv1d(in_channel,1,kernel_size=1, bn=False, activation=nn.Tanh())
            finals.append(end)
            self.final = nn.Sequential(*finals)
        
    def interpolation(self, latents: Tensor, voxel_centers: Tensor, query_xyz: Tensor, index: Tensor) :
        gathered_centers = voxel_centers[index] # M, K, 3
        gathered_query_xyz = query_xyz.unsqueeze(1).repeat(1, index.shape[-1], 1) # M, K, 3
        gathered_relative_coords = gathered_query_xyz - gathered_centers # M, K, 3
        gathered_latents = latents[index] # M, K, C
        if self.architecture == 'Concatenate':
            gathered_coords = self.coords_enc.embed(gathered_relative_coords/ self.voxel_size)
            gathered_emb_and_coords = torch.cat([gathered_latents, gathered_coords], dim=-1) # M, K, C + enc_dim
            padding_mask = (index < 0)  # shape (N, K), with True for padded indices
            gathered_emb_and_coords[padding_mask.unsqueeze(-1).repeat(1, 1, self.feature_dim+self.enc_dim)] = 0
            interpolated_features = self.att_pooling(gathered_emb_and_coords) #M, 1, hidden_dim
            return interpolated_features
            
        gathered_distance = torch.norm(gathered_relative_coords, dim=-1, keepdim=True) # M, K, 1
        gathered_pos_encoding = self.interpolation_layer(torch.cat((gathered_distance, gathered_relative_coords, gathered_query_xyz, gathered_centers), -1)) # M, K, hidden_dim
        gathered_features = torch.cat((gathered_latents, gathered_pos_encoding), -1) # M, K, C+hidden_dim

        # Calculate the weights for interpolation based on distances
        # Here, we simply use the inverse of the distance. Closer voxels have higher weights.
        if self.interpolation_mode =='inverse_distance':
            distances = torch.norm(gathered_relative_coords, dim=2)
            weights = 1.0 / (distances + 1e-8) # Adding a small constant to avoid division by zero
            padding_mask = (index < 0)  # shape (N, K), with True for padded indices
            weights[padding_mask] = 1e-10  # or another small number close to zero
            normalized_weights = weights / torch.sum(weights, dim=1, keepdim=True) # Normalize the weights to sum up to 1
            
            # Compute the interpolated features
            interpolated_features = torch.sum(self.fc(gathered_features) * normalized_weights.unsqueeze(-1), dim=1, keepdim=True) #M, 1, hidden_dim+latent_dim
            interpolated_features = self.mlp(interpolated_features) #M, 1, hidden_dim

        elif self.interpolation_mode == 'attention':
            padding_mask = (index < 0)  # shape (N, K), with True for padded indices
            gathered_features[padding_mask.unsqueeze(-1).repeat(1, 1, self.feature_dim+self.hidden_dim)] = 0
            interpolated_features = self.att_pooling(gathered_features) #M, 1, hidden_dim

        # elif self.interpolation_mode =='zero':
        #     interpolated_features = self.mlp(interpolated_features)       

        return interpolated_features

    def forward(self, latents: Tensor, voxel_coords: Tensor, query_xyz: Tensor, index: Tensor) -> Tensor:
        # latents (M, C)
        # voxel_centers (N, 3)
        # query_xyz (M, 3)
        # index (M, K), padding with negative values

        voxel_centers = voxel_coords[:, 1:4] * self.voxel_size + self.voxel_size / 2.0 
        interpolated_features = self.interpolation(latents, voxel_centers, query_xyz, index) # M, 1, hidden_dim
        
        if self.architecture == "Concatenate":
            interpolated_emb_and_coords = interpolated_features.squeeze(1)
            x = self.in_layer(interpolated_emb_and_coords) # M, hidden_dim
            x = self.before_skip(x)

            inp_proj = self.skip_proj(interpolated_emb_and_coords)
            x = x + inp_proj

            x = self.after_skip(x)
            return x.squeeze(-1)

        else:
            interpolated_features = torch.cat((interpolated_features, query_xyz.unsqueeze(1)), dim=-1)# M, 1, 3+hidden_dim
            out = self.final(interpolated_features) # M, 1, 1
            return out.squeeze(-1).squeeze(-1) # M

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
    
    
class MultiScaleInterpolatedDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.interpolation_mode = kwargs['decoder_cfg'].interpolation_mode
        self.enc_mlp = EncMLP(self.ENC_DIM, kwargs['decoder_cfg'].k_neighbors)

        self.last_n_layers = kwargs['decoder_cfg'].last_n_layers
        self.k_neighbors = kwargs['decoder_cfg'].k_neighbors
        self.num_hidden_layers_after = kwargs['decoder_cfg'].num_hidden_layers_after
        self.num_hidden_layers_before = kwargs['decoder_cfg'].num_hidden_layers_before
        self.multi_scale_aggregation = kwargs['decoder_cfg'].multi_scale_aggregation

        self.per_scale_in = nn.ModuleList([
            nn.Sequential(nn.Linear(self.feature_dim * (scale + 1), self.feature_dim), nn.ReLU())
            for scale in range(self.last_n_layers)
        ])
        
        self.per_scale_att_pooling = nn.ModuleList([
            Att_pooling(self.enc_dim+self.feature_dim, self.enc_dim+self.feature_dim, self.activation_fn)
            for _ in range(self.last_n_layers)
        ])

        self.per_scale_out = nn.ModuleList([
            ResMLPBlock(
                self.feature_dim + self.enc_dim, self.hidden_dim, self.activation_fn, self.num_hidden_layers_before, self.num_hidden_layers_after
            ) for _ in range(self.last_n_layers)
        ])

        multi_scale_dim = self.hidden_dim * self.last_n_layers if self.multi_scale_aggregation == 'cat' else self.hidden_dim
        if self.supervision == 'UDF':
            self.out = nn.Sequential(
                nn.Linear(multi_scale_dim, multi_scale_dim),  
                nn.Linear(multi_scale_dim, 1), 
                self.activation_fn
            )
        if self.supervision == 'Distance':
            self.out = nn.Sequential(
                nn.Linear(multi_scale_dim, multi_scale_dim), 
                nn.Linear(multi_scale_dim, 1), 
                self.activation_fn
            )
        else: 
            self.out = nn.Sequential(
                nn.Linear(multi_scale_dim, multi_scale_dim), 
                nn.Linear(multi_scale_dim, 1), 
                nn.Tanh()
            )
        
    def interpolation(self, latents: Tensor, voxel_centers: Tensor, query_xyz: Tensor, index: Tensor, scale: int) :
        gathered_centers = voxel_centers[index] # M, K, 3
        gathered_query_xyz = query_xyz.unsqueeze(1).repeat(1, index.shape[-1], 1) # M, K, 3
        gathered_relative_coords = gathered_query_xyz - gathered_centers # M, K, 3
        gathered_latents = latents[index] # M, K, C
        gathered_coords = self.coords_enc.embed(gathered_relative_coords/ (self.voxel_size * 2**scale))
        gathered_emb_and_coords = torch.cat([gathered_latents, gathered_coords], dim=-1) # M, K, C + enc_dim
        padding_mask = (index < 0)  # shape (N, K), with True for padded indices
        gathered_emb_and_coords[padding_mask.unsqueeze(-1).repeat(1, 1, self.feature_dim+self.enc_dim)] = 0
        interpolated_features = self.per_scale_att_pooling[scale](gathered_emb_and_coords) #M, 1, hidden_dim
        return interpolated_features

    def mink_neighbor(self, sparse_tensor, query_xyz, tensor_stride):
        voxel_coords, index, inverse_index = ME.utils.sparse_quantize(coordinates=query_xyz, return_index=True, return_inverse=True, quantization_size=(self.voxel_size*tensor_stride))
        voxel_coords = ME.utils.batched_coordinates(
        [voxel_coords], dtype=voxel_coords.dtype, 
        device=voxel_coords.device)  
        query_sp = ME.SparseTensor(features = voxel_coords, coordinates=voxel_coords, tensor_stride = tensor_stride)
        cm = sparse_tensor.coordinate_manager
        source_key = sparse_tensor.coordinate_map_key
        query_key = query_sp.coordinate_map_key
        kernel_map = cm.kernel_map(
            source_key,
            query_key,
            kernel_size=1,
            region_type=ME.RegionType.HYPER_CUBE, # default 0 hypercube
        )  
        neighbor_idx = torch.full((voxel_coords.shape[0], 1), -1, dtype=torch.long).to(query_xyz.device)
        for key in kernel_map.keys():
            in_out = kernel_map[key].long()
            neighbor_idx[in_out[0], key] = in_out[1]
        
        neighbor_map = neighbor_idx[inverse_index]
        return neighbor_map

    def forward(self, encoder_outputs: list, query_xyz: Tensor) -> Tensor:
        # encoder_outputs: list of multi-scale Minkowski Tensors
        # voxel_centers (N, 3)
        # query_xyz (M, 3)
        # index (M, K), padding with negative values

        s_features = []
        batch_size = int((encoder_outputs[0].C)[:, 0].max().item() + 1)  # Assuming query_xyz includes batch index in column 0
        points_per_batch = query_xyz.shape[0] // batch_size  # Points per batch assuming equal distribution
        s_features = []
        num_voxels = 0

        for scale in range(self.last_n_layers):
            voxel_size = self.voxel_size * 2 ** scale
            voxel_coords = encoder_outputs[-scale - 1].C
            num_voxels += voxel_coords.shape[0]
            latents = self.per_scale_in[scale](encoder_outputs[-scale - 1].F)
            
            # Calculate voxel centers considering batch dimension
            # if scale == 0:
            voxel_centers = voxel_coords[:, 1:4] * self.voxel_size + voxel_size / 2.0
            # else:
            #     voxel_centers = voxel_coords[:, 1:4] * self.voxel_size

            batched_voxel_centers = []
            batched_latents = []
            batched_query_xyz = []
            offset = 0
            all_indices = []

            for b in range(batch_size):
                batch_mask = voxel_coords[:, 0] == b
                batch_voxel_centers = voxel_centers[batch_mask]
                batch_latents = latents[batch_mask]

                batch_start_idx = b * points_per_batch
                batch_end_idx = (b + 1) * points_per_batch
                batch_query_xyz = query_xyz[batch_start_idx:batch_end_idx]

                # Perform KNN for the current batch
                # if scale==0:
                knn_output = knn_points(batch_query_xyz.unsqueeze(0).to(torch.device("cuda")),
                                        batch_voxel_centers.unsqueeze(0).to(torch.device("cuda")),
                                        K=self.k_neighbors)
                indices = knn_output.idx.squeeze(0)
                # else:
                #     indices = self.mink_neighbor(encoder_outputs[-scale-1], batch_query_xyz, 2**scale)
                    # knn_output = knn_points(batch_query_xyz.unsqueeze(0).to(torch.device("cuda")),
                    # batch_voxel_centers.unsqueeze(0).to(torch.device("cuda")),
                    # K=1)
                    # indices = knn_output.idx.squeeze(0)

                # indices = torch.ones(batch_query_xyz.shape[0], self.k_neighbors, dtype=torch.long).to(torch.device("cuda"))

                # Adjust indices to global index space
                if offset > 0:
                    indices += offset
                offset += batch_voxel_centers.shape[0]

                batched_voxel_centers.append(batch_voxel_centers)
                batched_latents.append(batch_latents)
                batched_query_xyz.append(batch_query_xyz)
                all_indices.append(indices)

            # Concatenate all batched data
            batched_voxel_centers = torch.cat(batched_voxel_centers, dim=0)
            batched_latents = torch.cat(batched_latents, dim=0)
            batched_query_xyz = torch.cat(batched_query_xyz, dim=0)
            batched_indices = torch.cat(all_indices, dim=0)

            # Compute interpolated features for all batches at once
            # if scale ==0:
            interpolated_features = self.interpolation(batched_latents, batched_voxel_centers, batched_query_xyz, batched_indices, scale)
            # else:
                # interpolated_features = batched_latents[batched_indices[:, 0]]
            
            x = self.per_scale_out[scale](interpolated_features.squeeze(1))
            s_features.append(x)

        if self.multi_scale_aggregation == 'cat':
            features = torch.cat(s_features, dim=1)
        else:
            features = sum(s_features)

        out = self.out(features)
        print(f'num_voxels: {num_voxels}')
        return out.squeeze(-1)