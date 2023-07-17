from pyexpat import features
import time
import os
import numpy as np
import math
import torchmetrics
import torch
from einops import repeat
from torch import Tensor, nn
from typing import Callable, Tuple
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
        embed_dim: int,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layes_before_skip: int,
        num_hidden_layes_after_skip: int,
        out_dim: int,
    ) -> None:
        super().__init__()

        self.coords_enc = CoordsEncoder(in_dim)
        coords_dim = self.coords_enc.out_dim

        self.in_layer = nn.Sequential(nn.Linear(embed_dim + coords_dim, hidden_dim), nn.ReLU())

        self.skip_proj = nn.Sequential(nn.Linear(embed_dim + coords_dim, hidden_dim), nn.ReLU())

        before_skip = []
        for _ in range(num_hidden_layes_before_skip):
            before_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.before_skip = nn.Sequential(*before_skip)

        after_skip = []
        for _ in range(num_hidden_layes_after_skip):
            after_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        after_skip.append(nn.Linear(hidden_dim, out_dim))
        self.after_skip = nn.Sequential(*after_skip)

    def forward(self, embeddings: Tensor, coords: Tensor, index: Tensor) -> Tensor:
        # embeddings (B, D1)
        # coords (N, D2)
        # index (N, )
        coords = self.coords_enc.embed(coords)
        selected_embeddings = embeddings.F[index]

        # selected_embeddings = embeddings[index]
        # Concatenate the selected embeddings and the encoded coordinates
        emb_and_coords = torch.cat([selected_embeddings, coords], dim=-1)

        x = self.in_layer(emb_and_coords)
        x = self.before_skip(x)

        inp_proj = self.skip_proj(emb_and_coords)
        x = x + inp_proj

        x = self.after_skip(x)

        return x.squeeze(-1)

class Dense_Generator(pl.LightningModule):
    def __init__(self, model, num_step, threshold, filter_val):
        super().__init__()
        self.model = model
        self.model.eval()
        self.num_step = num_step
        self.threshold = threshold
        self.filter_val = filter_val

    # def generate_point_cloud(self, data, modulations):
    #     start = time.time()

    #     # ensure data is on the correct device
    #     inputs = data['inputs'].to(self.device)

    #     # freeze model parameters
    #     for param in self.model.parameters():
    #         param.requires_grad = False

    #     # sample_num set to 200000
    #     sample_num = 200000

    #     # Initialize samples in CUDA device
    #     samples_cpu = np.zeros((0, 3))

    #     # Initialize samples and move to CUDA device
    #     samples = torch.rand(1, sample_num, 3).float().to(self.device) * 3 - 1.5
    #     samples.requires_grad = True

    #     encoding = self.model.encoder(inputs)

    #     i = 0
    #     while len(samples_cpu) < num_points:
    #         print('iteration', i)

    #         for j in range(num_steps):
    #             print('refinement', j)
    #             df_pred = torch.clamp(self.model.decoder(data, *encoding), max=self.threshold)
    #             df_pred.sum().backward()
    #             gradient = samples.grad.detach()
    #             samples = samples.detach()
    #             df_pred = df_pred.detach()
    #             inputs = inputs.detach()
    #             samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  
    #             samples = samples.detach()
    #             samples.requires_grad = True

    #         print('finished refinement')

    #         if not i == 0:
    #             # Move samples to CPU, detach from computation graph, convert to numpy array, and stack to samples_cpu
    #             samples_cpu = np.vstack((samples_cpu, samples[df_pred < filter_val].detach().cpu().numpy()))

    #         samples = samples[df_pred < 0.03].unsqueeze(0)
    #         indices = torch.randint(samples.shape[1], (1, sample_num))
    #         samples = samples[[[0, ] * sample_num], indices]
    #         samples += (self.threshold / 3) * torch.randn(samples.shape).to(self.device)  # 3 sigma rule
    #         samples = samples.detach()
    #         samples.requires_grad = True

    #         i += 1
    #         print(samples_cpu.shape)

    #     duration = time.time() - start
    #     return samples_cpu, duration

    # def generate_df(self, data):
    #     # Move the inputs and points to the appropriate device
    #     inputs = data['inputs'].to(self.device)
    #     points = data['point_cloud'].to(self.device)
    #     scale = data['scale']
    #     # Compute the encoding of the input
    #     encoding = self.model.encoder(inputs)

    #     # Compute the predicted distance field for the points using the decoder
    #     df_pred = self.model.decoder(points, *encoding).squeeze(0)
    #     # Scale distance field back w.r.t. original point cloud scale
    #     # The predicted distance field is returned to the cpu
    #     df_pred_cpu = df_pred.detach().cpu().numpy()
    #     df_pred_cpu *= scale.detach().cpu().numpy()

    #     return df_pred_cpu