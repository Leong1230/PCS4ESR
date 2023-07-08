import functools
import torch.nn as nn
import pytorch_lightning as pl
import MinkowskiEngine as ME
from hybridpc.model.module.common import ResidualBlock, UBlock


class Backbone(pl.LightningModule):
    def __init__(self, input_channel, output_channel, block_channels, block_reps):
        super().__init__()

        sp_norm = functools.partial(ME.MinkowskiBatchNorm)
        norm = functools.partial(nn.BatchNorm1d)

        # 1. U-Net
        self.unet = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=input_channel, out_channels=output_channel, kernel_size=3, dimension=3),
            UBlock([output_channel * c for c in block_channels], sp_norm, block_reps, ResidualBlock),
            sp_norm(output_channel),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, voxel_features, voxel_coordinates):
        output_dict = {}
        x = ME.SparseTensor(features=voxel_features, coordinates=voxel_coordinates)
        unet_out = self.unet(x)
        return unet_out #B, C
