import functools
import torch.nn as nn
import pytorch_lightning as pl
import MinkowskiEngine as ME
from hybridpc.model.module.common import ResidualBlock, UBlock


class Backbone(pl.LightningModule):
    def __init__(self, backbone_type, input_channel, output_channel, block_channels, block_reps, sem_classes):
        super().__init__()

        sp_norm = functools.partial(ME.MinkowskiBatchNorm)
        norm = functools.partial(nn.BatchNorm1d)
        self.backbone_type = backbone_type

        # 1. Unet
        self.unet = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=input_channel, out_channels=output_channel, kernel_size=3, dimension=3),
            UBlock([output_channel * c for c in block_channels], sp_norm, block_reps, ResidualBlock),
            sp_norm(output_channel),
            ME.MinkowskiReLU(inplace=True)
        )
        #2. Convs
        layers = []
        for c in block_channels:
            layers.append(ME.MinkowskiConvolution(
                in_channels=input_channel,
                out_channels=output_channel * c,
                kernel_size=3, 
                dimension=3))
            layers.append(sp_norm(output_channel * c))
            layers.append(ME.MinkowskiReLU(inplace=True))
            input_channel = output_channel * c # The output of the current layer is the input for the next one

        self.convs = nn.Sequential(*layers)

        # 2.1 semantic prediction branch
        self.semantic_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, sem_classes)
        )

    def forward(self, voxel_features, voxel_coordinates, v2p_map):
        output_dict = {}
        x = ME.SparseTensor(features=voxel_features, coordinates=voxel_coordinates)
        if self.backbone_type == 'Conv':
            unet_out = self.convs(x)
        else: 
            unet_out = self.unet(x)
        output_dict["point_features"] = unet_out.features[v2p_map]
        output_dict["semantic_scores"] = self.semantic_branch(output_dict["point_features"])
        output_dict["voxel_features"] = voxel_features

        return output_dict
        
