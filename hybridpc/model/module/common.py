import torch.nn as nn
import torch
import MinkowskiEngine as ME
from collections import OrderedDict
from torch_scatter import scatter_mean, scatter_max, scatter_add
from pytorch3d.ops import knn_points
import pytorch_lightning as pl

class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name="",
            instance_norm=False,
            instance_norm_func=None
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size, eps=1e-6, momentum=0.99))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class _BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)

class ActivationConv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=_BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d
        )

    def forward(self, x):
        # Reshape input from [B, K, C] to [B, C, K]
        x = x.transpose(1, 2)
        # Apply convolution and other operations defined in __init__
        x = super().forward(x)
        # Reshape output back to [B, K, C]
        x = x.transpose(1, 2)
        return x


class BasicConvolutionBlock(pl.LightningModule):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

class ResnetBlockFC(pl.LightningModule):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
    
class ResnetBlockFC2(pl.LightningModule):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
    
class ResidualBlock(pl.LightningModule):

    def __init__(self, in_channels, out_channels, dimension, norm_fn=None):
        super().__init__()
        self.downsample = None
        if norm_fn is None:
            norm_fn = ME.MinkowskiBatchNorm

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, dimension=dimension)
            )

        self.conv_branch = nn.Sequential(
            norm_fn(in_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=dimension),
            norm_fn(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=dimension)
        )

    def forward(self, x):
        identity = x
        x = self.conv_branch(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        return x


# class VGGBlock(pl.LightningModule):

#     def __init__(self, in_channels, out_channels, dimension, norm_fn=None):
#         super().__init__()
#         if norm_fn is None:
#             norm_fn = ME.MinkowskiBatchNorm

#         self.conv_layers = nn.Sequential(
#             norm_fn(in_channels),
#             ME.MinkowskiReLU(inplace=True),
#             ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=dimension)
#         )

#     def forward(self, x):
#         return self.conv_layers(x)


class UBlock(pl.LightningModule):

    def __init__(self, n_planes, norm_fn, block_reps, block):

        super().__init__()

        self.nPlanes = n_planes
        self.D = 3

        blocks = {'block{}'.format(i): block(n_planes[0], n_planes[0], self.D, norm_fn) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = nn.Sequential(blocks)

        if len(n_planes) > 1:
            self.conv = nn.Sequential(
                norm_fn(n_planes[0]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(n_planes[0], n_planes[1], kernel_size=2, stride=2, dimension=self.D)
            )

            self.u = UBlock(n_planes[1:], norm_fn, block_reps, block)

            self.deconv = nn.Sequential(
                norm_fn(n_planes[1]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolutionTranspose(n_planes[1], n_planes[0], kernel_size=2, stride=2, dimension=self.D)
            )

            blocks_tail = {'block{}'.format(i): block(n_planes[0] * (2 - i), n_planes[0], self.D, norm_fn) for i in
                           range(block_reps)}
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = nn.Sequential(blocks_tail)

    def forward(self, x):
        out = self.blocks(x)
        identity = out

        if len(self.nPlanes) > 1:
            out = self.conv(out)
            out = self.u(out)
            out = self.deconv(out)
            out = ME.cat(identity, out)
            out = self.blocks_tail(out)
        return out

class MultiScaleUBlock(pl.LightningModule):
    """ U-Net block with multiple output from different layers"""

    def __init__(self, n_planes, norm_fn, block_reps, block, cfg):
        super().__init__()

        self.nPlanes = n_planes
        self.D = 3

        blocks = {'block{}'.format(i): block(n_planes[0], n_planes[0], self.D, norm_fn) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = nn.Sequential(blocks)

        if len(n_planes) > 1:
            self.conv = nn.Sequential(
                norm_fn(n_planes[0]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(n_planes[0], n_planes[1], kernel_size=2, stride=2, dimension=self.D)
            )

            self.u = MultiScaleUBlock(n_planes[1:], norm_fn, block_reps, block, cfg)

            self.deconv = nn.Sequential(
                norm_fn(n_planes[1]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolutionTranspose(n_planes[1], n_planes[0], kernel_size=2, stride=2, dimension=self.D)
            )

            blocks_tail = {'block{}'.format(i): block(n_planes[0] * (2 - i), n_planes[0], self.D, norm_fn) for i in
                           range(block_reps)}
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = nn.Sequential(blocks_tail)

    def forward(self, x, data_dict):
        layer_outputs = []  # Store outputs from each significant step
        out = self.blocks(x)
        identity = out

        if len(self.nPlanes) > 1:
            out = self.conv(out)
            out, sub_layer_outputs = self.u(out, data_dict)  # Recursive call to the next depth
            layer_outputs.extend(sub_layer_outputs)  # Collect all sub layer outputs

            out = self.deconv(out)
            out = ME.cat(identity, out)  # Concatenate with the first captured output

            out = self.blocks_tail(out)
            layer_outputs.append(out)  # Capture output after the tail blocks

        return out, layer_outputs  # Return the final output and all collected outputs

# class MultiScaleUBlock(pl.LightningModule):
#     """ U-Net block with multiple output from different layers"""

#     def __init__(self, n_planes, norm_fn, block_reps, block, cfg):
#         super().__init__()

#         self.nPlanes = n_planes
#         self.D = 3

#         blocks = {'block{}'.format(i): block(n_planes[0], n_planes[0], self.D, norm_fn) for i in range(block_reps)}
#         blocks = OrderedDict(blocks)
#         self.blocks = nn.Sequential(blocks)

#         if len(n_planes) > 1:
#             self.conv = nn.Sequential(
#                 norm_fn(n_planes[0]),
#                 ME.MinkowskiReLU(inplace=True),
#                 ME.MinkowskiConvolution(n_planes[0], n_planes[1], kernel_size=2, stride=2, dimension=self.D)
#             )

#             self.u = MultiScaleUBlock(n_planes[1:], norm_fn, block_reps, block, cfg)

#             self.deconv = nn.Sequential(
#                 norm_fn(n_planes[1]),
#                 ME.MinkowskiReLU(inplace=True),
#                 ME.MinkowskiConvolutionTranspose(n_planes[1], n_planes[0], kernel_size=2, stride=2, dimension=self.D)
#             )

#             blocks_tail = {'block{}'.format(i): block(n_planes[0] * (2 - i), n_planes[0], self.D, norm_fn) for i in
#                            range(block_reps)}
#             blocks_tail = OrderedDict(blocks_tail)
#             self.blocks_tail = nn.Sequential(blocks_tail)

#     def forward(self, x, data_dict):
#         layer_outputs = []  # Store outputs from each significant step
#         out = self.blocks(x)
#         identity = out

#         if len(self.nPlanes) > 1:
#             out = self.conv(out)
#             out, sub_layer_outputs = self.u(out, data_dict)  # Recursive call to the next depth
#             layer_outputs.extend(sub_layer_outputs)  # Collect all sub layer outputs

#             out = self.deconv(out)
#             out = ME.cat(identity, out)  # Concatenate with the first captured output

#             out = self.blocks_tail(out)
#             layer_outputs.append(out)  # Capture output after the tail blocks

#         return out, layer_outputs  # Return the final output and all collected outputs
    
class MultiScaleEncoderUBlock(pl.LightningModule):
    """ U-Net block with multiple output from different layers"""

    def __init__(self, n_planes, norm_fn, block_reps, block, cfg):
        super().__init__()

        self.nPlanes = n_planes
        self.n_layers = cfg.model.network.encoder.unet_blocks_num
        self.add_n_scale_pn = cfg.model.network.encoder.add_n_scale_pn
        self.only_large_pn = cfg.model.network.encoder.only_large_pn
        self.voxel_size = cfg.data.voxel_size
        self.D = 3

        blocks = {'block{}'.format(i): block(n_planes[0], n_planes[0], self.D, norm_fn) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = nn.Sequential(blocks)


        if len(n_planes) > 1:
            self.conv = nn.Sequential(
                norm_fn(n_planes[0]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(n_planes[0], n_planes[1], kernel_size=2, stride=2, dimension=self.D)
            )
            if len(n_planes) == (self.n_layers - self.add_n_scale_pn + 1):
                self.local_pointnet = LocalPointNet(6, n_planes[1], 2*n_planes[1], norm_fn, ResidualBlock, scatter_type='mean', n_blocks=3)

            self.u = MultiScaleEncoderUBlock(n_planes[1:], norm_fn, block_reps, block, cfg)

            self.deconv = nn.Sequential(
                norm_fn(n_planes[1]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolutionTranspose(n_planes[1], n_planes[0], kernel_size=2, stride=2, dimension=self.D)
            )

            blocks_tail = {'block{}'.format(i): block(n_planes[0] * (2 - i), n_planes[0], self.D, norm_fn) for i in
                           range(block_reps)}
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = nn.Sequential(blocks_tail)

    def encoder_queries(self, sp, query_xyz):
        all_indices = []
        offset = 0
        batch_size = int((sp.C)[:, 0].max().item() + 1)  # Assuming query_xyz includes batch index in column 0
        points_per_batch = query_xyz.shape[0] // batch_size  # Points per batch assuming equal distribution
        voxel_centers = sp.C[:, 1:4] * self.voxel_size + self.voxel_size * (2**self.add_n_scale_pn) / 2.0
        for b in range(batch_size):
            batch_mask = sp.C[:, 0] == b
            batch_voxel_centers = voxel_centers[batch_mask]

            batch_start_idx = b * points_per_batch
            batch_end_idx = (b + 1) * points_per_batch
            batch_query_xyz = query_xyz[batch_start_idx:batch_end_idx]

            knn_output = knn_points(batch_query_xyz.unsqueeze(0).to(torch.device("cuda")),
                                    batch_voxel_centers.unsqueeze(0).to(torch.device("cuda")),
                                    K=1)
            indices = knn_output.idx.squeeze(0).squeeze(-1)
            if offset > 0:
                indices += offset
            offset += batch_voxel_centers.shape[0]

            all_indices.append(indices)
        indices = torch.cat(all_indices, dim=0)

        return voxel_centers, indices

    def forward(self, x, data_dict):
        layer_outputs = []  # Store outputs from each significant step
        out = self.blocks(x)
        identity = out

        if len(self.nPlanes) > 1:
            out = self.conv(out)
            if len(self.nPlanes) == (self.n_layers - self.add_n_scale_pn + 1):
                voxel_centers, indices = self.encoder_queries(out, data_dict['xyz'])
                relative_coords = data_dict['xyz'] - voxel_centers[indices]
                pn_out = self.local_pointnet(torch.cat((relative_coords, data_dict['point_features']), dim=1), indices)
                max_indices = indices.max().item()
                pad_size = (0, 0, 0, out.shape[0] - pn_out.shape[0])  # (padding_left, padding_right, padding_top, padding_bottom)
                pn_out = torch.nn.functional.pad(pn_out, pad_size, "constant", 0)
                a = out
                out = out + pn_out
                if self.only_large_pn: # Only use the large scale pointnet features
                    out = out - a


            out, sub_layer_outputs = self.u(out, data_dict)  # Recursive call to the next depth
            layer_outputs.extend(sub_layer_outputs)  # Collect all sub layer outputs

            out = self.deconv(out)
            out = ME.cat(identity, out)  # Concatenate with the first captured output

            out = self.blocks_tail(out)
            layer_outputs.append(out)  # Capture output after the tail blocks

        return out, layer_outputs  # Return the final output and all collected outputs


class LocalPointNet(pl.LightningModule):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        latent_dim (int): dimension of latent code c
        c_in (int): input point features dimension(3 + colors_dim)
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        n_blocks (int): number of blocks ResNetBlockFC layers
        block: block type
        norm_fn: 
    '''

    def __init__(self, c_in, latent_dim, hidden_dim, norm_fn, block, scatter_type='max', n_blocks=5):

        super().__init__()

        self.latent_dim = latent_dim
        self.fc_pos = nn.Linear(c_in, 2*hidden_dim)
        self.D = 3

        # blocks = {'block{}'.format(i): block(2*hidden_dim, hidden_dim, self.D, norm_fn) for i in range(n_blocks)}
        # blocks = OrderedDict(blocks)
        # self.blocks = nn.Sequential(blocks)        
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, latent_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        elif scatter_type == 'add':
            self.scatter = scatter_add
        else:
            raise ValueError('incorrect scatter type')
        

    def pool_local(self, point_features, indices):
        ''' Pooling local features within the voxel '''

        # Find the total number of voxels, K
        K = indices.max().item() + 1

        # Scatter the point features into the voxel features
        scattered_feat = self.scatter(point_features, indices, dim=0, dim_size=K)
        
        # Gather the voxel features back to the points
        gathered_feat = scattered_feat.index_select(0, indices)

        return gathered_feat    
    
    def forward(self, features_in, indices):
        net = self.fc_pos(features_in)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(net, indices)
            net = torch.cat([net, pooled], dim=1)
            net = block(net)

        c = self.fc_c(net)
        K = indices.max().item() + 1
        voxel_feat = self.scatter(c, indices, dim=0, dim_size=K)

        return voxel_feat
    

class LocalPointNet2(pl.LightningModule):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        latent_dim (int): dimension of latent code c
        c_in (int): input point features dimension(3 + colors_dim)
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        n_blocks (int): number of blocks ResNetBlockFC layers
        block: block type
        norm_fn: 
    '''

    def __init__(self, c_in, latent_dim, hidden_dim, norm_fn, block, scatter_type='max', n_blocks=5):

        super().__init__()

        self.latent_dim = latent_dim
        self.fc_pos = nn.Linear(c_in, 2*hidden_dim)
        self.D = 3

        # blocks = {'block{}'.format(i): block(2*hidden_dim, hidden_dim, self.D, norm_fn) for i in range(n_blocks)}
        # blocks = OrderedDict(blocks)
        # self.blocks = nn.Sequential(blocks)        
        self.blocks = nn.ModuleList([
            ResnetBlockFC2(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, latent_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        elif scatter_type == 'add':
            self.scatter = scatter_add
        else:
            raise ValueError('incorrect scatter type')
        

    def pool_local(self, point_features, indices):
        ''' Pooling local features within the voxel '''

        # Find the total number of voxels, K
        K = indices.max().item() + 1

        # Scatter the point features into the voxel features
        scattered_feat = self.scatter(point_features, indices, dim=0, dim_size=K)
        
        # Gather the voxel features back to the points
        gathered_feat = scattered_feat.index_select(0, indices)

        return gathered_feat    
    
    def forward(self, features_in, indices):
        net = self.fc_pos(features_in)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(net, indices)
            net = torch.cat([net, pooled], dim=1)
            net = block(net)

        c = self.fc_c(net)
        K = indices.max().item() + 1
        voxel_feat = self.scatter(c, indices, dim=0, dim_size=K)

        return voxel_feat

class DownsampleUBlock(pl.LightningModule):

    def __init__(self, n_planes, norm_fn, block_reps, block, downsample_steps):

        super().__init__()

        self.nPlanes = n_planes
        self.D = 3

        blocks = {'block{}'.format(i): block(n_planes[0], n_planes[0], self.D, norm_fn) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = nn.Sequential(blocks)

        if len(n_planes) > 1:
            self.conv = nn.Sequential(
                norm_fn(n_planes[0]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(n_planes[0], n_planes[1], kernel_size=2, stride=2, dimension=self.D)
            )

            self.u = UBlock(n_planes[1:], norm_fn, block_reps, block)

            self.deconv = nn.Sequential(
                norm_fn(n_planes[1]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolutionTranspose(n_planes[1], n_planes[0], kernel_size=2, stride=2, dimension=self.D)
            )

            blocks_tail = {'block{}'.format(i): block(n_planes[0] * (2 - i), n_planes[0], self.D, norm_fn) for i in
                           range(block_reps)}
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = nn.Sequential(blocks_tail)

    def forward(self, x):
        out = self.blocks(x)
        identity = out

        if len(self.nPlanes) > 1:
            out = self.conv(out)
            out = self.u(out)
            out = self.deconv(out)
            out = ME.cat(identity, out)
            out = self.blocks_tail(out)
        return out

class SparseConvEncoder(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConvolutionBlock(input_dim, 32, 3)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 64, kernel_size=2, stride=2),
            ResidualBlock(64, 64, 3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(128, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(128, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(
            planes * self.expansion, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNetBase(pl.LightningModule):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )

        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x)


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)


class ResFieldNetBase(ResNetBase):
    def network_initialization(self, in_channels, out_channels, D):
        field_ch = 32
        field_ch2 = 64
        self.field_network = nn.Sequential(
            ME.MinkowskiSinusoidal(in_channels, field_ch),
            ME.MinkowskiBatchNorm(field_ch),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(field_ch, field_ch),
            ME.MinkowskiBatchNorm(field_ch),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiToSparseTensor(),
        )
        self.field_network2 = nn.Sequential(
            ME.MinkowskiSinusoidal(field_ch + in_channels, field_ch2),
            ME.MinkowskiBatchNorm(field_ch2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(field_ch2, field_ch2),
            ME.MinkowskiBatchNorm(field_ch2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiToSparseTensor(),
        )

        ResNetBase.network_initialization(self, field_ch2, out_channels, D)

    def forward(self, x: ME.TensorField):
        otensor = self.field_network(x)
        otensor2 = self.field_network2(otensor.cat_slice(x))
        return ResNetBase.forward(self, otensor2)


class ResFieldNet14(ResFieldNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResFieldNet18(ResFieldNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResFieldNet34(ResFieldNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResFieldNet50(ResFieldNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResFieldNet101(ResFieldNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)


