import torch
import MinkowskiEngine as ME

# Create example coordinates and features
coords = torch.tensor([[0, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 2, 0],
                       [0, 0, 3, 0],
                       [0, 0, 4, 0],
                       [0, 1, 0, 0],
                       [0, 1, 1, 0],
                       [0, 1, 2, 0],
                       [0, 1, 3, 0],
                       [0, 1, 4, 0],
                       [0, 2, 0, 0],
                       [0, 2, 1, 0],
                       [0, 2, 2, 0],
                       [0, 2, 3, 0],
                       [0, 2, 4, 0]], dtype=torch.int32)

features = torch.rand(len(coords), 3)

# Create SparseTensor
input = ME.SparseTensor(features=features, coordinates=coords)

# Define sparse convolution with stride 2
conv = ME.MinkowskiConvolution(
    in_channels=3,
    out_channels=6,
    kernel_size=2,
    stride=2,
    dimension=3
)

# Apply convolution
output = conv(input)

# Print shapes and details
print("Input coordinates shape:", input.C.shape)
print("Input features shape:", input.F.shape)
print("Output coordinates shape:", output.C.shape)
print("Output features shape:", output.F.shape)
print("Output coordinates:", output.C)
