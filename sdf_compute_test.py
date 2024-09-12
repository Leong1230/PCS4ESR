import torch
import ext  # Assuming ext is the module where sdfgen is defined

# Define the input shapes
N = 10  # Number of query points
M = 20  # Number of reference points

# Create random CUDA tensors for query_pos, ref_xyz, and ref_normal
query_pos = torch.rand(N, 3, device='cuda')
ref_xyz = torch.rand(M, 3, device='cuda')
ref_normal = torch.rand(M, 3, device='cuda')

# Call the sdf_from_points function
mc_query_sdf = -ext.sdfgen.sdf_from_points(query_pos, ref_xyz, ref_normal, 8, 0.02, False)[0]
print(mc_query_sdf)