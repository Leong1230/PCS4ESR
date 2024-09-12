import os
import numpy as np
from plyfile import PlyData, PlyElement

def sample_and_save_ply(input_path, output_path, max_points=200000):
    ply_data = PlyData.read(input_path)
    vertex = ply_data['vertex']
    pos = np.stack([vertex[t] for t in ('x', 'y', 'z')], axis=1)
    nls = np.stack([vertex[t] for t in ('nx', 'ny', 'nz')], axis=1) if 'nx' in vertex and 'ny' in vertex and 'nz' in vertex else np.zeros_like(pos)
    
    if len(pos) > max_points:
        indices = np.random.choice(len(pos), max_points, replace=False)
        pos = pos[indices]
        nls = nls[indices]
    
    vertex_data = np.zeros(len(pos), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    vertex_data['x'] = pos[:, 0]
    vertex_data['y'] = pos[:, 1]
    vertex_data['z'] = pos[:, 2]
    vertex_data['nx'] = nls[:, 0]
    vertex_data['ny'] = nls[:, 1]
    vertex_data['nz'] = nls[:, 2]
    
    ply_element = PlyElement.describe(vertex_data, 'vertex')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PlyData([ply_element]).write(output_path)

def process_dataset(root_folder, output_folder, max_points=200000):
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.ply'):
                input_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(input_path, root_folder)
                output_path = os.path.join(output_folder, relative_path.replace('.ply', '_subsampled.ply'))
                sample_and_save_ply(input_path, output_path, max_points)

root_folder = 'scenenn_dec24_data'
output_folder = 'scenenn_sub_data'
process_dataset(root_folder, output_folder)
