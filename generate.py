import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models.generation import Generator
import torch
import configs.config_loader as cfg_loader
import os
import trimesh
import numpy as np
from tqdm import tqdm
from arrgh import arrgh
# import open3d as o3d

cfg = cfg_loader.get_config()

device = torch.device("cuda")
net = model.NDF()

dataset = voxelized_data.VoxelizedDataset('test',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=1,
                                          num_sample_points=cfg.num_sample_points_generation,
                                          num_workers=30,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)

gen = Generator(net, cfg.exp_name, device=device)

out_path = cfg.data_dir


def gen_iterator(out_path, dataset, gen_p):
    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)

    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)

    for i, data in tqdm(enumerate(loader)):

        path = os.path.normpath(data['path'][0])
        export_path = out_path + '/{}_df.npz'.format(path.split(os.sep)[-1])

        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
            continue
        """ used for generated dense pointcloud """
        # else:
        #     os.makedirs(export_path)

        distance_field = gen.generate_df(data)
        np.savez(export_path, distance_field=distance_field)
        
        """ used to generate dense pointcloud iteratively """
        # for num_steps in [7]:
        #     point_cloud, duration = gen.generate_point_cloud(data, num_steps)
        #     arrgh(point_cloud, duration, num_steps)
        #     np.savez(export_path + 'dense_point_cloud_{}'.format(num_steps), point_cloud=point_cloud, duration=duration)
        #     print('num_steps', num_steps, 'duration', duration)

        # # create an Open3D point cloud object
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # # estimate normals
        # pcd.estimate_normals()

        # # run Poisson surface reconstruction
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

        # # remove low density vertices
        # vertices_to_remove = densities < np.quantile(densities, 0.01)
        # mesh.remove_vertices_by_mask(vertices_to_remove)

        # # export the mesh as an .obj file
        # mesh.export(export_path + 'dense_point_cloud_{}.obj'.format(num_steps))


gen_iterator(out_path, dataset, gen)
