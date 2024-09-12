import os
import hydra
import torch
from tqdm import tqdm
import numpy as np
import open3d as o3d
import json
import time
import trimesh
from collections import defaultdict
from torch.utils.data import Dataset
from pathlib import Path
from importlib import import_module
import pytorch_lightning as pl
from hybridpc.data.data_module import DataModule
from hybridpc.evaluation import UnitMeshEvaluator, MeshEvaluator
from hybridpc.model.module import Dense_Generator, Interpolated_Dense_Generator, MultiScale_Interpolated_Dense_Generator, visualize_tool
from nksr.svh import SparseFeatureHierarchy
import MinkowskiEngine as ME


# def udf_evaluation(out_path, dataset, gen_p):
#     global gen
#     gen = gen_p

#     if not os.path.exists(out_path):
#         os.makedirs(out_path)
#     print(out_path)

#     # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
#     loader = dataset.get_loader(shuffle=True)

#     for i, data in tqdm(enumerate(loader)):
#         path = os.path.normpath(data['path'][0])
#         export_path = out_path + '/{}_df.npz'.format(path.split(os.sep)[-1])

#         if os.path.exists(export_path):
#             print('Path exists - skip! {}'.format(export_path))
#             continue
#         """ used for generated dense pointcloud """
#         # else:
#         #     os.makedirs(export_path)

#         distance_field = gen.generate_df(data)
#         np.savez(export_path, distance_field=distance_field)
        
#         """ used to generate dense pointcloud iteratively """
#         for num_steps in [7]:
#             point_cloud, duration = gen.generate_point_cloud(data, num_steps)
#             arrgh(point_cloud, duration, num_steps)
#             np.savez(export_path + 'dense_point_cloud_{}'.format(num_steps), point_cloud=point_cloud, duration=duration)
#             print('num_steps', num_steps, 'duration', duration)

#         # create an Open3D point cloud object
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(point_cloud)

#         # estimate normals
#         pcd.estimate_normals()

#         # run Poisson surface reconstruction
#         mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

#         # remove low density vertices
#         vertices_to_remove = densities < np.quantile(densities, 0.01)
#         mesh.remove_vertices_by_mask(vertices_to_remove)

#         # export the mesh as an .obj file
#         mesh.export(export_path + 'dense_point_cloud_{}.obj'.format(num_steps))


# class ScanNetDataset(Dataset):
#     def __init__(self, split, partial_input=False, **kwargs):
#         self.over_fitting = kwargs.get("over_fitting", False)
#         self.num_input_points = kwargs.get("num_input_points", 5000)
#         self.std_dev = kwargs.get("std_dev", 0.00)

#         self.split = 'train' if self.over_fitting else split # use only train set for overfitting
#         self.base_path = Path(kwargs.get("base_path", None))

#         if self.split == "test":
#             with (self.base_path / "metadata" / "scannetv2_test.txt").open() as f:
#                 self.scenes = [t.strip() for t in f.readlines()]
#         elif self.split == "train":
#             with (self.base_path / "metadata" / "scannetv2_train.txt").open() as f:
#                 self.scenes = [t.strip() for t in f.readlines()]
#         else:
#             with (self.base_path / "metadata" / "scannetv2_val.txt").open() as f:
#                 self.scenes = [t.strip() for t in f.readlines()]
        
#         # self.scenes = self.scenes[:4]
#         if self.over_fitting:
#             self.split = 'val'
#             self.scenes = ['scene0221_00']
        
#     def __len__(self):
#         return len(self.scenes)

#     def _get_item(self, data_id, rng):
#         scene_name = self.scenes[data_id]

#         data = {}
#         scene_path = os.path.join(self.base_path, self.split, f"{scene_name}.pth")
#         full_data = torch.load(scene_path)
#         full_points = full_data['xyz'].astype(np.float32)
#         full_normals = full_data['normal'].astype(np.float32)

#         if self.num_input_points != -1:
#             sample_indices = np.random.choice(full_points.shape[0], self.num_input_points, replace=True)
#             partial_points = full_points[sample_indices]
#             partial_normals = full_normals[sample_indices]

#         else:
#             partial_points = full_points
#             partial_normals = full_normals

#         if isinstance(self.std_dev, (float, int)):
#             std_dev = [self.std_dev] * 3  # Same standard deviation for x, y, z
#         noise = np.random.normal(0, self.std_dev, partial_points.shape)
#         partial_points += noise

#         data = {
#             "xyz": partial_points,
#             "partial_normal": partial_normals,
#             "all_xyz": full_points,
#             "full_normal": full_normals
#         }

#         return data
def convert_non_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int32):
        return int(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def normalize_pointcloud_to_unit_cube(pointcloud):
    # Calculate the bounds
    min_bound = np.min(pointcloud, axis=0)
    max_bound = np.max(pointcloud, axis=0)

    # Calculate the scale and translate factors
    scale = 1.0 / (max_bound - min_bound).max()
    translate = -0.5 * (max_bound + min_bound)

    # Normalize the pointcloud to the unit cube centered at the origin
    pointcloud_normalized = (pointcloud + translate) * scale
    return pointcloud_normalized

def compute_scale_factor(source_mesh, target_mesh):
    source_bound = source_mesh.get_axis_aligned_bounding_box()
    target_bound = target_mesh.get_axis_aligned_bounding_box()

    source_extent = source_bound.get_extent()
    target_extent = target_bound.get_extent()

    scale_factor = target_extent / source_extent
    return scale_factor

# Function to compute centroid
def compute_centroid(points):
    return np.mean(points, axis=0)

# Function to translate points so that centroid is at the origin
def translate_to_origin(points):
    centroid = compute_centroid(points)
    return points - centroid

class ScanNetDataset(Dataset):
    def __init__(self, split, partial_input=False, **kwargs):
        self.over_fitting = kwargs.get("over_fitting", False)
        self.num_input_points = kwargs.get("num_input_points", 5000)
        self.std_dev = kwargs.get("std_dev", 0.00)

        self.split = 'train' if self.over_fitting else split # use only train set for overfitting
        self.base_path = Path(kwargs.get("base_path", None))

        if self.split == "test":
            with (self.base_path / "metadata" / "scannetv2_test.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        elif self.split == "train":
            with (self.base_path / "metadata" / "scannetv2_train.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        else:
            with (self.base_path / "metadata" / "scannetv2_val.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        
        # self.scenes = self.scenes[:4]
        if self.over_fitting:
            self.split = 'val'
            self.scenes = ['scene0221_00']
        
    def __len__(self):
        return len(self.scenes)

    def _get_item(self, data_id, rng):
        scene_name = self.scenes[data_id]

        data = {}
        scene_path = os.path.join(self.base_path, self.split, f"{scene_name}.pth")
        full_data = torch.load(scene_path)
        full_points = full_data['xyz'].astype(np.float32)
        full_normals = full_data['normal'].astype(np.float32)

        if self.num_input_points != -1:
            sample_indices = np.random.choice(full_points.shape[0], self.num_input_points, replace=True)
            partial_points = full_points[sample_indices]
            partial_normals = full_normals[sample_indices]

        else:
            partial_points = full_points
            partial_normals = full_normals

        if isinstance(self.std_dev, (float, int)):
            std_dev = [self.std_dev] * 3  # Same standard deviation for x, y, z
        noise = np.random.normal(0, self.std_dev, partial_points.shape)
        partial_points += noise

        data = {
            "xyz": partial_points,
            "normal": partial_normals,
            "all_xyz": full_points,
            "all_normal": full_normals
        }

        return data
    
def pt_input_splat(data_dict, cfg):
    batch_size = len(data_dict['voxel_nums'])
    points_per_batch = cfg.data.num_input_points
    batch_voxel_coords = []
    batch_indices = []
    batch_splat_point_features = []
    for b in range(batch_size):
        batch_start_idx = b * points_per_batch
        batch_end_idx = (b + 1) * points_per_batch
        xyz = data_dict['xyz'][batch_start_idx:batch_end_idx]
        point_features = data_dict['point_features'][batch_start_idx:batch_end_idx]
        svh = SparseFeatureHierarchy(
            voxel_size=cfg.data.voxel_size,
            depth=4,
            device=xyz.device
        )
        svh.build_point_splatting(xyz)
        grid = svh.grids[0]
        # Get voxel idx
        xyz_grid = grid.world_to_grid(xyz)
        indices = grid.ijk_to_index(xyz_grid.round().int())
        voxel_coords =  grid.active_grid_coords()
        voxel_coords = voxel_coords[:torch.max(indices).item() + 1]
        batch_voxel_coords.append(voxel_coords)
        splat_point_features = torch.zeros(len(voxel_coords), point_features.shape[1], device=point_features.device)
        splat_point_features[indices] = point_features
        batch_splat_point_features.append(splat_point_features)


    batch_voxel_coords = ME.utils.batched_coordinates(batch_voxel_coords, dtype=data_dict['xyz'].dtype, device=data_dict['xyz'].device)
    batch_voxel_centers = batch_voxel_coords[:, 1:4] * cfg.data.voxel_size + cfg.data.voxel_size / 2.0
    batch_ids = batch_voxel_coords[:, 0].int()
    batch_splat_point_features = torch.cat(batch_splat_point_features, dim=0)

    return batch_voxel_centers, batch_splat_point_features, batch_ids
    
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    # device = torch.device("cpu")
    Algorigthm = "DMC" # DMC or DensePointcloud or MC or Objective
    # fix the seed
    pl.seed_everything(cfg.global_test_seed, workers=True)

    print("=> initializing trainer...")
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, logger=False)

    output_path = os.path.join(cfg.exp_output_root_path, "inference", cfg.model.inference.split, "udf_visualizations")
    os.makedirs(output_path, exist_ok=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)
    data_module.setup("test")
    val_loader = data_module.val_dataloader()
    
    print("=> initializing model...")
    model = getattr(import_module("hybridpc.model"), cfg.model.network.module)(cfg)
    # Load checkpoint
    if os.path.isfile(cfg.model.ckpt_path):
        print(f"=> loading model checkpoint '{cfg.model.ckpt_path}'")
        checkpoint = torch.load(cfg.model.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint successfully.")
    else:
        print(f"=> no checkpoint found at '{cfg.model.ckpt_path}'")
        # If there's specific behavior when no checkpoint is found, handle it here.

    model.to(device)
    model.eval()

    # if "MultiScaleInterpolated" in cfg.model.network.udf_decoder.decoder_type:
    dense_generator = MultiScale_Interpolated_Dense_Generator(
        model.udf_decoder,
        model.mask_decoder,
        cfg.model.network.udf_decoder.decoder_type,
        cfg.data.voxel_size,
        cfg.model.dense_generator.num_steps,
        cfg.model.dense_generator.num_points,
        cfg.model.dense_generator.threshold,
        cfg.model.dense_generator.filter_val,
        cfg.model.network.udf_decoder.neighbor_type,
        cfg.model.network.udf_decoder.k_neighbors,
        cfg.model.network.udf_decoder.last_n_layers,
        cfg.data.reconstruction
    )
    # elif "Interpolated" in cfg.model.network.udf_decoder.decoder_type:
    #     dense_generator = Interpolated_Dense_Generator(
    #         model.udf_decoder,
    #         cfg.model.network.udf_decoder.decoder_type,
    #         cfg.data.voxel_size,
    #         cfg.model.dense_generator.num_steps,
    #         cfg.model.dense_generator.num_points,
    #         cfg.model.dense_generator.threshold,
    #         cfg.model.dense_generator.filter_val,
    #         cfg.model.network.udf_decoder.neighbor_type,h
    #         cfg.model.network.udf_decoder.k_neighbors
    #     )
    # else:
    #     dense_generator = Dense_Generator(
    #         model.udf_decoder,
    #         cfg.data.voxel_size,
    #         cfg.model.dense_generator.num_steps,
    #         cfg.model.dense_generator.num_points,
    #         cfg.model.dense_generator.threshold,
    #         cfg.model.dense_generator.filter_val,
    #         cfg.model.dense_generator.type
    #     )

    # Initialize a dictionary to keep track of sums and count
    eval_sums = defaultdict(float)
    batch_count = 0
    # total_voxel_num = 0
    # dataset = ScanNetDataset(split='val', partial_input=True, base_path='/localhome/zla247/theia1_data/scannetv2', over_fitting=False, num_input_points=10000, std_dev=0.00)
    # total_scenes = len(dataset)

    print("=> start inference...")
    # with torch.inference_mode():
    start_time = time.time()
    total_reconstruction_duration = 0.0
    total_knn_time = 0.0
    total_dmc_time = 0.0
    total_attentive_time = 0.0 
    total_interpolation_time = 0.0
    total_forward_duration = 0.0
    total_after_layers_time = 0.0
    total_decoder_time = 0.0
    total_grid_splat_time = 0.0
    total_sdf_error = 0.0
    total_normal_error = 0.0
    results_dict = []
    for batch in tqdm(val_loader, desc="Inference", unit="batch"):
    # for data_id in tqdm(range(total_scenes), desc="Processing scenes"):
        # Move the batch to the GPU
        if batch['scene_names'] == ['scene0207_00']:
            test = 1
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        if 'gt_geometry' in batch:
            gt_geometry = batch['gt_geometry']
            batch['all_xyz'], batch['all_normal'], _ = gt_geometry[0].torch_attr()
        # total_voxel_num += batch['voxel_nums'][0]
        gt_pointcloud = batch['all_xyz'].cpu().numpy()
        input_pointcloud = batch['xyz'].cpu().numpy()
        process_start = time.time()
        forward_start = time.time()
        if cfg.model.network.backbone == 'PointTransformerV3':
            pt_data = {}
            if cfg.model.network.encoder.input_splat: 
                pt_data['coord'], pt_data['feat'], pt_data['batch'] = pt_input_splat(batch, cfg)
                pt_data['grid_size'] = 0.01
            else:
                pt_data['feat'] = batch['point_features']
                # batch_ids = torch.cat([torch.full((10000,), i) for i, n in enumerate(batch['voxel_nums'])]).to(batch['voxel_features'].device)

                pt_data['offset'] = batch['xyz_splits']
                pt_data['grid_size'] = 0.01
                pt_data['coord'] = batch['xyz']
            encoder_outputs = model.point_transformer(pt_data)
        else:
            encoder_outputs = model.encoder(batch)
        forward_end = time.time()
        torch.set_grad_enabled(False)

        if Algorigthm == 'DMC':
            # dense_pointcloud, duration = dense_generator.generate_point_cloud(batch, encoder_outputs, device)
            dmc_mesh, re_time, dmc_time, attentive_time, interpolation_time, after_layers_time, decoder_time, grid_splat_time, dmc_vertices, dmc_values, colors, voxel_centers = dense_generator.generate_dual_mc_mesh(batch, encoder_outputs, device)
            total_knn_time += re_time
            total_dmc_time += dmc_time
            total_attentive_time += attentive_time
            total_interpolation_time += interpolation_time
            total_after_layers_time += after_layers_time
            total_decoder_time += decoder_time
            total_grid_splat_time += grid_splat_time
            # Calculate time taken for these three steps
            process_end = time.time()
            total_forward_duration += forward_end - forward_start
            process_duration = process_end - process_start
            total_reconstruction_duration += process_duration  # Accumulate the duration
            print(f"Time taken for the forward pass: {total_forward_duration:.2f} seconds")
            print(f"Time taken for the reconstruction process: {total_reconstruction_duration:.2f} seconds")
            print(f"Time taken for the knn search: {total_knn_time:.2f} seconds")
            print(f"Time taken for the DMC: {total_dmc_time:.2f} seconds")
            print(f"Time taken for the attentive pooling: {total_attentive_time:.2f} seconds")
            print(f"Time taken for the interpolation: {total_interpolation_time:.2f} seconds")
            print(f"Time taken for the after layers: {total_after_layers_time:.2f} seconds")
            print(f"Time taken for the decoder: {total_decoder_time:.2f} seconds")
            print(f"Time taken for the grid splat: {total_grid_splat_time:.2f} seconds")
            # Evaluate the reconstructed mesh
            evaluator = UnitMeshEvaluator(n_points=100000, metric_names=UnitMeshEvaluator.ESSENTIAL_METRICS)
            # evaluator = MeshEvaluator(n_points=int(5e6), metric_names=MeshEvaluator.ESSENTIAL_METRICS)
            eval_dict, translation, scale = evaluator.eval_mesh(dmc_mesh, batch['all_xyz'], None, onet_samples=None)
            # pc_eval_dict = eval_pointcloud(dense_pointcloud, gt_pointcloud, input_pointcloud)
        elif Algorigthm == 'MC':
            mesh, voxel_centers = dense_generator.generate_mesh(batch, encoder_outputs, device)
            dense_pointcloud, duration = dense_generator.generate_point_cloud(batch, encoder_outputs, device)
            # visualize_tool(dense_pointcloud, gt_pointcloud, input_pointcloud, batch['scene_names'])
            eval_dict = eval_pointcloud(dense_pointcloud, gt_pointcloud, input_pointcloud)
        elif Algorigthm == 'Objective':
            sdf_error, normal_error = dense_generator.compute_objective_function(cfg, batch, encoder_outputs, device)
            total_sdf_error += sdf_error
            total_normal_error += normal_error
            continue
        else:
            dense_pointcloud, duration = dense_generator.generate_point_cloud(batch, encoder_outputs, device)
            eval_dict = eval_pointcloud(dense_pointcloud, gt_pointcloud, input_pointcloud)

        # eval_dict['voxel_num'] = batch['voxel_nums'][0]
        for k, v in eval_dict.items():
            eval_sums[k] += v
        scene_name = batch['scene_names']
        eval_dict["data_id"] = batch_count
        eval_dict["scene_name"] = scene_name
        results_dict.append(eval_dict)
        print(f"Scene Name: {scene_name}")
        print(f"completeness: {eval_dict['completeness']:.4f}")
        print(f"accuracy: {eval_dict['accuracy']:.4f}")
        print(f"Chamfer-L2: {eval_dict['chamfer-L2']:.4f}")
        print(f"Chamfer-L1: {eval_dict['chamfer-L1']:.4f}")
        # print(f"F-Score (0.5%): {eval_dict['f-score-05']:.4f}")
        print(f"F-Score (1.0%): {eval_dict['f-score-10']:.4f}")
        print(f"F-Score (1.5%): {eval_dict['f-score-15']:.4f}")
        print(f"F-Score (2.0%): {eval_dict['f-score-20']:.4f}")
        # print(f"voxel_num: {eval_dict['voxel_num']}")
        print()  # For better readability

        # Save the mesh
        if cfg.data.reconstruction.visualization.visualize:
            chamfer_L1 = eval_dict['chamfer-L1']
            if Algorigthm == 'DMC':
                if cfg.data.reconstruction.visualization.Mesh:
                    mesh_file = f"{cfg.data.dataset_root_path}/Visualizations/DMC_visualizations/GT-{cfg.data.reconstruction.gt_sdf}_{cfg.model.network.backbone}_{cfg.data.dataset}-Scene-{cfg.data.intake_start}_Mask-gt-{cfg.data.reconstruction.gt_mask}-{cfg.data.reconstruction.mask_threshold}_CD-L1_{chamfer_L1:.4f}_mesh.obj"  # or "output_mesh.obj" for OBJ format
                    o3d.io.write_triangle_mesh(mesh_file, dmc_mesh)
                if cfg.data.reconstruction.visualization.Input_points:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(batch['xyz'].cpu().numpy())
                    point_cloud_file = f"{cfg.data.dataset_root_path}/Visualizations/DMC_visualizations/GT-{cfg.data.reconstruction.gt_sdf}_{cfg.model.network.backbone}_{cfg.data.dataset}-Scene-{cfg.data.intake_start}_Mask-gt-{cfg.data.reconstruction.gt_mask}-{cfg.data.reconstruction.mask_threshold}_CD-L1_{chamfer_L1:.4f}_input_pcd.ply"
                    o3d.io.write_point_cloud(point_cloud_file, pcd)
                if cfg.data.reconstruction.visualization.Pooled_points:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
                    yellow = np.array([[1.0, 1.0, 0.0] for _ in range(len(voxel_centers))])
                    pcd.colors = o3d.utility.Vector3dVector(yellow)
                    point_cloud_file = f"{cfg.data.dataset_root_path}/Visualizations/DMC_visualizations/GT-{cfg.data.reconstruction.gt_sdf}_{cfg.model.network.backbone}_{cfg.data.dataset}-Scene-{cfg.data.intake_start}_Mask-gt-{cfg.data.reconstruction.gt_mask}-{cfg.data.reconstruction.mask_threshold}_CD-L1_{chamfer_L1:.4f}_Layer-{cfg.model.network.udf_decoder.scale_visualization}_pcd.ply"
                    o3d.io.write_point_cloud(point_cloud_file, pcd)
                if cfg.data.reconstruction.visualization.Dense_points:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(batch['all_xyz'].cpu().numpy())
                    point_cloud_file = f"{cfg.data.dataset_root_path}/Visualizations/DMC_visualizations/GT-{cfg.data.reconstruction.gt_sdf}_{cfg.model.network.backbone}_{cfg.data.dataset}-Scene-{cfg.data.intake_start}_Mask-gt-{cfg.data.reconstruction.gt_mask}-{cfg.data.reconstruction.mask_threshold}_CD-L1_{chamfer_L1:.4f}_dense_pcd.ply"
                    o3d.io.write_point_cloud(point_cloud_file, pcd)

        # elif Algorigthm == 'MC':
        #     chamfer_L1 = eval_dict['chamfer-L1']
        #     mesh_file = f"../../theia1_data/Visualizations/DMC_visualizations/MC_SDF_{cfg.model.network.udf_decoder.decoder_type}_Voxel-{cfg.data.voxel_size}_{batch['scene_names']}_CD-L1_{chamfer_L1:.4f}_num-{cfg.data.num_input_points}_mesh.obj"  # or "output_mesh.obj" for OBJ format
        #     o3d.io.write_triangle_mesh(mesh_file, mesh)

        # # Save dense pointcloud
        # chamfer_L1 = eval_dict['chamfer-L1']
        # dense_pcd = o3d.geometry.PointCloud()
        # dense_pcd.points = o3d.utility.Vector3dVector(dense_pointcloud)
        # # sign_pcd = sign_pcd.reshape(-1)

        # # # Create a color array based on the sign
        # # colors = np.zeros((len(dense_pointcloud), 3))  # N x 3 array for RGB colors
        # # colors[sign_pcd > 0] = [1, 0, 0]  # Red for positive sign
        # # colors[sign_pcd <= 0] = [0, 1, 0]  # Green for negative sign
        # # dense_pcd.colors = o3d.utility.Vector3dVector(colors)
        # point_cloud_file = f"../../theia1_data/Visualizations/DMC_visualizations/Signed-colors_Multi-scale_DensePC_Voxel-{cfg.data.voxel_size}_{batch['scene_names']}_CD-L1_{chamfer_L1:.4f}_dense_pcd.ply"
        # o3d.io.write_point_cloud(point_cloud_file, dense_pcd)     

        # Save the input point cloud and ground truth point cloud

        # gt_pointcloud = normalize_pointcloud_to_unit_cube(gt_pointcloud)
        # pointcloud_centroid = compute_centroid(gt_pointcloud)
        # gt_pointcloud = gt_pointcloud - pointcloud_centroid + mesh_centroid
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(gt_pointcloud)
        # ply_file = f"../../theia1_data/Visualizations/DMC_visualizations/Multi-scale_gt-PC_SDF_{cfg.model.network.udf_decoder.decoder_type}_Voxel-{cfg.data.voxel_size}_{scene_name}_CD-L1_{chamfer_L1:.4f}_num-{cfg.data.num_input_points}_gt_pointcloud.ply"
        # o3d.io.write_point_cloud(ply_file, pcd)

        # # input_pointcloud = normalize_pointcloud_to_unit_cube(input_pointcloud)
        # # pointcloud_centroid = compute_centroid(input_pointcloud)
        # # input_pointcloud = input_pointcloud - pointcloud_centroid + mesh_centroid
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(input_pointcloud)
        # ply_file = f"../data/Visualizations/{cfg.model.network.udf_decoder.decoder_type}_Voxel-{cfg.data.voxel_size}_{scene_name}_CD-L1_{chamfer_L1:.4f}_num-{cfg.data.num_input_points}_moved_input_pointcloud.ply"
        # o3d.io.write_point_cloud(ply_file, pcd)

        # voxel_centers = normalize_pointcloud_to_unit_cube(voxel_centers)
        # # pointcloud_centroid = compute_centroid(voxel_centers)
        # voxel_centers = voxel_centers - pointcloud_centroid + mesh_centroid    
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(voxel_centers)
        # ply_file = f"../data/Visualizations/{cfg.model.network.udf_decoder.decoder_type}_Voxel-{cfg.data.voxel_size}_{scene_name}_CD-L1_{chamfer_L1:.4f}_num-{cfg.data.num_input_points}_moved_voxel_centers.ply"
        # o3d.io.write_point_cloud(ply_file, pcd)

        """ import and align ground truth and baseline """
        # baseline_mesh = o3d.io.read_triangle_mesh("/localhome/zla247/projects/data/Selected_visualizations/Inter_Voxel-0.02_L1_0.0021_num-10K_mesh.obj")
        # nksr_mesh = o3d.io.read_triangle_mesh("/localhome/zla247/projects/data/Visualizations/Nksr_mesh_voxel_0.5_L1-1.22.obj")
        # # Compute centroids
        # baseline_centroid = compute_centroid(np.asarray(baseline_mesh.vertices))
        # nksr_centroid = compute_centroid(np.asarray(nksr_mesh.vertices))
        # scale_factor = compute_scale_factor(nksr_mesh, baseline_mesh)
        # # Scale the nksr_mesh
        # nksr_mesh.scale(scale_factor[0], center=nksr_centroid)
        # # Compute the translation needed after scaling
        # nksr_centroid_after_scaling = compute_centroid(np.asarray(nksr_mesh.vertices))
        # nksr_translation = baseline_centroid - nksr_centroid_after_scaling
        # nksr_mesh.translate(nksr_translation)
        # o3d.io.write_triangle_mesh("../data/Selected_visualizations/Nksr_mesh_voxel_0.5_L1-1.22.obj", nksr_mesh)

        # print(f"Mesh saved")

        batch_count += 1
        torch.cuda.empty_cache()
    # Path to the file where you want to save the results
    if Algorigthm == 'Objective':
        print(f"Total SDF error: {total_sdf_error:.5f}")
        print(f"Total normal error: {total_normal_error:.5f}")
        return
    file_path = 'results.txt'

    # Write the dictionary to the file
    with open(file_path, 'w') as file:
        for item in results_dict:
            file.write(json.dumps(item, default=convert_non_serializable) + '\n')

    print(f'Results saved to {file_path}')
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total reconstruction time for all scenes: {total_time:.2f} seconds")
    if batch_count > 0:
        print("\n--- Evaluation Metrics Averages ---")
        for k in eval_sums:
            average = eval_sums[k] / batch_count
            print(f"{k}: {average:.5f}")
    else:
        print("No batches were processed.")


if __name__ == "__main__":
    main()
