import os
from tqdm import tqdm
import statistics
from statistics import mode
import numpy as np
import MinkowskiEngine
from arrgh import arrgh
import random
import math
import h5py
import torch
from torch.utils.data import Dataset
import open3d as o3d
import matplotlib.cm as cm
from plyfile import PlyData
from pycarus.geometry.pcd import compute_udf_from_pcd, knn, compute_sdf_from_pcd
from hybridpc.util.transform import jitter, flip, rotz, elastic, crop



class GeneralDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = 'train' if self.cfg.data.over_fitting else split # use only train set for overfitting
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.num_point = cfg.data.num_point
        self.rotate_num = cfg.data.rotate_num
        self.take = cfg.data.take
        self.intake_start = cfg.data.intake_start
        self.use_relative = cfg.data.use_relative
        self.max_num_point = cfg.data.max_num_point
        self.sample_entire_scene = cfg.data.udf_queries.sample_entire_scene
        self.num_queries_on_surface = cfg.data.udf_queries.num_queries_on_surface
        self.queries_stds = cfg.data.udf_queries.queries_stds
        self.num_queries_per_std = cfg.data.udf_queries.num_queries_per_std
        # Setting ratio parameters for point selection
        self.ratio_on_surface = cfg.data.udf_queries.queries_ratio_on_surface
        self.ratio_per_std = cfg.data.udf_queries.queries_ratio_per_std
        self.in_memory = cfg.data.in_memory
        self.dataset_split = "test" if split == "test" else "train" # train and val scenes and all under train set
        self.data_map = {
            "train": cfg.data.metadata.train_list,
            "val": cfg.data.metadata.val_list,
            "test": cfg.data.metadata.test_list
        }
        self._load_from_disk()
        self.data = []

        for idx, sample in tqdm(enumerate(self.scenes), desc="Voxelizing and Sample points", ncols=80):
            if self.sample_entire_scene:
                processed_sample = self.preprocess_sample_entire_scene(sample, idx)
            else:
                processed_sample = self.preprocess_sample(sample)
            self.data.append(processed_sample)


        # self.label_map = self.get_semantic_mapping_file(cfg.data.metadata.combine_file)
        # self.filelist = os.path.join(self.dataset_root_path, self.data_map[self.split]) # train.txt, val.txt, test.txt
        # self.filenames, self.labels = self.load_filenames()

        # if self.cfg.data.over_fitting:
        #     # self.random_idx = random.randint(0, len(self.filenames) - 1)
        #     self.random_idx = 0
        
        # if self.in_memory:
        #     print('Load ' + self.split + ' dataset into memory')
        #     self.samples = [self.read_file(os.path.join(self.dataset_root_path, self.dataset_split, f))
        #                     for f in tqdm(self.filenames, ncols=80, leave=False)]    
        #     self.data = []
        #     for sample in tqdm(self.samples, desc="Voxelizing and Sample points", ncols=80):
        #         if self.sample_entire_scene:
        #             processed_sample = self.preprocess_sample_entire_scene(sample)
        #         else:
        #             processed_sample = self.preprocess_sample(sample)
        #         self.data.append(processed_sample)

    def _load_from_disk(self):
        with open(getattr(self.cfg.data.metadata, f"{self.split}_list")) as f:
            self.scene_names = [line.strip() for line in f]
        self.scenes = []
        if self.cfg.data.over_fitting:
            self.scene_names = self.scene_names[self.intake_start:self.take+self.intake_start]
        for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data from disk"):
            scene_path = os.path.join(self.cfg.data.dataset_path, self.split, f"{scene_name}.pth")
            scene = torch.load(scene_path)
            scene["xyz"] -= scene["xyz"].mean(axis=0)
            scene["rgb"] = scene["rgb"].astype(np.float32) / 127.5 - 1
            scene['scene_name'] = scene_name
            self.scenes.append(scene) 

    # def preprocess_sample(self, sample):
    #     # Voxelize the points
    #     voxel_coords, unique_map, inverse_map = MinkowskiEngine.utils.sparse_quantize(
    #         sample['points'], return_index=True, return_inverse=True, quantization_size=self.voxel_size)

    #     # Get unique voxel coordinates and count the number of points per voxel
    #     unique_voxel_coords, counts = np.unique(inverse_map, return_counts=True, axis=0)
        
    #     # Compute the number of unique labels per voxel
    #     labels_per_voxel = [np.unique(sample['labels'][inverse_map == i]) for i in range(len(unique_voxel_coords))]
    #     num_labels_per_voxel = [len(labels) for labels in labels_per_voxel]
    #     all_points = []
    #     all_colors = []
    #     all_labels = []
    #     all_query_points = []
    #     all_values = []
    #     all_voxel_indices = []
    #     all_query_voxel_indices = []

    #     # Initialize counter for non-empty voxels
    #     num_non_empty_voxels = 0
    #     for voxel_idx in range(len(unique_voxel_coords)):
    #         mask = (inverse_map == voxel_idx)

    #         points_in_selected_voxel = sample['points'][mask]
    #         num_points_in_voxel = len(points_in_selected_voxel)

    #         if num_points_in_voxel == 0:  # Skip if there are no points in the voxel
    #             continue
    #         # Assume points_in_selected_voxel is your points tensor of shape (N, 3)
    #         # and voxel_size is the size of your voxel

    #         # Shift the points to the range [0, voxel_size]
    #         points_in_selected_voxel -= np.min(points_in_selected_voxel, 0)

    #         # Scale to the range [0, 1]
    #         points_in_selected_voxel /= self.voxel_size

    #         # Shift and scale to the range [-1, 1]
    #         norm_points_in_selected_voxel = 2.0 * points_in_selected_voxel - 1.0

    #         norm_points_in_selected_voxel_tensor = torch.tensor(np.asarray(norm_points_in_selected_voxel))

    #         # Calculate number of queries based on the ratio and the number of points in the voxel
    #         num_queries_on_surface = int(num_points_in_voxel * self.ratio_on_surface + 1)
    #         num_queries_per_std = [int(num_points_in_voxel * self.ratio_per_std + 1)] * 4  # A list of 4 equal values

    #         query_points, values = compute_udf_from_pcd(
    #             norm_points_in_selected_voxel_tensor,
    #             num_queries_on_surface,
    #             self.queries_stds,
    #             num_queries_per_std
    #         )

    #         # Convert tensors to numpy arrays:
    #         query_points = query_points.cpu().numpy()
    #         values = values.cpu().numpy()

    #         # Check for NaN values:
    #         nan_mask_query_points = np.isnan(query_points).any(axis=1)
    #         nan_mask_values = np.isnan(values)

    #         # Check if there are any NaNs in either query_points or values:
    #         nan_mask_combined = nan_mask_query_points | nan_mask_values

    #         # If there are any NaNs, print a warning and remove them:
    #         if np.any(nan_mask_combined):
    #             # print(f"Warning: found NaN in data, removing corresponding rows.")
    #             query_points = query_points[~nan_mask_combined]
    #             values = values[~nan_mask_combined]

    #         if self.use_relative:
    #             all_points.append(norm_points_in_selected_voxel)  # Output points in not normalized within each voxel
    #         else:
    #             all_points.append(points_in_selected_voxel)  # Output points in not normalized within each voxel
    #         all_colors.append(sample['colors'][mask])
    #         all_labels.append(sample['labels'][mask])
    #         all_query_points.append(query_points)  # Output query points in normalized within each voxel
    #         all_values.append(values)
    #         all_voxel_indices.append(np.full((points_in_selected_voxel.shape[0],), voxel_idx))
    #         all_query_voxel_indices.append(np.full((query_points.shape[0],), voxel_idx))

    #     # Concatenate all the data
    #     data = {
    #         "points": np.concatenate(all_points, axis=0), #N, 3
    #         "colors": np.concatenate(all_colors, axis=0), #N, 3
    #         "labels": np.concatenate(all_labels, axis=0), #N, 
    #         "voxel_indices": np.concatenate(all_voxel_indices, axis=0), #N, 
    #         "query_points": np.concatenate(all_query_points, axis=0), # M, 3
    #         "values": np.concatenate(all_values, axis=0), # M,
    #         "query_voxel_indices": np.concatenate(all_query_voxel_indices, axis=0), # M, 3
    #         "voxel_coords": voxel_coords.cpu().numpy() # K, 3
    #         # "num_non_empty_voxels": num_non_empty_voxels # int
    #     }
    #     return data
    
    def preprocess_sample_entire_scene(self, sample, idx):
        # Voxelize the points
        points = sample['xyz']
        point_xyz = points
        sem_labels = sample['sem_labels']

        point_features = np.zeros(shape=(len(points), 0), dtype=np.float32)
        if self.cfg.model.network.use_color:
            point_features = np.concatenate((point_features, sample['rgb']), axis=1)
        if self.cfg.model.network.use_normal:
            point_features = np.concatenate((point_features, sample['normal']), axis=1)

        point_features = np.concatenate((point_features, points), axis=1)  # add xyz to point features
        voxel_coords, voxel_features, unique_map, inverse_map = MinkowskiEngine.utils.sparse_quantize(
            sample['xyz'], point_features, return_index=True, return_inverse=True, quantization_size=self.voxel_size)

        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 # compute voxel_center in orginal coordinate system (torch.tensor)

        # Convert to tensor
        points_tensor = torch.tensor(np.asarray(points))

        # Calculate number of queries based on the ratio and the number of points
        num_queries_on_surface = int(len(points) * self.ratio_on_surface + 1)
        num_queries_per_std = [int(len(points) * self.ratio_per_std + 1)] * 4  # A list of 4 equal values
        min_range = np.min(points, axis=0)
        max_range = np.max(points, axis=0)

        query_points, values = compute_udf_from_pcd(
            points_tensor,
            num_queries_on_surface,
            self.queries_stds,
            num_queries_per_std,
            (torch.tensor(min_range), torch.tensor(max_range))
        ) # torch.tensor

        # create masks to visualize

        # Number of surface queries
        num_surface_queries = num_queries_on_surface

        # Number of Gaussian queries
        num_gaussian_queries = sum(num_queries_per_std) - num_queries_per_std[-1]# Sum of all Gaussian queries

        # Number of uniform queries
        num_uniform_queries = num_queries_per_std[-1]

        # Check if we have any discrepancy in the count
        if num_uniform_queries < 0:
            raise ValueError('The counts of queries do not match the total number of queries in `query_points`.')

        # Create masks
        mask_surface = torch.cat((torch.ones(num_surface_queries), torch.zeros(num_gaussian_queries + num_uniform_queries))).bool().cpu().numpy()
        mask_gaussian = torch.cat((torch.zeros(num_surface_queries), torch.ones(num_gaussian_queries), torch.zeros(num_uniform_queries))).bool().cpu().numpy()
        mask_uniform = torch.cat((torch.zeros(num_surface_queries + num_gaussian_queries), torch.ones(num_uniform_queries))).bool().cpu().numpy()

        # find the nearest voxel center for each query point
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_points = query_points.to(device)
        voxel_center = voxel_center.to(device)

        query_indices, _, _ = knn(query_points, voxel_center, 1)
        # query_indices, _, _ = knn(query_points, voxel_center, self.cfg.data.k_neighbours)
        inverse_map = inverse_map.to(device)
        query_indices = query_indices[:, 0]
        # query_indices = query_indices
        relative_coords = points - voxel_center[inverse_map].cpu().numpy()
        query_relative_coords = query_points.cpu().numpy() - voxel_center[query_indices].cpu().numpy()

        # remove query_points outside the voxel
        lower_bound = -self.voxel_size / 2
        upper_bound = self.voxel_size / 2
        # Create a mask
        mask = (query_relative_coords >= lower_bound) & (query_relative_coords <= upper_bound)
        # Reduce across the last dimension to get a (N, ) mask
        mask = np.all(mask,-1)
        query_indices = query_indices[mask]
        query_relative_coords = query_relative_coords[mask]
        values = values[mask].cpu().numpy()

        # elastic
        scale = (1 / self.cfg.data.voxel_size)
        if self.split == "train" and self.cfg.data.augmentation.elastic:
            point_xyz_elastic = elastic(point_xyz * scale, 6 * scale // 50, 40 * scale / 50)
            point_xyz_elastic = elastic(point_xyz_elastic, 20 * scale // 50, 160 * scale / 50)
        else:
            point_xyz_elastic = point_xyz * scale

        point_xyz_elastic -= point_xyz_elastic.min(axis=0)

        #crop
        if all(label == -1 for label in sample['sem_labels']):
            flag = idx
        # if self.split == "train":
        #     # HACK, in case there are few points left
        #     max_tries = 20
        #     valid_idxs_count = 0
        #     valid_idxs = np.ones(shape=points.shape[0], dtype=bool)
        #     if valid_idxs.shape[0] > self.max_num_point:
        #         while max_tries > 0:
        #             points_tmp, valid_idxs = crop(point_xyz_elastic, self.max_num_point, self.cfg.data.full_scale[1])
        #             valid_idxs_count = np.count_nonzero(valid_idxs)
        #             if valid_idxs_count >= (self.max_num_point // 2) and np.any(sem_labels[valid_idxs] != -1):
        #                 point_xyz_elastic = points_tmp
        #                 break
        #             max_tries -= 1
        #         if valid_idxs_count < (self.max_num_point // 2) or np.all(sem_labels[valid_idxs] == -1):
        #             raise Exception("Over-cropped!")

        #     # point_xyz_elastic = point_xyz_elastic[valid_idxs]
        #     point_xyz = point_xyz[valid_idxs]
        #     # normals = normals[valid_idxs]
        #     # colors = colors[valid_idxs]
        #     sem_labels = sem_labels[valid_idxs]

        # Concatenate all the data
        data = {
            "points": relative_coords,  # N, 3
            "colors": sample['rgb'],  # N, 3
            "labels": sample['sem_labels'],  # N,
            "voxel_indices": inverse_map.cpu().numpy(),  # N,
            "query_points": query_relative_coords,  # M, 3
            "query_voxel_indices": query_indices.cpu().numpy(),  # M,
            "values": values,  # M,
            "voxel_coords": voxel_coords.cpu().numpy(),  # K, 3
            "voxel_features": voxel_features,  # K, ?
            "scene_name": sample['scene_name']
        }
        # # computing voxel center coordinates
        # voxel_centers = self.voxel_size * voxel_coords[data['voxel_indices']] + self.voxel_size / 2

        # # recovering the absolute coordinates of points
        # absolute_points = relative_coords + voxel_centers.cpu().numpy()

        # # recover query points
        # query_voxel_indices = data["query_voxel_indices"]
        # query_points = data["query_points"]
        # query_voxel_centers = self.voxel_size * voxel_coords[query_voxel_indices] + self.voxel_size / 2
        # query_absolute_points = query_points + query_voxel_centers.cpu().numpy()
        # surface_queries = query_absolute_points[mask_surface[mask]]
        # gaussian_queries = query_absolute_points[mask_gaussian[mask]]
        # uniform_queries = query_absolute_points[mask_uniform[mask]]

        # # Create Open3D point cloud for points
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(absolute_points)
        # pcd.colors = o3d.utility.Vector3dVector(np.ones_like(absolute_points) * [1, 0, 0])  # red

        # # Create Open3D point cloud for query points
        # surface_query_pcd = o3d.geometry.PointCloud()
        # surface_query_pcd.points = o3d.utility.Vector3dVector(surface_queries)
        # surface_query_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(surface_queries) * [0, 1, 0])  # green
        # # Create Open3D point cloud for query points
        # gaussian_query_pcd = o3d.geometry.PointCloud()
        # gaussian_query_pcd.points = o3d.utility.Vector3dVector(gaussian_queries)
        # gaussian_query_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(gaussian_queries) * [0, 1, 0])
        # # Create Open3D point cloud for query points
        # uniform_query_pcd = o3d.geometry.PointCloud()
        # uniform_query_pcd.points = o3d.utility.Vector3dVector(uniform_queries)
        # uniform_query_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(uniform_queries) * [0, 1, 0])

        # # visualize the point clouds
        # # merged_points_cloud = pcd + query_pcd
        # # o3d.visualization.draw_geometries([pcd, query_pcd])

        # # Save the point clouds
        # save_dir = os.path.join(self.cfg.exp_output_root_path)
        # o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.cfg.data.voxel_size) + '_original.ply'), pcd)
        # o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.cfg.data.voxel_size) + '_surface.ply'), surface_query_pcd)
        # o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.cfg.data.voxel_size) + '_gaussian.ply'), gaussian_query_pcd)
        # o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.cfg.data.voxel_size) + '_uniform.ply'), uniform_query_pcd)


        return data
    
    def get_semantic_mapping_file(file_path):
        label_mapping = {}
        with open(file_path, "r") as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file)  # skip the header
            for line in tsv_file:
                label_mapping[line[1]] = int(line[4])  # use nyu40 label set
        return label_mapping


    def random_rotation_matrix(self):
        theta = np.random.uniform(0, 2*np.pi)  # Uniformly distributed angle between 0 and 2pi
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0,              0,             1]])
        return rotation_matrix

    def remap_labels(self, labels):
        unique_labels = np.unique(labels)
        mapping = {label: i for i, label in enumerate(unique_labels)}
        new_labels = np.vectorize(mapping.get)(labels)
        return new_labels

    def read_file(self, filename: str):
        """ Read a point cloud from a file and return a dictionary. """
        plydata = PlyData.read(filename)
        vtx = plydata['vertex']

        output = dict()
        points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
        output['points'] = points.astype(np.float32)

        if self.cfg.data.has_normal:
            normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
            output['normals'] = normal.astype(np.float32)
        if self.cfg.data.has_color:
            color = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=1)
            output['colors'] = color.astype(np.float32)
        if self.cfg.data.has_label:
            segmantic_labels = vtx['label']
            for label, segs in obj_name_to_segs.items():
                for seg in segs:
                    verts = seg_to_verts[seg]
                    if label not in label_map or label_map[label] not in filtered_label_map:
                        semantic_label = -1
                    else:
                        semantic_label = filtered_label_map[label_map[label]]
                    semantic_labels[verts] = semantic_label
            output['labels'] = label.astype(np.int32)

        return output

    def load_filenames(self):
        """ Load filenames from a filelist """
        filenames, labels = [], []
        with open(self.filelist) as fid:
            lines = fid.readlines()
        for line in lines:
            tokens = line.split()
            filename = tokens[0].replace('\\', '/')
            label = tokens[1] if len(tokens) == 2 else 0
            filenames.append(filename)
            labels.append(int(label))

        num = len(filenames)
        if self.take > num or self.take < 1:
            self.take = num

        return filenames[self.intake_start:self.take+self.intake_start], labels[self.intake_start:self.take+self.intake_start]
  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]
