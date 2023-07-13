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
from pycarus.geometry.pcd import compute_udf_from_pcd, knn 


class GeneralDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = 'train' if self.cfg.data.over_fitting else split # use only train set for overfitting
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.category_num = cfg.data.category_num
        self.num_point = cfg.data.num_point
        self.rotate_num = cfg.data.rotate_num
        self.take = cfg.data.take
        self.use_relative = cfg.data.use_relative
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
        self.filelist = os.path.join(self.dataset_root_path, self.data_map[split]) # train.txt, val.txt, test.txt
        self.filenames, self.labels = self.load_filenames()

        if self.cfg.data.over_fitting:
            # self.random_idx = random.randint(0, len(self.filenames) - 1)
            self.random_idx = 0
        
        if self.in_memory:
            print('Load ' + self.dataset_split + ' dataset into memory')
            self.samples = [self.read_file(os.path.join(self.dataset_root_path, self.dataset_split, f))
                            for f in tqdm(self.filenames, ncols=80, leave=False)]    
            self.data = []
            for sample in tqdm(self.samples, desc="Voxelizing and Sample points", ncols=80):
                if self.sample_entire_scene:
                    processed_sample = self.preprocess_sample_entire_scene(sample)
                else:
                    processed_sample = self.preprocess_sample(sample)
                self.data.append(processed_sample)

    def preprocess_sample(self, sample):
        # Voxelize the points
        voxel_coords, unique_map, inverse_map = MinkowskiEngine.utils.sparse_quantize(
            sample['points'], return_index=True, return_inverse=True, quantization_size=self.voxel_size)

        # Get unique voxel coordinates and count the number of points per voxel
        unique_voxel_coords, counts = np.unique(inverse_map, return_counts=True, axis=0)
        
        # Compute the number of unique labels per voxel
        labels_per_voxel = [np.unique(sample['labels'][inverse_map == i]) for i in range(len(unique_voxel_coords))]
        num_labels_per_voxel = [len(labels) for labels in labels_per_voxel]
        all_points = []
        all_colors = []
        all_labels = []
        all_query_points = []
        all_values = []
        all_voxel_indices = []
        all_query_voxel_indices = []

        # Initialize counter for non-empty voxels
        num_non_empty_voxels = 0
        for voxel_idx in range(len(unique_voxel_coords)):
            mask = (inverse_map == voxel_idx)

            points_in_selected_voxel = sample['points'][mask]
            num_points_in_voxel = len(points_in_selected_voxel)

            if num_points_in_voxel == 0:  # Skip if there are no points in the voxel
                continue
            # Assume points_in_selected_voxel is your points tensor of shape (N, 3)
            # and voxel_size is the size of your voxel

            # Shift the points to the range [0, voxel_size]
            points_in_selected_voxel -= np.min(points_in_selected_voxel, 0)

            # Scale to the range [0, 1]
            points_in_selected_voxel /= self.voxel_size

            # Shift and scale to the range [-1, 1]
            norm_points_in_selected_voxel = 2.0 * points_in_selected_voxel - 1.0

            norm_points_in_selected_voxel_tensor = torch.tensor(np.asarray(norm_points_in_selected_voxel))

            # Calculate number of queries based on the ratio and the number of points in the voxel
            num_queries_on_surface = int(num_points_in_voxel * self.ratio_on_surface + 1)
            num_queries_per_std = [int(num_points_in_voxel * self.ratio_per_std + 1)] * 4  # A list of 4 equal values

            query_points, values = compute_udf_from_pcd(
                norm_points_in_selected_voxel_tensor,
                num_queries_on_surface,
                self.queries_stds,
                num_queries_per_std
            )

            # Convert tensors to numpy arrays:
            query_points = query_points.cpu().numpy()
            values = values.cpu().numpy()

            # Check for NaN values:
            nan_mask_query_points = np.isnan(query_points).any(axis=1)
            nan_mask_values = np.isnan(values)

            # Check if there are any NaNs in either query_points or values:
            nan_mask_combined = nan_mask_query_points | nan_mask_values

            # If there are any NaNs, print a warning and remove them:
            if np.any(nan_mask_combined):
                # print(f"Warning: found NaN in data, removing corresponding rows.")
                query_points = query_points[~nan_mask_combined]
                values = values[~nan_mask_combined]

            if self.use_relative:
                all_points.append(norm_points_in_selected_voxel)  # Output points in not normalized within each voxel
            else:
                all_points.append(points_in_selected_voxel)  # Output points in not normalized within each voxel
            all_colors.append(sample['colors'][mask])
            all_labels.append(sample['labels'][mask])
            all_query_points.append(query_points)  # Output query points in normalized within each voxel
            all_values.append(values)
            all_voxel_indices.append(np.full((points_in_selected_voxel.shape[0],), voxel_idx))
            all_query_voxel_indices.append(np.full((query_points.shape[0],), voxel_idx))

        # Concatenate all the data
        data = {
            "points": np.concatenate(all_points, axis=0), #N, 3
            "colors": np.concatenate(all_colors, axis=0), #N, 3
            "labels": np.concatenate(all_labels, axis=0), #N, 
            "voxel_indices": np.concatenate(all_voxel_indices, axis=0), #N, 
            "query_points": np.concatenate(all_query_points, axis=0), # M, 3
            "values": np.concatenate(all_values, axis=0), # M,
            "query_voxel_indices": np.concatenate(all_query_voxel_indices, axis=0), # M, 3
            "voxel_coords": voxel_coords.cpu().numpy() # K, 3
            # "num_non_empty_voxels": num_non_empty_voxels # int
        }
        return data
    
    def preprocess_sample_entire_scene(self, sample):
        # Voxelize the points
        voxel_coords, unique_map, inverse_map = MinkowskiEngine.utils.sparse_quantize(
            sample['points'], return_index=True, return_inverse=True, quantization_size=self.voxel_size)

        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 # compute voxel_center in orginal coordinate system (torch.tensor)
        points = sample['points']

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

        # find the nearest voxel center for each query point
        query_indices, _, _ = knn(query_points, torch.tensor(voxel_center).clone().detach(), 1)
        query_indices = query_indices[:, 0]
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

        # Concatenate all the data
        data = {
            "points": relative_coords,  # N, 3
            "colors": sample['colors'],  # N, 3
            "labels": sample['labels'],  # N,
            "voxel_indices": inverse_map,  # N,
            "query_points": query_relative_coords,  # M, 3
            "query_voxel_indices": query_indices,  # M,
            "values": values,  # M,
            "voxel_coords": voxel_coords.cpu().numpy()  # K, 3
        }
        return data



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
            label = vtx['label']
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

        return filenames[:self.take], labels[:self.take]
  
    def visualize_voxel(self, output):

        # Convert data to Open3D format
        def numpy_to_open3d(data, colors):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # assuming colors are in [0, 255]
            return pcd

        # Color mapping for labels
        cmap = cm.get_cmap("tab20")

        # Normalize labels to [0, 1]
        normalized_labels = output['labels'].astype(float) / output['labels'].max() 

        # Use labels for color
        colors = cmap(normalized_labels)[:, :3] * 255  

        pcd = numpy_to_open3d(output['points'], colors)
        # Create an array of the same length as output['query_points'], all of color green
        green_color = np.tile(np.array([0, 255, 0]), (len(output['query_points']), 1))

        query_pcd = numpy_to_open3d(output['query_points'], green_color)

        # Create a voxel grid
        min_bound = np.array([-1, -1, -1])
        max_bound = np.array([1, 1, 1])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

        # Define the eight corners of the bounding box
        points = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]]
        ])

        # Define the twelve edges of the bounding box
        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # top edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ])

        # Create a LineSet and color the lines
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.paint_uniform_color([0, 0, 0])  # black color

        # Visualize the bounding box
        o3d.visualization.draw_geometries([line_set])

        # Visualize point cloud and voxel grid
        o3d.visualization.draw_geometries([pcd, line_set])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]
