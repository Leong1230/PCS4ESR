import os
from tqdm import tqdm
import statistics
from statistics import mode
import numpy as np
import MinkowskiEngine
import random
import math
import h5py
import torch
from torch.utils.data import Dataset
import open3d as o3d
import matplotlib.cm as cm
from plyfile import PlyData
from pycarus.geometry.pcd import compute_udf_from_pcd 


class GeneralDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.category_num = cfg.data.category_num
        self.num_point = cfg.data.num_point
        self.rotate_num = cfg.data.rotate_num
        self.take = cfg.data.take
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
        
        # if self.in_memory:
        #     """ Load all files into memory. """
        #     print('Load files into memory from ' + self.filelist)
        #     if self.cfg.data.over_fitting:
        #         self.samples = self.read_file(os.path.join(self.dataset_root_path, self.dataset_split, self.filenames[self.random_idx]))
        #         # Voxelize the points
        #         voxel_coords, unique_map, inverse_map = MinkowskiEngine.utils.sparse_quantize(
        #             self.samples['points'], return_index=True, return_inverse=True, quantization_size=self.voxel_size)

        #         # Get unique voxel coordinates and count the number of points per voxel
        #         unique_voxel_coords, counts = np.unique(inverse_map, return_counts=True, axis=0)
                
        #         # Compute the number of unique labels per voxel
        #         labels_per_voxel = [np.unique(self.samples['labels'][inverse_map == i]) for i in range(len(unique_voxel_coords))]
        #         num_labels_per_voxel = [len(labels) for labels in labels_per_voxel]

        #         print(f"Total number of voxels: {len(unique_voxel_coords)}")
        #         print(f"Number of points per voxel: min={counts.min()}, max={counts.max()}")
        #         print(f"Number of unique labels per voxel: min={min(num_labels_per_voxel)}, max={max(num_labels_per_voxel)}")

        #         # Find voxels that meet the constraints: the number of points is close to self.num_point 
        #         # and the number of unique labels equals self.voxel_category_num
        #         valid_voxels = np.abs(counts - self.num_point) < 0.4 * self.num_point  # Adjust tolerance as needed
        #         valid_voxels = valid_voxels & (np.array(num_labels_per_voxel) == self.voxel_category_num)

        #         if not np.any(valid_voxels):
        #                 raise ValueError("No voxel meets the criteria! Please adjust the constraints.")

        #         # Generate random voxel index from the valid voxels
        #         np.random.seed(42)  # You can replace '42' with any integer you want
        #         voxel_idx = np.random.choice(np.where(valid_voxels)[0])


        #         # Create a mask for all points within the selected voxel
        #         mask = (inverse_map == voxel_idx)

        #         # Print the information
        #         print(f"Number of points in selected voxel: {counts[voxel_idx]}")
        #         print(f"Number of unique labels in selected voxel: {len(labels_per_voxel[voxel_idx])}")

        #         # Define voxel_min and voxel_range
        #         points_in_selected_voxel = self.samples['points'][mask]
        #         voxel_min = points_in_selected_voxel.min(axis=0)
        #         voxel_max = points_in_selected_voxel.max(axis=0)
        #         voxel_range = voxel_max - voxel_min

        #         # Normalize the voxel to be within the range [-1, 1]
        #         norm_points_in_selected_voxel = 2 * (points_in_selected_voxel - voxel_min) / voxel_range - 1

        #         # Compute the unsigned distance function (UDF) from the normalized point cloud (PCD).

        #         # Assuming norm_points_in_selected_voxel is a numpy array of shape (N, 3)
        #         norm_points_in_selected_voxel_tensor = torch.tensor(np.asarray(norm_points_in_selected_voxel))

        #         query_points, values = compute_udf_from_pcd(
        #             norm_points_in_selected_voxel_tensor,
        #             self.num_queries_on_surface,
        #             self.queries_stds,
        #             self.num_queries_per_std
        #         )

        #         # After computation, if query_points and values are tensors and you want to convert them back to numpy arrays:
        #         query_points = query_points.cpu().numpy()
        #         values = values.cpu().numpy()
        #         self.data = {
        #             "points": norm_points_in_selected_voxel,
        #             "colors": self.samples['colors'][mask],
        #             "labels": self.remap_labels(self.samples['labels'][mask]),
        #             "query_points": query_points,
        #             "values": values
        #         }
        #         rotated_data = []

        #         for _ in range(self.rotate_num):
        #             rotation_matrix =self.random_rotation_matrix()

        #             rotated_points = np.dot(self.data["points"], rotation_matrix.T).astype(float)
        #             rotated_query_points = np.dot(self.data["query_points"], rotation_matrix.T).astype(float)


        #             rotated_data_dict = {
        #                 "points": rotated_points,
        #                 "colors": self.data["colors"],
        #                 "labels": self.data["labels"],
        #                 "query_points": rotated_query_points,
        #                 "values": self.data["values"]
        #             }
                    
        #             rotated_data.append(rotated_data_dict)

        #         self.data = rotated_data
        #         self.visualize_voxel(self.data[0])

        #     else:
        #         self.samples = [self.read_file(os.path.join(self.dataset_root_path, self.dataset_split, f))
        #                     for f in tqdm(self.filenames, ncols=80, leave=False)]

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
        if self.cfg.data.over_fitting:
            return len(self.data)
        else:
            return len(self.filenames)

    def __getitem__(self, idx):
        sample = self.samples[idx] if self.in_memory else \
            self.read_file(os.path.join(self.dataset_root_path, self.dataset_split, self.filenames[idx]))

        # Voxelize the points
        voxel_coords, unique_map, inverse_map = MinkowskiEngine.utils.sparse_quantize(
            sample['points'], return_index=True, return_inverse=True, quantization_size=self.voxel_size)

        # Get unique voxel coordinates and count the number of points per voxel
        unique_voxel_coords, counts = np.unique(inverse_map, return_counts=True, axis=0)
        
        # Compute the number of unique labels per voxel
        labels_per_voxel = [np.unique(sample['labels'][inverse_map == i]) for i in range(len(unique_voxel_coords))]
        num_labels_per_voxel = [len(labels) for labels in labels_per_voxel]

        if self.cfg.data.over_fitting:
            print(f"Total number of voxels: {len(unique_voxel_coords)}")
            print(f"Number of points per voxel: min={counts.min()}, max={counts.max()}")
            print(f"Number of unique labels per voxel: min={min(num_labels_per_voxel)}, max={max(num_labels_per_voxel)}")

            # Find voxels that meet the constraints: the number of points is close to self.num_point 
            # and the number of unique labels equals self.voxel_category_num
            valid_voxels = np.abs(counts - self.num_point) < 0.4 * self.num_point  # Adjust tolerance as needed
            valid_voxels = valid_voxels & (np.array(num_labels_per_voxel) == self.voxel_category_num)

            if not np.any(valid_voxels):
                    raise ValueError("No voxel meets the criteria! Please adjust the constraints.")

            # Generate random voxel index from the valid voxels
            voxel_idx = np.random.choice(np.where(valid_voxels)[0])

            # Create a mask for all points within the selected voxel
            mask = (inverse_map == voxel_idx)

            # Print the information
            print(f"Number of points in selected voxel: {counts[voxel_idx]}")
            print(f"Number of unique labels in selected voxel: {len(labels_per_voxel[voxel_idx])}")

            # Define voxel_min and voxel_range
            points_in_selected_voxel = sample['points'][mask]
            voxel_min = points_in_selected_voxel.min(axis=0)
            voxel_max = points_in_selected_voxel.max(axis=0)
            voxel_range = voxel_max - voxel_min

            # Normalize the voxel to be within the range [-1, 1]
            norm_points_in_selected_voxel = 2 * (points_in_selected_voxel - voxel_min) / voxel_range - 1

            # Compute the unsigned distance function (UDF) from the normalized point cloud (PCD).

            # Assuming norm_points_in_selected_voxel is a numpy array of shape (N, 3)
            norm_points_in_selected_voxel_tensor = torch.tensor(np.asarray(norm_points_in_selected_voxel))

            query_points, values = compute_udf_from_pcd(
                norm_points_in_selected_voxel_tensor,
                self.num_queries_on_surface,
                self.queries_stds,
                self.num_queries_per_std
            )

            # After computation, if query_points and values are tensors and you want to convert them back to numpy arrays:
            query_points = query_points.cpu().numpy()
            values = values.cpu().numpy()


            data = {
                "points": norm_points_in_selected_voxel,
                "colors": sample['colors'][mask],
                "labels": sample['labels'][mask],
                "query_points": query_points,
                "values": values
            }

            self.visualize_voxel(data)
            return data
        else:
            # Define placeholders for all voxel data
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

                if num_points_in_voxel == 1:  # Skip if there are no points in the voxel
                    flag = 1
                # Increment non-empty voxels counter
                num_non_empty_voxels += 1
                voxel_min = points_in_selected_voxel.min(axis=0)
                voxel_max = points_in_selected_voxel.max(axis=0)
                voxel_range = voxel_max - voxel_min

                # Normalize the voxel to be within the range [-1, 1]
                norm_points_in_selected_voxel = 2 * (points_in_selected_voxel - voxel_min) / voxel_range - 1
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
