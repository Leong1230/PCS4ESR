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
from pycarus.geometry.pcd import compute_udf_from_pcd, farthest_point_sampling
from plyfile import PlyData
from hybridpc.util.pc import crop
from hybridpc.util.transform import jitter, flip, rotz, elastic



class GeneralDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.voxel_category_num = cfg.data.voxel_category_num
        self.num_point = cfg.data.num_point
        self.num_queries_on_surface = cfg.data.num_queries_on_surface
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
            self.random_idx = random.randint(0, len(self.filenames) - 1)
        
        if self.in_memory:
            """ Load all files into memory. """
            print('Load files into memory from ' + self.filelist)
            if self.cfg.data.over_fitting:
                self.read_file(os.path.join(self.dataset_root_path, self.dataset_split, self.filenames[self.random_idx]))
            else:
                self.samples = [self.read_file(os.path.join(self.dataset_root_path, self.dataset_split, f))
                            for f in tqdm(self.filenames, ncols=80, leave=False)]

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

        # Convert output['points'] and output['query_points'] to PointCloud
        normalized_labels = output['labels'].astype(float) / output['labels'].max()  # Normalize labels to [0, 1]
        colors = cmap(output['labels'])[:, :3] * 255  # Use labels for color
        pcd = numpy_to_open3d(output['points'], colors)
        query_pcd = numpy_to_open3d(output['query_points'], np.array([0, 255, 0]))  # Use green for query_points

        # Create a voxel grid
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=-1, max_bound=1)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_axis_aligned_bounding_box(bbox, voxel_size=self.voxel_size)

        # Visualize point cloud and voxel grid
        o3d.visualization.draw_geometries([pcd, query_pcd, voxel_grid])

    def __len__(self):
        if self.cfg.data.over_fitting:
            return 1
        else:
            return len(self.filenames)


    def __getitem__(self, idx):
        if self.cfg.data.over_fitting:
            sample = self.samples[idx] if self.in_memory else \
                self.read_file(os.path.join(self.dataset_root_path, self.dataset_split, self.filenames[self.random_idx]))

            # Voxelize the points
            voxel_coords, unique_map = MinkowskiEngine.utils.sparse_quantize(
                sample['points'], return_index=True, quantization_size=self.voxel_size)

            # Get unique voxel coordinates and count the number of points per voxel
            unique_voxel_coords, counts = np.unique(voxel_coords, return_counts=True, axis=0)
            
            # Compute the number of unique labels per voxel
            labels_per_voxel = [np.unique(sample['labels'][unique_map == i]) for i in range(len(unique_voxel_coords))]
            num_labels_per_voxel = [len(labels) for labels in labels_per_voxel]

            # Find voxels that meet the constraints: the number of points is close to self.num_point 
            # and the number of unique labels equals self.voxel_category_num
            valid_voxels = np.abs(counts - self.num_point) < 0.1 * self.num_point  # Adjust tolerance as needed
            valid_voxels = valid_voxels & (np.array(num_labels_per_voxel) == self.voxel_category_num)

            if not np.any(valid_voxels):
                    raise ValueError("No voxel meets the criteria! Please adjust the constraints.")
            # The code for generating the voxel_idx and beyond remains the same.


            # Generate random voxel index from the valid voxels
            voxel_idx = np.random.choice(np.where(valid_voxels)[0])

            # Create a mask for all points within the selected voxel
            mask = np.all(voxel_coords[unique_map] == unique_voxel_coords[voxel_idx], axis=1)

            # Print the information
            print(f"Total number of voxels: {len(unique_voxel_coords)}")
            print(f"Number of points per voxel: min={counts.min()}, max={counts.max()}")
            print(f"Number of unique labels per voxel: min={min(num_labels_per_voxel)}, max={max(num_labels_per_voxel)}")
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
            query_points, values = compute_udf_from_pcd(
                norm_points_in_selected_voxel, 
                self.num_queries_on_surface,
            )

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



