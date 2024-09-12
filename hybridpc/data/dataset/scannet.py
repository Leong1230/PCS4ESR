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
# import nearest_neighbors
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import matplotlib.cm as cm
from plyfile import PlyData
import hybridpc.data.dataset.augmentation as t
from hybridpc.data.dataset.voxelizer import Voxelizer
from nksr.svh import SparseFeatureHierarchy, SparseIndexGrid




class Scannet(Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = 'val' if self.cfg.data.over_fitting else split # use only train set for overfitting
        # self.split = 'val'
        if 'ScanNet' in cfg.data: # subset of mixture data
            self.dataset_root_path = cfg.data.ScanNet.dataset_path
            self.dataset_path = cfg.data.ScanNet.dataset_path
        else:
            self.dataset_root_path = cfg.data.dataset_path
            self.dataset_path = cfg.data.dataset_path
        self.metadata = cfg.data.metadata
        self.voxel_size = cfg.data.voxel_size
        self.take = cfg.data.take
        self.loops = cfg.data.augmentation.loops
        self.intake_start = cfg.data.intake_start
        self.uniform_sampling = cfg.data.uniform_sampling
        self.input_splats = cfg.data.input_splats
        self.num_input_points = cfg.data.num_input_points
        self.num_query_points = cfg.data.num_query_points
        self.std_dev = cfg.data.std_dev
        # Setting ratio parameters for point selection
        self.queries_stds = cfg.data.sdf_queries.queries_stds
        self.ratio_on_surface = cfg.data.sdf_queries.queries_ratio_on_surface
        self.ratio_off_surface = cfg.data.sdf_queries.queries_ratio_off_surface
        self.ratio_per_std = cfg.data.sdf_queries.queries_ratio_per_std
        self.max_dist = cfg.data.supervision.sdf.max_dist

        self.input_splat = cfg.model.network.encoder.input_splat

        # for mask query
        self.mask_queries_stds = cfg.data.udf_queries.queries_stds
        # self.mask_ratio_on_surface = cfg.data.udf_queries.queries_ratio_on_surface
        # self.mask_ratio_off_surface = cfg.data.udf_queries.queries_ratio_off_surface
        # self.mask_ratio_per_std = cfg.data.udf_queries.queries_ratio_per_std
        self.mask_max_dist = cfg.data.supervision.udf.max_dist

        self.in_memory = cfg.data.in_memory
        self.k_neighbors = cfg.model.network.udf_decoder.k_neighbors
        self.neighbor_type = cfg.model.network.udf_decoder.neighbor_type
        self.dataset_split = "test" if split == "test" else "train" # train and val scenes and all under train set
        self.data_map = {
            "train": self.metadata.train_list,
            "val": self.metadata.val_list,
            "test": self.metadata.test_list
        }
        self.voxelizer = Voxelizer(
            voxel_size=self.voxel_size,
            clip_bound=None,
            use_augmentation=False,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if cfg.data.augmentation.use_aug:
            prevoxel_transform_train = [
                t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)

        self._load_from_disk()
        self.scenes = []


        if cfg.data.augmentation.use_aug and self.split == 'train':
            N = cfg.data.augmentation.loops
        else:
            N = 1

        for idx, sample in tqdm(enumerate(self.scenes_in), desc="Voxelizing and Sample points", ncols=80):
            for i in range(N):  # This loop will run N times if conditions are satisfied, otherwise just once
                processed_sample = self.preprocess_sample_entire_scene(sample, i)
                self.scenes.append(processed_sample)

    def _load_from_disk(self):
        with open(getattr(self.metadata, f"{self.split}_list")) as f:
            self.scene_names = [line.strip() for line in f]
        
        # self.scene_names = ['scene0648_01']
        # self.scene_names = ['scene0221_00']
        self.scenes_in = []
        if self.cfg.data.over_fitting:
            self.scene_names = self.scene_names[self.intake_start:self.take+self.intake_start]
            if len(self.scene_names) == 1: # if only one scene is taken, overfit on scene 0221_00
                self.scene_names = ['scene0221_00']
                # self.scene_names = ['scene0444_01', 'scene0176_00']
                # self.scene_names = ['scene0176_00']
        for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data from disk"):
            scene_path = os.path.join(self.dataset_path, self.split, f"{scene_name}.pth")
            scene = torch.load(scene_path)
            # scene["xyz"] -= scene["xyz"].mean(axis=0)
            # scene["rgb"] = scene["rgb"].astype(np.float32) / 127.5 - 1
            scene["xyz"] = scene["xyz"].astype(np.float32)
            scene["rgb"] = scene["rgb"].astype(np.float32)
            scene['scene_name'] = scene_name
            points = scene["xyz"]
            # Step 1: Load the point cloud into Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # Step 2: Compute the OBB
            obb = pcd.get_oriented_bounding_box()
            rotated_points = self.rotate_point_cloud_to_obb_orientation(np.asarray(pcd.points), obb)
            # scene["xyz"] = rotated_points.astype(np.float32)
            self.scenes_in.append(scene) 

    def rotate_point_cloud_to_obb_orientation(self, pcd_points, obb):
        # Get the rotation matrix and center of the OBB
        R = np.asarray(obb.R)  # Rotation matrix
        center = np.asarray(obb.center)  # Center of the OBB
        angle_radians = np.arctan2(R[1, 0], R[0, 0])
        
        # Create the rotation matrix for rotation around the Z-axis
        Rz = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
        rotated_points = np.dot(pcd_points, Rz.T)
        # Optional: Translate back if you want to maintain the original position
        # rotated_points += center
        
        return rotated_points
    
    def preprocess_sample_entire_scene(self, sample, i):
        # get query samples
        # sample['xyz'] = sample['xyz'][:100]
        # sample['normal'] = sample['normal'][:100]
        # sample['sem_labels'] = sample['sem_labels'][:100]
        xyz = sample['xyz']
        normal = sample['normal']
        sem_labels = sample['sem_labels']

        point_features = np.zeros(shape=(len(xyz), 0), dtype=np.float32)
        if self.cfg.model.network.use_color:
            point_features = np.concatenate((point_features, sample['rgb']), axis=1)
        if self.cfg.model.network.use_normal:
            point_features = np.concatenate((point_features, sample['normal']), axis=1)
        if self.cfg.model.network.use_xyz:
            point_features = np.concatenate((point_features, xyz), axis=1)  # add xyz to point features

        if self.split == "train" and self.cfg.data.augmentation.use_aug:
            xyz = self.prevoxel_transforms(xyz)
            xyz = xyz.astype(np.float32)
        # Convert to tensor
        points_tensor = torch.tensor(np.asarray(xyz))
        data = {
            "xyz": xyz,  # N, 3
            "point_features": point_features,  # N, K
            "normals": normal,  # N, 3
            "labels": sample['sem_labels'],  # N,
            "scene_name": f"{sample['scene_name']}"
        }

        return data
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scene = self.scenes[idx]
        # scene['xyz'] = scene['xyz'] - scene['xyz'].mean(axis=0)
        all_xyz = scene['xyz']
        all_normals = scene['normals']
        # all_xyz = all_xyz - all_xyz.mean(axis=0)
        scene_name = scene['scene_name']
        # """ dummpy loading for time comparison """
        # scene_path = os.path.join(self.dataset_path, self.split, f"{scene_name}.pth")
        # scene = torch.load(scene_path)

        seed_value = 42
        np.random.seed(seed_value)
        if 'scene0176_00' in scene_name:
            print(f"Scene name: {scene_name}")
        # sample input points
        num_points = scene["xyz"].shape[0]
        if self.num_input_points == -1:
            xyz = scene["xyz"]
            point_features = scene['point_features']
            labels = scene['labels']
            normals = scene['normals']
        else:
            if not self.uniform_sampling:
                # Number of blocks along each axis
                num_blocks = 2
                total_blocks = num_blocks ** 3
                self.common_difference = 200
                # Calculate block sizes
                block_sizes = (all_xyz.max(axis=0) - all_xyz.min(axis=0)) / num_blocks

                # Create the number_per_block array with an arithmetic sequence
                average_points_per_block = self.num_input_points // total_blocks
                number_per_block = np.array([
                    average_points_per_block + (i - total_blocks // 2) * self.common_difference
                    for i in range(total_blocks)
                ])
                
                # Adjust number_per_block to ensure the sum is self.num_input_points
                total_points = np.sum(number_per_block)
                difference = self.num_input_points - total_points
                number_per_block[-1] += difference

                # Sample points from each block
                sample_indices = []
                block_index = 0
                total_chosen_indices = 0
                remaining_points = 0  # Points to be added to the next block
                for i in range(num_blocks):
                    for j in range(num_blocks):
                        for k in range(num_blocks):
                            block_min = all_xyz.min(axis=0) + block_sizes * np.array([i, j, k])
                            block_max = block_min + block_sizes
                            block_mask = np.all((all_xyz >= block_min) & (all_xyz < block_max), axis=1)
                            block_indices = np.where(block_mask)[0]
                            num_samples = number_per_block[block_index] + remaining_points
                            remaining_points = 0  # Reset remaining points
                            block_index += 1
                            if len(block_indices) > 0:
                                chosen_indices = np.random.choice(block_indices, num_samples, replace=True)
                                sample_indices.extend(chosen_indices)
                                total_chosen_indices += len(chosen_indices)
                                # print(f"Block {block_index} - Desired: {num_samples}, Actual: {len(chosen_indices)}")
                                if len(chosen_indices) < num_samples:
                                    remaining_points += (num_samples - len(chosen_indices))
                            else:
                                # print(f"Block {block_index} - No points available. Adding {num_samples} points to the next block.")
                                remaining_points += num_samples
                
                # print(f"Total chosen indices: {total_chosen_indices}")
            else:
                if num_points < self.num_input_points:
                    print(f"Scene {scene_name} has less than {self.num_input_points} points. Sampling with replacement.")
                    sample_indices = np.random.choice(num_points, self.num_input_points, replace=True)
                else:
                    sample_indices = np.random.choice(num_points, self.num_input_points, replace=True)
    
        xyz = scene["xyz"][sample_indices]
        point_features = scene['point_features'][sample_indices]
        labels = scene['labels'][sample_indices]
        normals = scene['normals'][sample_indices]

        if isinstance(self.std_dev, (float, int)):
            std_dev = [self.std_dev] * 3  # Same standard deviation for x, y, z
        noise = np.random.normal(0, self.std_dev, xyz.shape)
        xyz += noise
        un_splats_xyz = xyz
        if self.input_splats:
            distance = 0.015  # Fixed distance for augmentation
            epsilon = 1e-8 
            N = xyz.shape[0]
            augmented_xyz = np.zeros((N * 5, 3)).astype(np.float32)
            augmented_normals = np.zeros((N * 5, 3)).astype(np.float32)
            augmented_labels = np.zeros((N * 5, ))

            # Copy original points and normals
            augmented_xyz[:N] = xyz
            augmented_normals[:N] = normals
            augmented_labels[:N] = labels

            # Generate two perpendicular vectors to each normal using cross products
            perp_vectors_1 = np.cross(normals, np.array([1, 0, 0]))
            perp_vectors_2 = np.cross(normals, np.array([0, 1, 0]))

            # Normalize the perpendicular vectors
            perp_vectors_1 /= (np.linalg.norm(perp_vectors_1, axis=1, keepdims=True) + epsilon)
            perp_vectors_2 /= (np.linalg.norm(perp_vectors_2, axis=1, keepdims=True) + + epsilon)

            # Compute new points by moving along the perpendicular vectors
            augmented_xyz[N:2*N] = xyz + perp_vectors_1 * distance
            augmented_xyz[2*N:3*N] = xyz - perp_vectors_1 * distance
            augmented_xyz[3*N:4*N] = xyz + perp_vectors_2 * distance
            augmented_xyz[4*N:5*N] = xyz - perp_vectors_2 * distance

            # The normals remain the same for augmented points
            augmented_normals[N:2*N] = normals
            augmented_normals[2*N:3*N] = normals
            augmented_normals[3*N:4*N] = normals
            augmented_normals[4*N:5*N] = normals
            augmented_labels[N:2*N] = labels
            augmented_labels[2*N:3*N] = labels
            augmented_labels[3*N:4*N] = labels
            augmented_labels[4*N:5*N] = labels

            # Update data dictionary
            xyz = augmented_xyz
            normals = augmented_normals
            labels = augmented_labels
            # normals[N:5*N] = np.zeros((4*N, 3)).astype(np.float32)

            point_features = np.zeros(shape=(len(xyz), 0), dtype=np.float32)
            if self.cfg.model.network.use_normal:
                point_features = np.concatenate((point_features, normals), axis=1)
            if self.cfg.model.network.use_xyz:
                point_features = np.concatenate((point_features, xyz), axis=1)  # add xyz to point features

            point_features = point_features.astype(np.float32)
        # else:
        voxel_coords, voxel_feats, voxel_labels, indices= self.voxelizer.voxelize(xyz, point_features, labels)

        # compute query indices and relative coordinates
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 
        relative_coords = xyz - voxel_center[indices]

        
        data = {
            "all_xyz": all_xyz,
            "all_normals": all_normals,
            "xyz": xyz,  # N, 3
            "un_splats_xyz": un_splats_xyz,  # N, 3
            "normals": normals,  # N, 3
            "relative_coords": relative_coords,  # N, 3
            "point_features": point_features,  # N, 3
            "indices": indices,  # N,
            "voxel_coords": voxel_coords,  # K, 3
            "voxel_feats": voxel_feats,  # K, 3
            "scene_name": scene['scene_name']
        }

        return data

