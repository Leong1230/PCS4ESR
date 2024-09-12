import os
from tqdm import tqdm
import numpy as np
import MinkowskiEngine
import random
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from plyfile import PlyData
import hybridpc.data.dataset.augmentation as t
from hybridpc.data.dataset.voxelizer import Voxelizer

class SceneNN(Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.num_input_points = cfg.data.num_input_points
        self.std_dev = cfg.data.std_dev
        self.intake_start = cfg.data.intake_start
        self.take = cfg.data.take
        self.input_splats = cfg.data.input_splats

        self.in_memory = cfg.data.in_memory
        self.k_neighbors = cfg.model.network.udf_decoder.k_neighbors
        self.neighbor_type = cfg.model.network.udf_decoder.neighbor_type
        self.voxelizer = Voxelizer(
            voxel_size=self.voxel_size,
            clip_bound=None,
            use_augmentation=False,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        self.filenames = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(self.dataset_root_path) for f in filenames if f.endswith('.ply')])
        if self.cfg.data.over_fitting:
            self.filenames = self.filenames[self.intake_start:self.take+self.intake_start]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """Get item."""

        # load the mesh
        scene_filename = self.filenames[idx]

        ply_data = PlyData.read(scene_filename)
        vertex = ply_data['vertex']
        pos = np.stack([vertex[t] for t in ('x', 'y', 'z')], axis=1)
        nls = np.stack([vertex[t] for t in ('nx', 'ny', 'nz')], axis=1) if 'nx' in vertex and 'ny' in vertex and 'nz' in vertex else np.zeros_like(pos)

        # if len(pos) > 200000:
        #     indices = np.random.choice(len(pos), 200000, replace=False)
        #     pos = pos[indices]
        #     nls = nls[indices]

        all_xyz = pos
        all_normals = nls
        scene_name = os.path.basename(scene_filename).replace('.ply', '')

        all_point_features = np.zeros(shape=(len(all_xyz), 0), dtype=np.float32)
        if self.cfg.model.network.use_normal:
            all_point_features = np.concatenate((all_point_features, all_normals), axis=1)
        if self.cfg.model.network.use_xyz:
            all_point_features = np.concatenate((all_point_features, all_xyz), axis=1)  # add xyz to point features

        # sample input points
        num_points = all_xyz.shape[0]
        if self.num_input_points == -1:
            xyz = all_xyz
            point_features = all_point_features
            normals = all_normals
        else:
            sample_indices = np.random.choice(num_points, self.num_input_points, replace=True)
            xyz = all_xyz[sample_indices]
            point_features = all_point_features[sample_indices]
            normals = all_normals[sample_indices]

        if isinstance(self.std_dev, (float, int)):
            std_dev = [self.std_dev] * 3  # Same standard deviation for x, y, z
        noise = np.random.normal(0, self.std_dev, xyz.shape)
        xyz += noise
        un_splats_xyz = xyz
        labels = xyz[:, 2]

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

        voxel_coords, voxel_feats, voxel_labels, indices = self.voxelizer.voxelize(xyz, point_features, labels)

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
            "scene_name": scene_name
        }

        return data
