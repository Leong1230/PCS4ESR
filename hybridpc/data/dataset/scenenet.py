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




class Scenenet(Dataset):
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
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.take = cfg.data.take
        self.intake_start = cfg.data.intake_start
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
        self.point_density = 20

        self.in_memory = cfg.data.in_memory
        self.k_neighbors = cfg.model.network.udf_decoder.k_neighbors
        self.neighbor_type = cfg.model.network.udf_decoder.neighbor_type
        self.dataset_split = "test" if split == "test" else "train" # train and val scenes and all under train set
        self.voxelizer = Voxelizer(
            voxel_size=self.voxel_size,
            clip_bound=None,
            use_augmentation=False,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        self.filenames = [
        "1Bathroom/107_labels.obj.ply",
        "1Bathroom/1_labels.obj.ply",
        "1Bathroom/28_labels.obj.ply",
        "1Bathroom/29_labels.obj.ply",
        "1Bathroom/4_labels.obj.ply",
        "1Bathroom/5_labels.obj.ply",
        "1Bathroom/69_labels.obj.ply",
        "1Bedroom/3_labels.obj.ply",
        "1Bedroom/77_labels.obj.ply",
        "1Bedroom/bedroom27.obj.ply",
        "1Bedroom/bedroom_1.obj.ply",
        "1Bedroom/bedroom_68.obj.ply",
        "1Bedroom/bedroom_wenfagx.obj.ply",
        "1Bedroom/bedroom_xpg.obj.ply",
        "1Kitchen/1-14_labels.obj.ply",
        "1Kitchen/102.obj.ply",
        "1Kitchen/13_labels.obj.ply",
        "1Kitchen/2.obj.ply",
        "1Kitchen/35_labels.obj.ply",
        "1Kitchen/kitchen_106_blender_name_and_mat.obj.ply",
        "1Kitchen/kitchen_16_blender_name_and_mat.obj.ply",
        "1Kitchen/kitchen_76_blender_name_and_mat.obj.ply",
        "1Living-room/cnh_blender_name_and_mat.obj.ply",
        "1Living-room/living_room_33.obj.ply",
        "1Living-room/lr_kt7_blender_scene.obj.ply",
        "1Living-room/pg_blender_name_and_mat.obj.ply",
        "1Living-room/room_89_blender.obj.ply",
        "1Living-room/room_89_blender_no_paintings.obj.ply",
        "1Living-room/yoa_blender_name_mat.obj.ply",
        "1Office/2_crazy3dfree_labels.obj.ply",
        "1Office/2_hereisfree_labels.obj.ply",
        "1Office/4_3dmodel777.obj.ply",
        "1Office/4_hereisfree_labels.obj.ply",
        "1Office/7_crazy3dfree_old_labels.obj.ply",
        ]
        self.filenames = [os.path.join(self.dataset_root_path , filename) for filename in self.filenames]
        self.filenames.sort()

        if self.cfg.data.over_fitting:
            self.filenames = self.filenames[self.intake_start:self.take+self.intake_start]


    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """Get item."""

        # load the mesh
        scene_filename = self.filenames[idx]

        data = np.loadtxt(scene_filename+".xyz", dtype=np.float32)

        pos = data[:,:3] 
        nls = data[:,3:]
        # # Flip the y and z axes
        # pos[:, [1, 2]] = pos[:, [2, 1]]
        # nls[:, [1, 2]] = nls[:, [2, 1]]


        all_xyz = pos
        all_normals = nls
        scene_name = scene_filename.split('/')[-1].replace('.obj', '').replace('.ply', '')

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
        labels = xyz[:, 2]
        un_splats_xyz = xyz
        if self.input_splats:
            distance = 0.02  # Fixed distance for augmentation
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
            normals[N:5*N] = np.zeros((4*N, 3)).astype(np.float32)

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
            "scene_name": scene_name
        }

        return data
