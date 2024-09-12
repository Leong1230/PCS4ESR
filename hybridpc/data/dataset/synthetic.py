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


class Synthetic(Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, cfg, split):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        '''
        # Attributes
        if 'Synthetic' in cfg.data: # subset of mixture data
            categories = cfg.data.Synthetic.classes
            self.dataset_folder = cfg.data.Synthetic.path
            self.multi_files = cfg.data.Synthetic.multi_files
            self.file_name = cfg.data.Synthetic.pointcloud_file
        else: 
            categories = cfg.data.classes
            self.dataset_folder = cfg.data.path
            self.multi_files = cfg.data.multi_files
            self.file_name = cfg.data.pointcloud_file
        self.scale = 2.2 # Emperical scale to transfer back to physical scale
        self.cfg = cfg
        self.split = 'val' if cfg.data.over_fitting else split # use only train set for overfitting
        self.std_dev = cfg.data.std_dev * self.scale
        self.voxel_size = cfg.data.voxel_size
        self.num_input_points = cfg.data.num_input_points
        self.input_splats = cfg.data.input_splats

        self.no_except = True

        self.intake_start = cfg.data.intake_start
        self.take = cfg.data.take
        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(self.dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(self.dataset_folder, c))]

        self.metadata = {
            c: {'id': c, 'name': 'n/a'} for c in categories
        } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.dataset_folder, c)
            if not os.path.isdir(subpath):
                print('Category %s does not exist in dataset.' % c)

            if self.split is None:
                self.models += [
                    {'category': c, 'model': m} for m in [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '') ]
                ]

            else:
                split_file = os.path.join(subpath, self.split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
                
                if '' in models_c:
                    models_c.remove('')

                self.models += [
                    {'category': c, 'model': m}
                    for m in models_c
                ]
        
        # overfit in one data
        if self.cfg.data.over_fitting:
            self.models = self.models[self.intake_start:self.take+self.intake_start]

        self.voxelizer = Voxelizer(
            voxel_size=self.voxel_size,
            clip_bound=None,
            use_augmentation=False,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)
    
    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
        
        item_path = os.path.join(model_path, 'item_dict.npz')
        item_dict = np.load(item_path, allow_pickle=True)
        points_dict = np.load(file_path, allow_pickle=True)
        points = points_dict['points'] * self.scale # roughly transfer back to physical scale
        normals = points_dict['normals']
        semantics = points_dict['semantics']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            normals = normals.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
            normals += 1e-4 * np.random.randn(*normals.shape)

        # Flip the y and z axes for points and normals and move to positive quadrant
        # points = points[:, [0, 2, 1]]
        # normals = normals[:, [0, 2, 1]]
        min_values = np.min(points, axis=0)
        points -= min_values

        return points, normals, semantics

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        
        all_xyz, all_normals, all_semantics = self.load(model_path, idx, c_idx)
        scene_name = f"{category}/{model}/{idx}"

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
            labels = all_semantics
            normals = all_normals
        else:
            sample_indices = np.random.choice(num_points, self.num_input_points, replace=True)
            xyz = all_xyz[sample_indices]
            point_features = all_point_features[sample_indices]
            labels = all_semantics[sample_indices]
            normals = all_normals[sample_indices]

        if isinstance(self.std_dev, (float, int)):
            std_dev = [self.std_dev] * 3  # Same standard deviation for x, y, z
        noise = np.random.normal(0, self.std_dev, xyz.shape)
        xyz += noise
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
            # augmented_labels[:N] = labels

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
            "scene_name": scene_name
        }
        return data
