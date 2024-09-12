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
import open3d as o3d
import matplotlib.cm as cm
from plyfile import PlyData
import hybridpc.data.dataset.augmentation as t
from hybridpc.data.dataset.voxelizer import Voxelizer

class ScannetRangeUDF(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        # self.split = 'train' if self.cfg.data.over_fitting else split # use only train set for overfitting
        self.split = split
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.take = cfg.data.take
        self.loops = cfg.data.augmentation.loops
        self.intake_start = cfg.data.intake_start
        self.num_input_points = cfg.data.num_input_points
        self.num_query_points = cfg.data.num_query_points
        # Setting ratio parameters for point selection
        self.queries_stds = cfg.data.udf_queries.queries_stds
        self.ratio_on_surface = cfg.data.udf_queries.queries_ratio_on_surface
        self.ratio_off_surface = cfg.data.udf_queries.queries_ratio_off_surface
        self.ratio_per_std = cfg.data.udf_queries.queries_ratio_per_std
        self.max_dist = cfg.data.udf_queries.max_dist
        self.in_memory = cfg.data.in_memory
        self.dataset_split = "test" if split == "test" else "train" # train and val scenes and all under train set
        self.data_map = {
            "train": cfg.data.metadata.train_list,
            "val": cfg.data.metadata.val_list,
            "test": cfg.data.metadata.test_list
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

        self.scenes = []


        if cfg.data.augmentation.use_aug and self.split == 'train':
            N = cfg.data.augmentation.loops
        else:
            N = 1

        for idx, sample in tqdm(enumerate(self.scenes_in), desc="Voxelizing and Sample points", ncols=80):
            for i in range(N):  # This loop will run N times if conditions are satisfied, otherwise just once
                processed_sample = self.preprocess_sample_entire_scene(sample, i)
                self.scenes.append(processed_sample)

    
    def __len__(self):
        return len(self.scenes)

    def get_train_batch(self, idx):
        try:
            path = self.data[idx]
            input_path = self.path + path
            samples_path = self.path + path
            on_surface_path = input_path + '/on_surface_{}points.npz'.format(self.num_on_surface_points)
            on_surface = np.load(on_surface_path)
            on_surface_points = np.array(on_surface['point_cloud'], dtype=np.float32)
            on_surface_label_path = input_path + '/on_surface_{}labels.npz'.format(self.num_on_surface_points)
            on_surface_labels = np.array(np.load(on_surface_label_path)['full'], dtype=np.float32)

            if self.mode == 'test':
                return {'on_surface_points': np.array(on_surface_points, dtype=np.float32), 'path' : path}

            ############################################################
            # prepare off-surface points from '/boundary_{}_samples.npz'.
            ############################################################
            input_dict = {}
            input_off_surface_points = []
            input_off_surface_df = []
            input_off_surface_labels = []
            for i, num in enumerate(self.num_samples):
                boundary_sample_ppath=samples_path+'/boundary_{}_points_{}.npz'.format(self.sample_sigmas[i],10*self.num_on_surface_points)
                boundary_sample_lpath=samples_path+'/boundary_{}_labels_{}.npz'.format(self.sample_sigmas[i],10*self.num_on_surface_points)
                boundary_sample_labels=np.load(boundary_sample_lpath)['full']
                boundary_samples=np.load(boundary_sample_ppath)
                boundary_sample_points=boundary_samples['points']
                boundary_sample_df=boundary_samples['df']
                
                subsample_indices = torch.randint(0, len(boundary_sample_points), (num,))
                input_off_surface_points.extend(boundary_sample_points[subsample_indices])
                input_off_surface_df.extend(boundary_sample_df[subsample_indices])
                input_off_surface_labels.extend(boundary_sample_labels[subsample_indices])

            num_on_surface = len(on_surface_points)
            num_off_surface = self.num_off_surface_points

            input_off_surface_points = np.array(input_off_surface_points, dtype=np.float32)
            input_on_surface_df = [0 for i in range(num_on_surface)]  # [0, 0, 0, ...], len(input_on_surface_df) = 10000
            df = input_on_surface_df + input_off_surface_df

            assert len(input_off_surface_points) == self.num_off_surface_points
            assert len(input_off_surface_df) == self.num_off_surface_points
            assert len(df) == num_on_surface + num_off_surface


            ############################################################
            # prepare on-surface points for RangeUDF input.
            ############################################################

            if not self.opt.fixed_input:
                permutation = torch.randperm(len(on_surface_points))
                on_surface_points = on_surface_points[permutation]
                on_surface_labels = on_surface_labels[permutation]
            else:
                print('Fixed input order')

            if self.opt.in_dim == 3:
                feature = on_surface_points
            elif self.opt.in_dim == 6:
                colors = on_surface['colors'] / 255
                feature = np.concatenate((on_surface_points, colors[permutation]), axis=-1)

            ############################
            # group semantic branch data
            ############################
            input_sem_interp_idx = []
            if self.mode == 'train':
                semantic_branch_points = on_surface_points
                semantic_branch_labels = on_surface_labels
                input_labels = np.concatenate((on_surface_labels, semantic_branch_labels))
                input_sem_interp_idx.append(nearest_neighbors.knn(on_surface_points, on_surface_points, self.opt.num_interp + 1, omp=True)[:, 1:])
  
            elif self.mode =='val':
                semantic_branch_points = input_off_surface_points
                semantic_branch_labels = input_off_surface_labels
                input_labels = np.concatenate((on_surface_labels, semantic_branch_labels))
                input_sem_interp_idx.append(nearest_neighbors.knn(on_surface_points, input_off_surface_points, self.opt.num_interp, omp=True))
            assert len(input_sem_interp_idx) == 1

            ############################
            # encoder input
            ############################
            input_on_surface_points = []
            input_neighbors = []
            input_pools = []
            input_on_interp_idx = []
            input_off_interp_idx = []
            input_off_interp_idx.append(nearest_neighbors.knn(on_surface_points, input_off_surface_points, self.opt.num_interp, omp=True))
            for i in range(self.opt.num_layers):

                neigh_idx = nearest_neighbors.knn(on_surface_points, on_surface_points, self.opt.num_neighbors, omp=True)
                sub_points = on_surface_points[:len(on_surface_points) // self.opt.sub_sampling_ratio[i]]
                down_sample = neigh_idx[:len(on_surface_points) // self.opt.sub_sampling_ratio[i]]
                on_up_sample = nearest_neighbors.knn(sub_points, on_surface_points, 1, omp=True)

                input_on_surface_points.append(on_surface_points)
                input_neighbors.append(neigh_idx)
                input_pools.append(down_sample)
                input_on_interp_idx.append(on_up_sample)
                on_surface_points = sub_points

            ############################################################
            # prepare input dict.
            ############################################################
            # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
            
            targets = - np.ones(len(input_labels))
            for i, c in enumerate(self.valid_labels):
               targets[input_labels[:, 0] == c] = i

            input_dict['feature'] = np.array(feature, dtype=np.float32)
            input_dict['df'] = np.array(df, dtype=np.float32)
            input_dict['targets'] = np.array(targets, dtype=np.long)

            input_dict['on_surface_points'] = input_on_surface_points
            input_dict['off_surface_points'] = input_off_surface_points
            input_dict['sem_branch_points'] = semantic_branch_points

            input_dict['input_neighbors'] = input_neighbors
            input_dict['input_pools'] = input_pools

            input_dict['on_interp_idx'] = input_on_interp_idx
            input_dict['off_interp_idx'] = input_off_interp_idx
            input_dict['sem_interp_idx'] = input_sem_interp_idx

            input_dict['path'] = path
            return input_dict
        except:
            print('Error with {}: {}'.format(path, traceback.format_exc()))
            raise

    def __getitem__(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scene = self.scenes[idx]
        all_xyz = scene['xyz']
        query_xyz = scene['query_xyz']
        values = scene['values']
        scene_name = scene['scene_name']
        # sample input points
        num_points = scene["xyz"].shape[0]
        if self.num_input_points == -1:
            xyz = scene["xyz"]
            point_features = scene['point_features']
            labels = scene['labels']
        else:
            sample_indices = np.random.choice(num_points, self.num_input_points, replace=True)
            xyz = scene["xyz"][sample_indices]
            point_features = scene['point_features'][sample_indices]
            labels = scene['labels'][sample_indices]

        #voxelization
        voxel_coords, voxel_feats, voxel_labels, indices= self.voxelizer.voxelize(xyz, point_features, labels)

        # compute query indices and relative coordinates
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 
        # query_indices, _, _ = knn(torch.from_numpy(query_xyz).to(device), torch.from_numpy(voxel_center).to(device), 1)
        query_indices = nearest_neighbors.knn(voxel_center, query_xyz, 1, omp=True)
        query_indices = query_indices[:, 0]

        relative_coords = xyz - voxel_center[indices]
        query_relative_coords = query_xyz - voxel_center[query_indices]

        # remove query_points outside the voxel
        lower_bound = -self.voxel_size / 2
        upper_bound = self.voxel_size / 2
        mask = (query_relative_coords >= lower_bound) & (query_relative_coords <= upper_bound)
        mask = np.all(mask,-1)
        query_xyz, query_indices, query_relative_coords, values = query_xyz[mask], query_indices[mask], query_relative_coords[mask], values[mask]

        # sample query points
        num_query_points = len(query_indices)
        sample_indices = np.random.choice(num_query_points, self.num_query_points, replace=True)
        query_xyz, query_values, query_indices, query_relative_coords = query_xyz[sample_indices], values[sample_indices], query_indices[sample_indices], query_relative_coords[sample_indices]

        data = {
            "all_xyz": all_xyz,
            "xyz": xyz,  # N, 3
            "relative_coords": relative_coords,  # N, 3
            "point_features": point_features,  # N, 3
            "indices": indices,  # N,
            "voxel_coords": voxel_coords,  # K, 3
            "voxel_feats": voxel_feats,  # K, 3
            "query_xyz": query_xyz, #M,
            "query_relative_coords": query_relative_coords,  # M, 3
            "query_indices": query_indices,  # M,
            "values": query_values,  # M,
            "scene_name": scene['scene_name']
        }

        return data