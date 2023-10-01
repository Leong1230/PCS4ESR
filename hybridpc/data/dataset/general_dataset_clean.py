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
import hybridpc.data.dataset.augmentation as t
from hybridpc.data.dataset.voxelizer import Voxelizer
from pycarus.geometry.pcd import compute_udf_from_pcd, knn, compute_sdf_from_pcd
from hybridpc.util.transform import jitter, flip, rotz, elastic, crop



class GeneralDataset(Dataset):
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
        self.split = 'train' if self.cfg.data.over_fitting else split # use only train set for overfitting
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.num_point = cfg.data.num_point
        self.rotate_num = cfg.data.rotate_num
        self.take = cfg.data.take
        self.loops = cfg.data.augmentation.loops
        self.intake_start = cfg.data.intake_start
        self.use_relative = cfg.data.use_relative 
        self.max_num_point = cfg.data.max_num_point
        if self.cfg.model.training_stage == 1:
            self.k_neighbors = cfg.model.network.udf_decoder.k_neighbors # 1 for no interpolation
        else:
            self.k_neighbors = cfg.model.network.seg_decoder.k_neighbors
        self.sample_entire_scene = cfg.data.udf_queries.sample_entire_scene
        self.num_queries_on_surface = cfg.data.udf_queries.num_queries_on_surface
        self.queries_stds = cfg.data.udf_queries.queries_stds
        self.ratio_on_surface = cfg.data.udf_queries.queries_ratio_on_surface
        self.ratio_per_std = cfg.data.udf_queries.queries_ratio_per_std
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
            use_augmentation=True if self.cfg.data.augmentation.method == 'original' else False,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if cfg.data.augmentation.use_aug:
            prevoxel_transform_train = [
                t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                # t.RandomDropout(0.2),
                # t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                # t.ChromaticAutoContrast(),
                t.ChromaticTranslation(cfg.data.augmentation.color_trans_ratio),
                t.ChromaticJitter(cfg.data.augmentation.color_jitter_std),
                t.HueSaturationTranslation(
                    cfg.data.augmentation.hue_max, cfg.data.augmentation.saturation_max),
            ]
            self.input_transforms = t.Compose(input_transforms)
        self._load_from_disk()
        self.scenes = []
        self.aug_scene_names = []


        if cfg.data.augmentation.use_aug and cfg.data.augmentation.method == 'N-times' and self.split == 'train':
            N = cfg.data.augmentation.loops
        else:
            N = 1

        for idx, sample in tqdm(enumerate(self.scenes_in), desc="Voxelizing and Sample points", ncols=80):
            for i in range(N):  # This loop will run N times if conditions are satisfied, otherwise just once
                processed_sample = self.preprocess_sample_entire_scene(sample, i)
                self.aug_scene_names.append(processed_sample['scene_name'])
                self.scenes.append(processed_sample)

    def _load_from_disk(self):
        with open(getattr(self.cfg.data.metadata, f"{self.split}_list")) as f:
            self.scene_names = [line.strip() for line in f]
        self.scenes_in = []
        if self.cfg.data.over_fitting:
            self.scene_names = self.scene_names[self.intake_start:self.take+self.intake_start]
        for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data from disk"):
            scene_path = os.path.join(self.cfg.data.dataset_path, self.split, f"{scene_name}.pth")
            scene = torch.load(scene_path)
            scene["xyz"] -= scene["xyz"].mean(axis=0)
            scene["rgb"] = scene["rgb"].astype(np.float32) / 127.5 - 1
            scene['scene_name'] = scene_name
            self.scenes_in.append(scene) 

    
    def preprocess_sample_entire_scene(self, sample, i):
        # Voxelize the points
        xyz = sample['xyz']
        sem_labels = sample['sem_labels']

        point_features = np.zeros(shape=(len(xyz), 0), dtype=np.float32)
        if self.cfg.model.network.use_color:
            point_features = np.concatenate((point_features, sample['rgb']), axis=1)
        if self.cfg.model.network.use_normal:
            point_features = np.concatenate((point_features, sample['normal']), axis=1)
        if self.cfg.model.network.use_xyz:
            point_features = np.concatenate((point_features, xyz), axis=1)  # add xyz to point features

        if self.split == "train" and self.cfg.data.augmentation.use_aug and self.cfg.data.augmentation.method == 'N-times':
            xyz = self.prevoxel_transforms(xyz)
            xyz = xyz.astype(np.float32)
        voxel_coords, feats, labels, inds_reconstruct = self.voxelizer.voxelize(xyz, point_features, sem_labels)
        voxel_coords = voxel_coords.astype(np.float32)

        # Calculate k-neighbors of original points
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inds_reconstruct, _, _ = knn(torch.from_numpy(xyz).to(device), torch.from_numpy(voxel_center).to(device), self.k_neighbors) #N, K, 3
        inds_reconstruct = inds_reconstruct.cpu().numpy()


        if self.cfg.model.training_stage == 2:
            data = {
                "xyz": xyz,  # N, 3
                "point_features": point_features,  # N, 3
                "features": feats,  # N, 3
                "labels": sample['sem_labels'],  # N,
                "voxel_coords": voxel_coords,
                "voxel_indices": inds_reconstruct,  # N, K
                "scene_name": f"{sample['scene_name']}_{i}"
            }
            return data
        
        query_absolute_points, values, unmasked_values, query_indices = self.sample_points(xyz, voxel_coords)
        data = {
            "xyz": xyz,  # N, 3
            "point_features": point_features,  # N, 3
            "features": feats,  # N, 3
            "labels": sample['sem_labels'],  # N,
            "voxel_coords": voxel_coords,
            "voxel_indices": inds_reconstruct,  # N, or N, K
            "query_absolute_points": query_absolute_points,
            "query_voxel_indices": query_indices,  # M, or M, K
            "values": values,  # M,
            "scene_name": f"{sample['scene_name']}_{i}"
        }

        return data
  
    def sample_points(self, xyz, voxel_coords):
        # Calculate number of queries based on the ratio and the number of points
        num_queries_on_surface = int(len(xyz) * self.ratio_on_surface + 1)
        num_queries_per_std = [int(len(xyz) * self.ratio_per_std + 1)] * 4

        min_range = np.min(xyz, axis=0)
        max_range = np.max(xyz, axis=0)

        query_points, values = compute_udf_from_pcd(
            torch.from_numpy(xyz),
            num_queries_on_surface,
            self.queries_stds,
            num_queries_per_std,
            (torch.tensor(min_range), torch.tensor(max_range)),
        )

        voxel_coords = torch.from_numpy(voxel_coords).int()
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0  # compute voxel center in orginal coordinate system (torch.tensor)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_points = query_points.to(device)
        voxel_center = voxel_center.to(device)
        query_indices, _, _ = knn(query_points, voxel_center, self.k_neighbors)    
        
        # remove query_points outside the voxel
        lower_bound = -self.voxel_size / 2
        upper_bound = self.voxel_size / 2
        
        query_relative_points = query_points - voxel_center[query_indices[:, 0]] # N, 3
        mask = ((query_relative_points >= lower_bound) & (query_relative_points <= upper_bound)).cpu().numpy()
        mask = np.all(mask, -1)
        
        query_indices = query_indices.cpu().numpy()[mask]
        query_absolute_points = query_points.cpu().numpy()[mask]
        unmasked_values = values.cpu().numpy()
        values = values.cpu().numpy()[mask]

        return query_absolute_points, values, unmasked_values, query_indices

    def __len__(self):
        return len(self.aug_scene_names)

    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        xyz = scene['xyz']
        point_features = scene['point_features']
        voxel_coords = scene['voxel_coords']
        feats = scene['features']
        labels = scene['labels']
        inds_reconstruct  = scene['voxel_indices']
        voxel_center =  voxel_coords * self.voxel_size + self.voxel_size / 2.0
        relative_coords = xyz[:, np.newaxis] - voxel_center[inds_reconstruct] # N, K, 3
        
        if self.cfg.model.training_stage == 2:
            if self.cfg.data.augmentation.method == 'N-times':
                data = {
                    "xyz": xyz,  # N, 3
                    "points": relative_coords,  # N, K , 3
                    "point_features": scene['point_features'],  # N, 3
                    "labels": scene['labels'],  # N,
                    "voxel_indices": inds_reconstruct,  # N, or N, K
                    "voxel_coords": voxel_coords,  # K, 3
                    "voxel_features": scene['features'],  # K, ?
                    "scene_name": scene['scene_name']
                }
                return data
            
            else: 
                if self.split == 'train':
                    xyz = self.prevoxel_transforms(xyz)
                    voxel_coords, feats, labels, inds_reconstruct = self.voxelizer.voxelize(xyz, point_features, labels)
                    voxel_coords, feats, labels = self.input_transforms(voxel_coords, feats, labels)
                else:
                    voxel_coords, feats, labels, inds_reconstruct = self.voxelizer.voxelize(xyz, point_features, labels)
                voxel_center =  voxel_coords * self.voxel_size + self.voxel_size / 2.0
                relative_coords = xyz[:, np.newaxis] - voxel_center[inds_reconstruct] # N, K, 3
                data = {
                    "xyz": xyz,  # N, 3
                    "points": relative_coords,  # N, K , 3
                    "point_features": scene['point_features'],  # N, 3
                    "labels": labels,  # N,
                    "voxel_indices": inds_reconstruct,  # N, or N, K
                    "voxel_coords": voxel_coords,  # K, 3
                    "voxel_features": feats,  # K, ?
                    "scene_name": scene['scene_name']
                }
                return data

        if self.cfg.model.training_stage == 1:
            query_points = scene['query_absolute_points']
            query_indices = scene['query_voxel_indices']
            # Number of data points you have
            M = scene["query_absolute_points"].shape[0]
            # Number of points you want to keep
            num_to_keep = int(M * self.cfg.data.udf_queries.ratio_per_epoch)
            # Randomly select indices
            query_relative_points = query_points[:, np.newaxis] - voxel_center[query_indices] # M, K, 3
            perm_indices = torch.randperm(M)[:num_to_keep]
            query_absolute_points = scene['query_absolute_points'][perm_indices]
            values = scene['values'][perm_indices]
            query_indices = query_indices[perm_indices]
            query_relative_points = query_relative_points[perm_indices]

            data = {
                "xyz": xyz,  # N, 3
                "points": relative_coords,  # N, K , 3
                "point_features": scene['point_features'],  # N, 3
                "labels": scene['labels'],  # N,
                "voxel_indices": inds_reconstruct,  # N, or N, K
                "voxel_coords": voxel_coords,  # K, 3
                "voxel_features": feats,  # K, ?
                "absolute_query_points": query_absolute_points,
                "query_points": query_relative_points,  # M, 3 or M, K, 3
                "query_voxel_indices": query_indices,  # M, or M, K
                "values": values,  # M,
                "scene_name": scene['scene_name']
            }

            return data

