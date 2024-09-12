import os
from pathlib import Path
from torch.utils.data import Dataset

import numpy as np
from hybridpc.data.dataset.voxelizer import Voxelizer

from hybridpc.utils.transform import ComposedTransforms
from hybridpc.data.dataset.carla_gt_geometry import get_class
from hybridpc.data.dataset.general_dataset import DatasetSpec as DS
from hybridpc.data.dataset.general_dataset import RandomSafeDataset

from pycg import exp


class Carla(RandomSafeDataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

        self.skip_on_error = False
        self.custom_name = "carla"
        self.cfg = cfg
        self.voxel_size = cfg.data.voxel_size

        self.split = 'val' if cfg.data.over_fitting else split # use only train set for overfitting
        split = self.split
        self.intake_start = cfg.data.intake_start
        self.take = cfg.data.take
        self.input_splats = cfg.data.input_splats

        self.gt_type = cfg.data.supervision.gt_type

        self.transforms = ComposedTransforms([cfg.data.transforms])
        self.transforms = None
        self.use_dummy_gt = False
        self.voxelizer = Voxelizer(
            voxel_size=self.voxel_size,
            clip_bound=None,
            use_augmentation=False,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)
        
        # If drives not specified, use all sub-folders
        base_path = Path(cfg.data.base_path)
        drives = cfg.data.drives
        if drives is None:
            drives = os.listdir(base_path)
            drives = [c for c in drives if (base_path / c).is_dir()]
        self.drives = drives
        self.input_path = cfg.data.input_path

        # Get all items
        self.all_items = []
        self.drive_base_paths = {}
        for c in drives:
            self.drive_base_paths[c] = base_path / c
            split_file = self.drive_base_paths[c] / (split + '.lst')
            with split_file.open('r') as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.all_items += [{'drive': c, 'item': m} for m in models_c]

        if self.cfg.data.over_fitting:
            self.all_items = self.all_items[self.intake_start:self.take+self.intake_start]



    def __len__(self):
        return len(self.all_items)

    def get_name(self):
        return f"{self.custom_name}-cat{len(self.drives)}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        # self.num_input_points = 50000
        drive_name = self.all_items[data_id]['drive']
        item_name = self.all_items[data_id]['item']

        named_data = {}

        try:
            if self.input_path is None:
                input_data = np.load(self.drive_base_paths[drive_name] / item_name / 'pointcloud.npz')
            else:
                input_data = np.load(Path(self.input_path) / drive_name / item_name / 'pointcloud.npz')
        except FileNotFoundError:
            exp.logger.warning(f"File not found for AV dataset for {item_name}")
            raise ConnectionAbortedError

        named_data[DS.SHAPE_NAME] = "/".join([drive_name, item_name])
        named_data[DS.INPUT_PC]= input_data['points'].astype(np.float32)
        named_data[DS.TARGET_NORMAL] = input_data['normals'].astype(np.float32)

        geom_cls = get_class(self.gt_type)
        named_data[DS.GT_GEOMETRY] = geom_cls.load(self.drive_base_paths[drive_name] / item_name / "groundtruth.bin")

        if self.transforms is not None:
            named_data = self.transforms(named_data, rng)

        point_features = np.zeros(shape=(len(named_data[DS.INPUT_PC]), 0), dtype=np.float32)
        if self.cfg.model.network.use_normal:
            point_features = np.concatenate((point_features, named_data[DS.TARGET_NORMAL]), axis=1)
        if self.cfg.model.network.use_xyz:
            point_features = np.concatenate((point_features, named_data[DS.INPUT_PC]), axis=1)  # add xyz to point features

        xyz = named_data[DS.INPUT_PC]
        normals = named_data[DS.TARGET_NORMAL]
        labels = np.zeros(len(xyz), dtype=np.int32)
        num_points = xyz.shape[0]
        print(f"num_points: {num_points}")
        # sample_indices = np.random.choice(num_points, self.num_input_points, replace=False)
    
        # xyz = xyz[sample_indices]
        # point_features = point_features[sample_indices]
        # labels = labels[sample_indices]
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
            "gt_geometry": named_data[DS.GT_GEOMETRY],
            "xyz": xyz,  # N, 3
            "un_splats_xyz": un_splats_xyz,  # N,
            "normals": normals,  # N, 3
            "scene_name": named_data[DS.SHAPE_NAME],
            "point_features": point_features,  # N, K
            "relative_coords": relative_coords,  # N, 3
            "indices": indices,  # N,
            "voxel_coords": voxel_coords,  # K, 3
            "voxel_feats": voxel_feats,  # K, 3
        }

        return data
