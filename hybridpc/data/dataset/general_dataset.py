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
        # self.split = split
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.num_point = cfg.data.num_point
        self.rotate_num = cfg.data.rotate_num
        self.take = cfg.data.take
        self.loops = cfg.data.augmentation.loops
        self.intake_start = cfg.data.intake_start
        self.use_relative = cfg.data.use_relative
        self.max_num_point = cfg.data.max_num_point
        self.k_neighbors = cfg.model.network.k_neighbors # 1 for no interpolation
        self.sample_entire_scene = cfg.data.udf_queries.sample_entire_scene
        self.num_queries_on_surface = cfg.data.udf_queries.num_queries_on_surface
        self.queries_stds = cfg.data.udf_queries.queries_stds
        # self.num_queries_per_std = cfg.data.udf_queries.num_queries_per_std
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
            if self.cfg.data.augmentation.method == 'original':
                input_transforms = [
                    t.RandomDropout(0.2),
                    t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                    # t.ChromaticAutoContrast(),
                    t.ChromaticTranslation(cfg.data.augmentation.color_trans_ratio),
                    t.ChromaticJitter(cfg.data.augmentation.color_jitter_std),
                    t.HueSaturationTranslation(
                        cfg.data.augmentation.hue_max, cfg.data.augmentation.saturation_max),
                ]
            else:
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


        if cfg.data.augmentation.use_aug and cfg.data.augmentation.method == 'N-times' and self.split == 'train':
            N = cfg.data.augmentation.loops
        else:
            N = 1

        for idx, sample in tqdm(enumerate(self.scenes_in), desc="Voxelizing and Sample points", ncols=80):
            for i in range(N):  # This loop will run N times if conditions are satisfied, otherwise just once
                processed_sample = self.preprocess_sample_entire_scene(sample, i)
                self.scenes.append(processed_sample)




        # self.label_map = self.get_semantic_mapping_file(cfg.data.metadata.combine_file)
        # self.filelist = os.path.join(self.dataset_root_path, self.data_map[self.split]) # train.txt, val.txt, test.txt
        # self.filenames, self.labels = self.load_filenames()

        # if self.cfg.data.over_fitting:
        #     # self.random_idx = random.randint(0, len(self.filenames) - 1)
        #     self.random_idx = 0
        
        # if self.in_memory:
        #     print('Load ' + self.split + ' dataset into memory')
        #     self.samples = [self.read_file(os.path.join(self.dataset_root_path, self.dataset_split, f))
        #                     for f in tqdm(self.filenames, ncols=80, leave=False)]    
        #     self.data = []
        #     for sample in tqdm(self.samples, desc="Voxelizing and Sample points", ncols=80):
        #         if self.sample_entire_scene:
        #             processed_sample = self.preprocess_sample_entire_scene(sample)
        #         else:
        #             processed_sample = self.preprocess_sample(sample)
        #         self.data.append(processed_sample)

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
        # Convert to tensor
        points_tensor = torch.tensor(np.asarray(xyz))

        # Calculate number of queries based on the ratio and the number of points
        num_queries_on_surface = int(len(xyz) * self.ratio_on_surface + 1)
        num_queries_per_std = [int(len(xyz) * self.ratio_per_std + 1)] * 4  # A list of 4 equal values
        min_range = np.min(xyz, axis=0)
        max_range = np.max(xyz, axis=0)

        query_points, values = compute_udf_from_pcd(
            points_tensor,
            num_queries_on_surface,
            self.queries_stds,
            num_queries_per_std,
            (torch.tensor(min_range), torch.tensor(max_range))
        ) # torch.tensor

        # # compute query indices and relative coordinates
        # if self.cfg.model.network.k_neighbors == 1: # query the nearest voxel
        #     voxel_coords = torch.from_numpy(voxel_coords).int()
        #     # coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        #     voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 # compute voxel_center in orginal coordinate system (torch.tensor)
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     query_points = query_points.to(device)
        #     voxel_center = voxel_center.to(device)
        #     query_indices, _, _ = knn(query_points, voxel_center, 1)
        #     query_indices = query_indices[:, 0]
        #     relative_coords = xyz - voxel_center[inds_reconstruct].cpu().numpy()
        #     query_relative_coords = query_points.cpu().numpy() - voxel_center[query_indices].cpu().numpy()

        #     # remove query_points outside the voxel
        #     lower_bound = -self.voxel_size / 2
        #     upper_bound = self.voxel_size / 2
        #     # Create a mask
        #     mask = (query_relative_coords >= lower_bound) & (query_relative_coords <= upper_bound)
        #     # Reduce across the last dimension to get a (N, ) mask
        #     mask = np.all(mask,-1)
        #     query_indices = query_indices[mask]
        #     query_relative_coords = query_relative_coords[mask]
        #     unmasked_values = values.cpu().numpy()
        #     values = values[mask].cpu().numpy()

        # else: # query multiple voxels
        voxel_coords = torch.from_numpy(voxel_coords).int()
        voxel_center = voxel_coords * self.voxel_size + self.voxel_size / 2.0 # compute voxel_center in orginal coordinate system (torch.tensor)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_points = query_points.to(device)
        voxel_center = voxel_center.to(device)
        query_indices, _, _ = knn(query_points, voxel_center, self.cfg.model.network.k_neighbors)
        inds_reconstruct, _, _ = knn(points_tensor.to(device), voxel_center, self.cfg.model.network.k_neighbors)
        relative_coords = xyz[:, np.newaxis] - voxel_center[inds_reconstruct].cpu().numpy() # N, K, 3
        query_relative_coords = query_points.cpu().numpy()[:, np.newaxis] - voxel_center[query_indices].cpu().numpy() #M, K, 3
        
        # remove query_points outside the voxel
        lower_bound = -self.voxel_size / 2
        upper_bound = self.voxel_size / 2
        # Create a mask
        mask = (query_relative_coords[:, 0, :] >= lower_bound) & (query_relative_coords[:, 0, :] <= upper_bound)
        # Reduce across the last dimension to get a (N, ) mask
        mask = np.all(mask,-1)
        query_indices = query_indices[mask]
        query_relative_coords = query_relative_coords[mask]
        unmasked_values = values.cpu().numpy()
        values = values[mask].cpu().numpy()
        inds_reconstruct = inds_reconstruct.cpu().numpy()




        data = {
            "xyz": xyz,  # N, 3
            "points": relative_coords,  # N, 3 or N, K, 3
            "point_features": point_features,  # N, 3
            "features": feats,  # N, 3
            "labels": sample['sem_labels'],  # N,
            "voxel_coords": voxel_coords.cpu().numpy(),
            "voxel_indices": inds_reconstruct,  # N, or N, K
            "absolute_query_points": query_points.cpu().numpy(),
            "query_points": query_relative_coords,  # M, 3 or M, K , 3
            "query_voxel_indices": query_indices.cpu().numpy(),  # M, or M, K
            "values": values,  # M,
            "unmasked_values": unmasked_values,
            "scene_name": f"{sample['scene_name']}_{i}"
        }

        # voxel_id = 10
        # # voxel_center = data['voxel_coords'][:, 1:4][voxel_id] * self.voxel_size - self.voxel_size
        # relative_points = (data['points'][data['voxel_indices'][:, 0] == voxel_id])[:, 0, :]
        # # absolute_points = relative_points + voxel_center.cpu().numpy()
        # # Append the absolute coordinates to the all_points list
        # # all_original_points.append(absolute_points)
        
        # query_relative_points = (data['query_points'][data['query_voxel_indices'][:, 0] == voxel_id])[:, 0, :]
        # # query_absolute_points = query_relative_points + voxel_center.cpu().numpy()
        # # all_query_points.append(query_absolute_points)




        # # all_points_np = np.vstack(all_original_points)
        # # all_dense_points_np = np.vstack(all_dense_points)
        # # all_query_points_np = np.vstack(all_query_points)
        # # Use open3d to visualize the point cloud
        # points_cloud = o3d.geometry.PointCloud()
        # points_cloud.points = o3d.utility.Vector3dVector(relative_points)
        # query_points_cloud = o3d.geometry.PointCloud()
        # query_points_cloud.points = o3d.utility.Vector3dVector(query_relative_points)
    
        # red_color = [1, 0, 0]
        # blue_color = [0, 0, 1]
        # points_cloud.paint_uniform_color(red_color)
        # query_points_cloud.paint_uniform_color(blue_color)

        # # Merge the point clouds
        # merged_pcd = points_cloud + query_points_cloud
        # original_points_cloud = merged_pcd

        # o3d.visualization.draw_geometries([original_points_cloud])

        # # computing voxel center coordinates

        # voxel_centers = self.voxel_size * voxel_coords[data['voxel_indices']] + self.voxel_size / 2

        # # recovering the absolute coordinates of points
        # absolute_points = relative_coords + voxel_centers.cpu().numpy()

        # # recover query points
        # query_voxel_indices = data["query_voxel_indices"]
        # query_points = data["query_points"]
        # query_voxel_centers = self.voxel_size * voxel_coords[query_voxel_indices] + self.voxel_size / 2
        # query_absolute_points = query_points + query_voxel_centers.cpu().numpy()
        # surface_queries = query_absolute_points[mask_surface[mask]]
        # gaussian_queries = query_absolute_points[mask_gaussian[mask]]
        # uniform_queries = query_absolute_points[mask_uniform[mask]]

        # # Create Open3D point cloud for points
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(absolute_points)
        # pcd.colors = o3d.utility.Vector3dVector(np.ones_like(absolute_points) * [1, 0, 0])  # red

        # # Create Open3D point cloud for query points
        # surface_query_pcd = o3d.geometry.PointCloud()
        # surface_query_pcd.points = o3d.utility.Vector3dVector(surface_queries)
        # surface_query_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(surface_queries) * [0, 1, 0])  # green
        # # Create Open3D point cloud for query points
        # gaussian_query_pcd = o3d.geometry.PointCloud()
        # gaussian_query_pcd.points = o3d.utility.Vector3dVector(gaussian_queries)
        # gaussian_query_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(gaussian_queries) * [0, 1, 0])
        # # Create Open3D point cloud for query points
        # uniform_query_pcd = o3d.geometry.PointCloud()
        # uniform_query_pcd.points = o3d.utility.Vector3dVector(uniform_queries)
        # uniform_query_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(uniform_queries) * [0, 1, 0])

        # # visualize the point clouds
        # # merged_points_cloud = pcd + query_pcd
        # # o3d.visualization.draw_geometries([pcd, query_pcd])

        # # Save the point clouds
        # save_dir = os.path.join(self.cfg.exp_output_root_path)
        # o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.cfg.data.voxel_size) + '_original.ply'), pcd)
        # o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.cfg.data.voxel_size) + '_surface.ply'), surface_query_pcd)
        # o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.cfg.data.voxel_size) + '_gaussian.ply'), gaussian_query_pcd)
        # o3d.io.write_point_cloud(os.path.join(save_dir, 'voxel_' + str(self.cfg.data.voxel_size) + '_uniform.ply'), uniform_query_pcd)


        return data
    
    def get_semantic_mapping_file(file_path):
        label_mapping = {}
        with open(file_path, "r") as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file)  # skip the header
            for line in tsv_file:
                label_mapping[line[1]] = int(line[4])  # use nyu40 label set
        return label_mapping


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
            segmantic_labels = vtx['label']
            for label, segs in obj_name_to_segs.items():
                for seg in segs:
                    verts = seg_to_verts[seg]
                    if label not in label_map or label_map[label] not in filtered_label_map:
                        semantic_label = -1
                    else:
                        semantic_label = filtered_label_map[label_map[label]]
                    semantic_labels[verts] = semantic_label
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

        return filenames[self.intake_start:self.take+self.intake_start], labels[self.intake_start:self.take+self.intake_start]
  

    def __len__(self):
        return len(self.scenes)

    # def __getitem__(self, idx):

    #     return self.data[idx]
    
    def __getitem__(self, idx):

        # index = index_long % len(self.data_paths)
        # if self.use_shm:
        #     locs_in = SA.attach("shm://%s_%s_%06d_locs_%08d" %
        #                         (self.dataset_name, self.split, self.identifier, index)).copy()
        #     feats_in = SA.attach("shm://%s_%s_%06d_feats_%08d" %
        #                          (self.dataset_name, self.split, self.identifier, index)).copy()
        #     labels_in = SA.attach("shm://%s_%s_%06d_labels_%08d" %
        #                           (self.dataset_name, self.split, self.identifier, index)).copy()
        # else:
            # locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            # labels_in[labels_in == -100] = 255
            # labels_in = labels_in.astype(np.uint8)
            # # no color in the input point cloud, e.g nuscenes
            # if np.isscalar(feats_in) and feats_in == 0:
            #     feats_in = np.zeros_like(locs_in)
            # feats_in = (feats_in + 1.) * 127.5
        scene = self.scenes[idx]
        locs_in = scene['xyz']
        point_features = scene['point_features']

        voxel_coords_in = scene['voxel_coords']
        feats_in = scene['features']
        labels_in = scene['labels']
        inds_reconstruct  = scene['voxel_indices']

        if self.cfg.data.augmentation.method=='N-times':
            if self.split == "train" and self.cfg.data.augmentation.use_aug:
                voxel_coords, feats, labels = self.input_transforms(voxel_coords_in, feats_in, labels_in)
            else:
                voxel_coords = voxel_coords_in
                feats = feats_in
                labels = labels_in

        if self.cfg.data.augmentation.method=='original':
            if self.split == "train" and self.cfg.data.augmentation.use_aug:
                locs = self.prevoxel_transforms(locs_in)
                locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(locs, point_features, labels_in)
                voxel_coords, feats, labels = self.input_transforms(locs, feats, labels)
            else:
                voxel_coords, feats, labels, inds_reconstruct = self.voxelizer.voxelize(locs_in, point_features, labels_in)

        # Number of data points you have
        M = scene["query_points"].shape[0]
        # Number of points you want to keep
        num_to_keep = int(M * self.cfg.data.udf_queries.ratio_per_epoch)
        # Randomly select indices
        perm_indices = torch.randperm(M)[:num_to_keep]

        data = {
            "xyz": scene['xyz'],  # N, 3
            "points": scene['points'],  # N, 3 or N, K , 3
            "point_features": scene['point_features'],  # N, 3
            "labels": scene['labels'],  # N,
            "voxel_indices": inds_reconstruct,  # N, or N, K
            "voxel_coords": voxel_coords,  # K, 3
            "voxel_features": feats,  # K, ?
            "absolute_query_points": scene['absolute_query_points'][perm_indices],
            "query_points": scene['query_points'][perm_indices],  # M, 3 or M, K, 3
            "query_voxel_indices": scene['query_voxel_indices'][perm_indices],  # M, or M, K
            "values": scene['values'][perm_indices],  # M,
            "unmasked_values": scene['unmasked_values'],  # M,
            "scene_name": scene['scene_name']
        }

        return data

