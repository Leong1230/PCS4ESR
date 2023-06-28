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


# class GeneralDataset(Dataset):
#     def __init__(self, cfg, split):
#         self.cfg = cfg
#         self.split = split
#         self.dataset_root_path = cfg.data.dataset_path
#         self.part_ids = cfg.data.valid_part_ids
#         self.file_suffix = cfg.data.file_suffix
#         self.full_scale = cfg.data.full_scale
#         self.scale = cfg.data.scale
#         self.max_num_point = cfg.data.max_num_point
#         self.data_map = {
#             "train": cfg.data.metadata.train_list,
#             "val": cfg.data.metadata.val_list,
#             "test": cfg.data.metadata.test_list
#         }
#         self._load_from_disk()
#         if cfg.model.model.use_multiview:
#             self.multiview_hdf5_file = h5py.File(self.cfg.data.metadata.multiview_file, "r", libver="latest")

#     def _load_from_disk(self):
#         with open(self.data_map[self.split]) as f:
#             self.scene_names = [line.strip() for line in f]
#         self.objects = []
#         self.object_num = 0
#         self.scene_num = 0

#         for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data from disk"):
#             self.scene_num +=1
#             scene_path = os.path.join(self.dataset_root_path, self.split, scene_name + self.file_suffix)
#             scene = torch.load(scene_path)
#             for object in scene["objects"]:
#                 self.object_num +=1
#                 # object["xyz"] -= object["xyz"].mean(axis=0)
#                 object["xyz"] -= object["obb"]["centroid"]
#                 object["rgb"] = object["rgb"].astype(np.float32) / 127.5 - 1
#                 object["scene_id"] = scene_name
#                 # if object["obb"]["up"][2] >= 0:
#                 #     object["class"] = np.array([1])
#                 # else:
#                 #     object["class"] = np.array([0])
#                 if mode(object["sem_labels"]) not in self.cfg.data.ignore_classes:
#                     self.objects.append(object)

#         print("object number in ", self.split, "set: ", self.object_num )
#         print("scene number in ", self.split, "set: ", self.scene_num )

#     def __len__(self):
#         return len(self.objects)

#     def _get_front_direction_class(self, object):
#         x = object["obb"]["front"][0]
#         y = object["obb"]["front"][1]
#         z = object["obb"]["front"][2]
#         lat = math.atan2(z, math.sqrt(x*x+y*y))
#         lng = math.atan2(y, x)
#         lat_class = np.round(self.cfg.data.lat_class*(lat+math.pi/2)/(math.pi))
#         lat_class.astype(np.int)
#         if lat_class == self.cfg.data.lat_class:
#             lat_class = 0
#         lng_class = np.round(self.cfg.data.lng_class*(lng+math.pi)/(2*math.pi))
#         lng_class.astype(np.int)
#         if lng_class == self.cfg.data.lng_class:
#             lng_class = 0
#         return lng_class, lat_class

#     def _get_up_direction_class(self, object):
#         x = object["obb"]["up"][0]
#         y = object["obb"]["up"][1]
#         z = object["obb"]["up"][2]
#         up = math.atan2(y, z)
#         up_class = np.round(self.cfg.data.up_class*(up+math.pi)/(2*math.pi))
#         up_class.astype(np.int)
#         if up_class == self.cfg.data.up_class:
#             up_class = 0
#         return up_class

#     def _get_rotation_matrix(self, object):
#         front = object["obb"]["front"]
#         up = object["obb"]["up"]
#         side = np.cross(up, front)
#         rotated_axis = np.vstack((front, side, up))
#         axis = np.vstack(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
#         R = np.matmul(np.linalg.inv(axis), rotated_axis)
#         return R

#     def _get_augmentation_matrix(self):
#         m = np.eye(3)
#         if self.cfg.data.augmentation.jitter_xyz:
#             m = np.matmul(m, jitter())
#         if self.cfg.data.augmentation.flip:
#             flip_m = flip(0, random=True)
#             m *= flip_m
#         if self.cfg.data.augmentation.rotation:
#             t = np.random.rand() * 2 * np.pi
#             rot_m = rotz(t)
#             m = np.matmul(m, rot_m)  # rotation around z
#         return m.astype(np.float32)

#     def _get_cropped_inst_ids(self, instance_ids, valid_idxs):
#         """
#         Postprocess instance_ids after cropping
#         """
#         instance_ids = instance_ids[valid_idxs]
#         j = 0
#         while j < instance_ids.max():
#             if np.count_nonzero(instance_ids == j) == 0:
#                 instance_ids[instance_ids == instance_ids.max()] = j
#             j += 1
#         return instance_ids

#     def _get_inst_info(self, xyz, instance_ids, sem_labels):
#         """
#         :param xyz: (n, 3)
#         :param instance_ids: (n), int, (0~nInst-1, -1)
#         :return: num_instance, dict
#         """
#         instance_num_point = []  # (nInst), int
#         unique_instance_ids = np.unique(instance_ids)
#         unique_instance_ids = unique_instance_ids[unique_instance_ids != self.cfg.data.ignore_label]
#         num_instance = unique_instance_ids.shape[0]
#         # (n, 3), float, (meanx, meany, meanz)
#         instance_info = np.empty(shape=(xyz.shape[0], 3), dtype=np.float32)
#         instance_cls = np.full(shape=unique_instance_ids.shape[0], fill_value=self.cfg.data.ignore_label, dtype=np.int8)
#         for index, i in enumerate(unique_instance_ids):
#             inst_i_idx = np.where(instance_ids == i)[0]

#             # instance_info
#             xyz_i = xyz[inst_i_idx]

#             mean_xyz_i = xyz_i.mean(0)

#             # offset
#             instance_info[inst_i_idx] = mean_xyz_i

#             # instance_num_point
#             instance_num_point.append(inst_i_idx.size)

#             # semantic label
#             cls_idx = inst_i_idx[0]
#             instance_cls[index] = sem_labels[cls_idx] - len(self.cfg.data.ignore_classes) if sem_labels[cls_idx] != self.cfg.data.ignore_label else sem_labels[cls_idx]
#             # bounding boxes

#         return num_instance, instance_info, instance_num_point, instance_cls

#     def __getitem__(self, idx):
#         object = self.objects[idx]

#         scene_id = object["scene_id"]
#         points = object["xyz"]  # (N, 3)
#         colors = object["rgb"]  # (N, 3)
#         normals = object["normal"]
#         obbs = object["obb"]
#         # lng_class, lat_class = np.array([self._get_front_direction_class(object)]).astype(np.int)
#         lng_class, lat_class = self._get_front_direction_class(object)
#         up_class = self._get_up_direction_class(object)
#         lng_class = np.array([lng_class]).astype(np.int)
#         lat_class = np.array([lat_class]).astype(np.int)
#         up_class = np.array([up_class]).astype(np.int)


#         # get rotation matrix
#         R = self._get_rotation_matrix(object) 

#         # get rotated canonical coordinate
#         rotated_points = np.matmul(points, R)
#         # rotated_points = np.matmul(points, np.linalg.inv(R))

#         if self.cfg.model.model.use_multiview:
#             multiviews = self.multiview_hdf5_file[scene_id]
#         instance_ids = object["instance_ids"]
#         sem_labels = object["sem_labels"]
#         data = {"scan_id": scene_id}

#         # augment
#         if self.split == "train":
#             aug_matrix = self._get_augmentation_matrix()
#             points = np.matmul(points, aug_matrix)
#             normals = np.matmul(normals, np.transpose(np.linalg.inv(aug_matrix)))
#             if self.cfg.data.augmentation.jitter_rgb:
#                 # jitter rgb
#                 colors += np.random.randn(3) * 0.1

#         # scale
#         scaled_points = points * self.scale

#         # elastic
#         if self.split == "train" and self.cfg.data.augmentation.elastic:
#             scaled_points = elastic(scaled_points, 6 * self.scale // 50, 40 * self.scale / 50)
#             scaled_points = elastic(scaled_points, 20 * self.scale // 50, 160 * self.scale / 50)

#         # offset
#         scaled_points -= scaled_points.min(axis=0)

#         # crop
#         if self.split == "train":
#             # HACK, in case there are few points left
#             max_tries = 10
#             valid_idxs_count = 0
#             valid_idxs = np.ones(shape=scaled_points.shape[0], dtype=np.bool)
#             if valid_idxs.shape[0] > self.max_num_point:
#                 while max_tries > 0:
#                     points_tmp, valid_idxs = crop(scaled_points, self.max_num_point, self.full_scale[1])
#                     valid_idxs_count = np.count_nonzero(valid_idxs)
#                     if valid_idxs_count >= 5000:
#                         scaled_points = points_tmp
#                         break
#                     max_tries -= 1
#                 if valid_idxs_count < 5000:
#                     raise Exception("Over-cropped!")

#             scaled_points = scaled_points[valid_idxs]
#             points = points[valid_idxs]
#             normals = normals[valid_idxs]
#             colors = colors[valid_idxs]
#             if self.cfg.model.model.use_multiview:
#                 multiviews = np.asarray(multiviews)[valid_idxs]
#             sem_labels = sem_labels[valid_idxs]
#             instance_ids = self._get_cropped_inst_ids(instance_ids, valid_idxs)

#         num_instance, instance_info, instance_num_point, instance_semantic_cls = self._get_inst_info(
#             points, instance_ids, sem_labels)

#         feats = np.zeros(shape=(len(scaled_points), 0), dtype=np.float32)
#         if self.cfg.model.model.use_color:
#             feats = np.concatenate((feats, colors), axis=1)
#         if self.cfg.model.model.use_normal:
#             feats = np.concatenate((feats, normals), axis=1)
#         if self.cfg.model.model.use_multiview:
#             feats = np.concatenate((feats, multiviews), axis=1)

#         data["locs"] = points  # (N, 3)
#         data["rotated_locs"] = rotated_points # (N, 3)
#         data["locs_scaled"] = scaled_points  # (N, 3)
#         data["colors"] = colors
#         data["feats"] = feats  # (N, 3)
#         data["sem_labels"] = sem_labels  # (N,)
#         data["instance_ids"] = instance_ids  # (N,) 0~total_nInst, -1
#         data["num_instance"] = np.array(num_instance, dtype=np.int32)  # int
#         data["instance_info"] = instance_info  # (N, 12)
#         data["instance_num_point"] = np.array(instance_num_point, dtype=np.int32)  # (num_instance,)
#         data["instance_semantic_cls"] = instance_semantic_cls
#         data["lng_class"] = lng_class
#         data["lat_class"] = lat_class
#         data["up_class"] = up_class
#         data["R"] = R # (, 3)
#         data["obb"] = obbs
#         return data
