import os
import torch
import open3d as o3d
import numpy as np
import pandas as pd
import logging
import trimesh
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from multiscan.utils import io
from plyfile import PlyData, PlyElement
from core import Preprocess
import pdb

log = logging.getLogger(__name__)

class InstanceSegmentation(Preprocess):
    def __init__(self, cfg, scan_dir):
        super().__init__(cfg, scan_dir)

        if self.debug:
            log.setLevel(logging.DEBUG)
    
    def clean_mesh(self):
        mesh = self.construct_o3d_mesh()
        if self.debug:
            o3d.visualization.draw_geometries([mesh])
        
        objects = self.annotations['objects']

        remove_indices = None
        tri_indices = {}
        for obj in objects:
            obj_label = obj['label']
            obj_id = obj['objectId']
            if obj_label.lower().startswith('remove'):
                remove_indices = self.object_triangles(obj_id)
            else:
                tri_indices[obj_id] = self.object_triangles(obj_id)
        
        if remove_indices is not None:
            remove_indices = np.sort(remove_indices)
            mesh.remove_triangles_by_index(remove_indices.tolist())
            mesh.remove_unreferenced_vertices()
        
            offset = 0
            for idx in remove_indices:
                for key, val in tri_indices.items():
                    val[val > (idx - offset)] -= 1
                offset += 1
        return mesh, tri_indices

    @staticmethod
    def triangle_idx_to_vertex_idx(mesh, tri_indicies):
        face_data = np.asarray(mesh.triangles)
        selected_triangles = face_data[tri_indicies]
        vertex_indices = np.unique(selected_triangles.ravel())
        return vertex_indices

    def process(self, output_dir='output'):
        try:
            super().process(output_dir)
        except Exception as e:
            log.error(f'Error: input scan dir {self.input_dir} does not match pattern scene_xxxxx_xx')
            return

        mesh, tri_indices = self.clean_mesh()
        face_data = np.asarray(mesh.triangles)
        sem_labels = np.full(shape=(np.asarray(mesh.vertices).shape[0]), fill_value=-1, dtype=np.int32)
        instance_ids = np.full(shape=(sem_labels.shape[0]), fill_value=-1, dtype=np.int32)
        objects = self.annotations['objects']
        object_output = []
        # output data
        xyz = np.ascontiguousarray(np.asarray(mesh.vertices))
        rgb = np.ascontiguousarray(np.asarray(mesh.vertex_colors)) * 255.0
        normal = np.ascontiguousarray(np.asarray(mesh.vertex_normals))
        faces = np.ascontiguousarray(np.asarray(mesh.triangles))

        for inst_count, obj in enumerate(objects):
            object = {}
            obj_label = obj['label']
            if obj_label.lower().startswith('remove'):
                continue
            obj_id = obj['objectId']
            obj_tri_indices = tri_indices[obj_id]
            selected_tri_indices = face_data[obj_tri_indices]
            unique_vertex_indices = np.unique(selected_tri_indices.ravel())

            obj_semantic_id = self.object_semantic_id(obj_label)
            sem_labels[unique_vertex_indices] = obj_semantic_id - 1 if obj_semantic_id > 0 else -1

            vertex_indicies = self.triangle_idx_to_vertex_idx(mesh, obj_tri_indices)
            if obj_semantic_id in [1, 2, 3]:
                object["instance_id"] = -1
                instance_ids[vertex_indicies] = -1
            else:
                object["instance_id"] = inst_count
                instance_ids[vertex_indicies] = inst_count
            object["xyz"] = xyz[vertex_indicies]
            object["rgb"] = rgb[vertex_indicies]
            object["normal"] = normal[vertex_indicies]
            object["obb"] = obj["obb"]
            object["instance_ids"] = (instance_ids[vertex_indicies]).astype(np.int32)
            object["sem_labels"] = (sem_labels[vertex_indicies]).astype(np.int32)
            object_output.append(object)
        # separate train/val/test cases
        # torch.save({"xyz": xyz.astype(np.float32), "rgb": rgb.astype(np.float32), "normal": normal.astype(np.float32), 'faces': faces,
        #                 "sem_labels": sem_labels.astype(np.int32), "instance_ids": instance_ids.astype(np.int32)},
        #                 os.path.join(self.output_dir, f'{self.scan_id}.pth'))
        torch.save({"objects":object_output},
                        os.path.join(self.output_dir, f'{self.scan_id}.pth'))

        # make semantic_label_idxs configurable
        instance_label_new = np.zeros(instance_ids.shape, dtype=np.int32)
        instance_num = int(instance_ids.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_ids == inst_id)[0]
            if instance_mask.size == 0:
                continue
            sem_id = int(sem_labels[instance_mask[0]])
            if(sem_id == -1): sem_id = 0
            instance_label_new[instance_mask] = sem_id * 1000 + inst_id + 1

        np.savetxt(os.path.join(self.output_dir, self.scan_id + '.txt'), instance_label_new, fmt='%d')


@hydra.main(config_path="../configs", config_name="multiscan", version_base="1.2")
def main(cfg : DictConfig):
    scan_dirs = io.get_folder_list(os.path.abspath(cfg.input_dir), join_path=True)
    scans_split = pd.read_csv(os.path.join(cfg.csv_dir, cfg.scans_split_csv))

    splits = ['train', 'val', 'test']
    for split in splits:
        scans = scans_split[scans_split['split'] == split]
        for i, row in tqdm(scans.iterrows(), total=scans.shape[0]):
            scan_id = row['scanId']
            scan_dir = os.path.join(os.path.abspath(cfg.input_dir), scan_id)
            # assert scan_dir in scan_dirs
            
            inst_segm = InstanceSegmentation(cfg, scan_dir)
            inst_segm.process(os.path.join(os.path.abspath(cfg.output_dir), split))

if __name__ == "__main__":
    main()