from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import torch
from torch.utils.data import Sampler, DistributedSampler, Dataset
import pytorch_lightning as pl
from arrgh import arrgh


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.dataset = getattr(import_module('hybridpc.data.dataset'), data_cfg.data.dataset)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(self.data_cfg, "train")
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "test" or stage is None:
            self.val_set = self.dataset(self.data_cfg, self.data_cfg.model.inference.split)
        if stage == "predict" or stage is None:
                self.test_set = self.dataset(self.data_cfg, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.data_cfg.data.batch_size, shuffle=True, pin_memory=True,
                          collate_fn=_sparse_collate_fn, num_workers=self.data_cfg.data.num_workers, drop_last=True)

    def val_dataloader(self):          
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=_sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=_sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, pin_memory=True, collate_fn=_sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)


def _sparse_collate_fn(batch):
    if "gt_geometry" in batch[0]:
        """ for dataset with ground truth geometry """
        data = {}
        xyz = []
        un_splats_xyz = []
        normals = []
        indices = []
        point_features = []
        relative_coords = []
        gt_geometry_list = []
        scene_names_list = []
        voxel_nums_list = []
        voxel_coords_list = []
        voxel_features_list = []
        cumulative_voxel_coords_len = 0  # Keep track of the cumulative length


        for i, b in enumerate(batch):
            voxel_nums_list.append(b["voxel_coords"].shape[0])
            voxel_coords_list.append(b["voxel_coords"])
            voxel_features_list.append(b["voxel_feats"])
            relative_coords.append(torch.from_numpy(b["relative_coords"]))
            scene_names_list.append(b["scene_name"])
            xyz.append(torch.from_numpy(b["xyz"]))
            un_splats_xyz.append(torch.from_numpy(b["un_splats_xyz"]))
            normals.append(torch.from_numpy(b["normals"]))
            point_features.append(torch.from_numpy(b["point_features"]))
            gt_geometry_list.append(b["gt_geometry"])
            indices.append(torch.from_numpy(b["indices"] + 
            cumulative_voxel_coords_len))

            cumulative_voxel_coords_len += len(b["voxel_coords"])

        data['xyz'] = torch.cat(xyz, dim=0)
        data['un_splats_xyz'] = torch.cat(un_splats_xyz, dim=0)
        data['normals'] = torch.cat(normals, dim=0)
        data['point_features'] = torch.cat(point_features, dim=0)
        data['scene_names'] = scene_names_list
        # data['xyz_splits'] = [c.shape[0] for c in xyz]
        data['xyz_splits'] = torch.tensor([c.shape[0] for c in xyz])
        data['gt_geometry'] = gt_geometry_list

        data['indices'] = torch.cat(indices, dim=0)
        data['relative_coords'] = torch.cat(relative_coords, dim=0)
        data["voxel_coords"], data["voxel_features"] = ME.utils.sparse_collate(
            coords=voxel_coords_list, feats=voxel_features_list
        )

        return data

    data = {}
    relative_coords = []
    xyz = []
    un_splats_xyz = []
    normals = []
    all_xyz = []
    all_normals = []
    point_features = []
    indices = []
    query_relative_coords = []
    voxel_coords_list = []
    voxel_features_list = []
    voxel_coords = []
    batch_ids = []
    scene_names_list = []
    gt_geometry_list = []

    cumulative_voxel_coords_len = 0  # Keep track of the cumulative length
    voxel_nums_list = []

    for i, b in enumerate(batch):
        scene_names_list.append(b["scene_name"])
        voxel_nums_list.append(b["voxel_coords"].shape[0])
        voxel_coords_list.append(b["voxel_coords"])
        voxel_features_list.append(b["voxel_feats"])
        relative_coords.append(torch.from_numpy(b["relative_coords"]))
        xyz.append(torch.from_numpy(b["xyz"]))
        un_splats_xyz.append(torch.from_numpy(b["un_splats_xyz"]))
        normals.append(torch.from_numpy(b["normals"]))
        all_xyz.append(torch.from_numpy(b["all_xyz"]))
        all_normals.append(torch.from_numpy(b["all_normals"]))
        point_features.append(torch.from_numpy(b["point_features"]))
        indices.append(torch.from_numpy(b["indices"] + cumulative_voxel_coords_len))

        # Update the cumulative length for the next iteration
        cumulative_voxel_coords_len += len(b["voxel_coords"])

    data['all_xyz'] = torch.cat(all_xyz, dim=0)
    data['all_normals'] = torch.cat(all_normals, dim=0)
    data['xyz'] = torch.cat(xyz, dim=0)
    data['un_splats_xyz'] = torch.cat(un_splats_xyz, dim=0)
    data['normals'] = torch.cat(normals, dim=0)
    data['relative_coords'] = torch.cat(relative_coords, dim=0)
    data['point_features'] = torch.cat(point_features, dim=0)
    data['indices'] = torch.cat(indices, dim=0)
    data["voxel_coords"], data["voxel_features"] = ME.utils.sparse_collate(
        coords=voxel_coords_list, feats=voxel_features_list
    ) # size: (N, 4)
    data['scene_names'] = scene_names_list
    data['voxel_nums'] = voxel_nums_list
    data['row_splits'] = [c.shape[0] for c in all_xyz]
    data['xyz_splits'] = torch.tensor([c.shape[0] for c in xyz])

    return data



