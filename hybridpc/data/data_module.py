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
                          collate_fn=_sparse_collate_fn, num_workers=self.data_cfg.data.num_workers)

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
    data = {}
    points = []
    colors = []
    labels = []
    voxel_indices = []
    query_points = []
    values = []
    query_voxel_indices = []
    voxel_coords_list = []
    voxel_coords = []
    batch_ids = []

    cumulative_voxel_coords_len = 0  # Keep track of the cumulative length

    for i, b in enumerate(batch):
        voxel_coords_list.append(b["voxel_coords"])
        points.append(torch.from_numpy(b["points"]))
        colors.append(torch.from_numpy(b["colors"]))
        labels.append(torch.from_numpy(b["labels"]))
        voxel_indices.append(b["voxel_indices"] + cumulative_voxel_coords_len)
        query_points.append(torch.from_numpy(b["query_points"]))
        values.append(torch.from_numpy(b["values"]))
        query_voxel_indices.append(b["query_voxel_indices"] + cumulative_voxel_coords_len)

        # Create a batch ID for each point and query point in the batch
        batch_ids.append(torch.full((b["points"].shape[0],), fill_value=i, dtype=torch.int32))

        # Update the cumulative length for the next iteration
        cumulative_voxel_coords_len += len(b["voxel_coords"])

    data['points'] = torch.cat(points, dim=0)
    data['colors'] = torch.cat(colors, dim=0)
    data['labels'] = torch.cat(labels, dim=0).long()
    data['voxel_indices'] = torch.cat(voxel_indices, dim=0)
    data['query_points'] = torch.cat(query_points, dim=0)
    data['values'] = torch.cat(values, dim=0)
    data['query_voxel_indices'] = torch.cat(query_voxel_indices, dim=0)
    fake_features = [np.zeros_like(arr) for arr in voxel_coords_list]
    data["voxel_coords"], data["fake_features"] = ME.utils.sparse_collate(voxel_coords_list, fake_features) # size: (N, 4)
    data['batch_ids'] = torch.cat(batch_ids, dim=0)

    return data


