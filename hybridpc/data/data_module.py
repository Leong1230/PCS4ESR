from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Sampler, DistributedSampler, Dataset
import pytorch_lightning as pl
from hybridpc.common_ops.functions import common_ops


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
                          collate_fn=sparse_collate_fn, num_workers=self.data_cfg.data.num_workers)

    def val_dataloader(self):          
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, pin_memory=True, collate_fn=sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)


def sparse_collate_fn(batch):
    data = {}

    # Batch together points, colors, labels, query_points, values
    points = []
    colors = []
    labels = []
    query_points = []
    values = []
    for i, b in enumerate(batch):
        points.append(torch.from_numpy(b["points"]))
        colors.append(torch.from_numpy(b["colors"]))
        labels.append(torch.from_numpy(b["labels"]))
        query_points.append(torch.from_numpy(b["query_points"]))
        values.append(torch.from_numpy(b["values"]))
    
    data["points"] = torch.cat(points, dim=0)  # size: (N, 3)
    data["colors"] = torch.cat(colors, dim=0)  # size: (N, 3)
    data["labels"] = torch.cat(labels, dim=0)  # size: (N,)
    data["query_points"] = torch.cat(query_points, dim=0)  # size: (M, 3)
    data["values"] = torch.cat(values, dim=0)  # size: (M,)
    
    return data
