from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
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

    if len(batch) == 1:  # if batch size is 1
        data["points"] = torch.unsqueeze(torch.cat(points, dim=0), 0).to(torch.float32)  # size: (1, N, 3)
        data["colors"] = torch.unsqueeze(torch.cat(colors, dim=0), 0)  # size: (1, N, 3)
        data["labels"] = torch.unsqueeze(torch.cat(labels, dim=0), 0)  # size: (1, N)
        data["query_points"] = torch.unsqueeze(torch.cat(query_points, dim=0), 0).to(torch.float32)  # size: (1, M, 3)
        data["values"] = torch.unsqueeze(torch.cat(values, dim=0), 0)  # size: (1, M)
    else:
        data["points"] = torch.stack(points, dim=0).to(torch.float32)  # size: (B, N, 3)
        data["colors"] = torch.stack(colors, dim=0)  # size: (B, N, 3)
        data["labels"] = torch.stack(labels, dim=0)  # size: (B, N)
        data["query_points"] = torch.stack(query_points, dim=0).to(torch.float32) # size: (B, M, 3)
        data["values"] = torch.stack(values, dim=0)  # size: (B, M)
        data["points"] = data["points"].to(torch.float32)
        data["query_points"] = data["query_points"].to(torch.float32)
        arrgh(data)
    return data 
