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

        if self.world_size > 1: 
            sampler = DistributedInfSampler(dataset, shuffle=flags.shuffle)
        else:
            sampler = InfSampler(dataset, shuffle=flags.shuffle)

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
    locs = []
    rotated_locs = []
    locs_scaled = []
    colors = []
    vert_batch_ids = []
    feats = []
    sem_labels = []
    instance_ids = []
    instance_info = []  # (N, 3)
    instance_num_point = []  # (total_nInst), int
    instance_offsets = [0]
    total_num_inst = 0
    object_lng_class = []
    object_lat_class = []
    object_up_class = []
    R = []
    front_direction = []
    up_direction = []
    instance_cls = []  # (total_nInst), long
    batch_divide = []
    scan_ids = []

    for i, b in enumerate(batch):
        scan_ids.append(b["scan_id"])
        locs.append(torch.from_numpy(b["locs"]))
        rotated_locs.append(torch.from_numpy(b["rotated_locs"]))
        locs_scaled.append(torch.from_numpy(b["locs_scaled"]).int())
        colors.append(torch.from_numpy(b["colors"]))
        vert_batch_ids.append(torch.full((b["locs_scaled"].shape[0],), fill_value=i, dtype=torch.int16))
        batch_divide.append(torch.tensor([b["locs_scaled"].shape[0]]).int())
        feats.append(torch.from_numpy(b["feats"]))
        instance_ids.append(torch.from_numpy(b["instance_ids"]))

        sem_labels.append(torch.from_numpy(b["sem_labels"]))
        instance_ids.append(torch.from_numpy(b["sem_labels"]))

        object_lng_class.append(torch.from_numpy(b["lng_class"]))
        object_lat_class.append(torch.from_numpy(b["lat_class"]))
        object_up_class.append(torch.from_numpy(b["up_class"]))
        front_direction.append(torch.tensor(b["obb"]["front"]))
        up_direction.append(torch.tensor(b["obb"]["up"]))
        R.append(torch.tensor(b["R"]))
        instance_info.append(torch.from_numpy(b["instance_info"]))
        instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
        instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

        instance_cls.extend(b["instance_semantic_cls"])

    tmp_locs_scaled = torch.cat(locs_scaled, dim=0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    data['scan_ids'] = scan_ids
    data["locs"] = torch.cat(locs, dim=0)  # float (N, 3)
    data["rotated_locs"] = torch.cat(rotated_locs, dim=0)  # float (N, 3)
    data["colors"] = torch.cat(colors, dim=0)  # float (N, 3)

    data["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)
    data["feats"] = torch.cat(feats, dim=0)

    data["sem_labels"] = torch.cat(sem_labels, dim=0)  # int (N,)
    data["instance_ids"] = torch.cat(instance_ids, dim=0)  # int, (N,)
    data["instance_info"] = torch.cat(instance_info, dim=0)  # float (total_nInst, 3)
    data["instance_num_point"] = torch.cat(instance_num_point, dim=0)  # (total_nInst)
    data["instance_offsets"] = torch.tensor(instance_offsets, dtype=torch.int32)  # int (B+1)
    data["instance_semantic_cls"] = torch.tensor(instance_cls, dtype=torch.int32)  # long (total_nInst)
    data["lng_class"] = torch.tensor(object_lng_class)
    data["lat_class"] = torch.tensor(object_lat_class)
    data["up_class"] = torch.tensor(object_up_class)
    data["front_direction"] = torch.stack(front_direction)
    data["up_direction"] = torch.stack(up_direction)
    data["R"] = torch.stack(R)

    #batch divide
    data["batch_divide"] = batch_divide
    # voxelize
    data["voxel_locs"], data["v2p_map"], data["p2v_map"] = common_ops.voxelization_idx(tmp_locs_scaled,
                                                                                       data["vert_batch_ids"],
                                                                                       len(batch),
                                                                                       4)
    return data

class InfSampler(Sampler):
  def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
    self.dataset = dataset
    self.shuffle = shuffle
    self.reset_sampler()

  def reset_sampler(self):
    num = len(self.dataset)
    indices = torch.randperm(num) if self.shuffle else torch.arange(num)
    self.indices = indices.tolist()
    self.iter_num = 0

  def __iter__(self):
    return self

  def __next__(self):
    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value

  def __len__(self):
    return len(self.dataset)


class DistributedInfSampler(DistributedSampler):
  def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
    super().__init__(dataset, shuffle=shuffle)
    self.reset_sampler()

  def reset_sampler(self):
    self.indices = list(super().__iter__())
    self.iter_num = 0

  def __iter__(self):
    return self

  def __next__(self):
    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value

