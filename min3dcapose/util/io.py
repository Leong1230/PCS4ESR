import os
import torch
import struct
import open3d as o3d
import numpy as np
from tqdm import tqdm
from min3dcapose.evaluation.gravity_aligned_obb import gravity_aligned_mobb
from min3dcapose.evaluation.visualization import get_directions, o3d_render


def save_prediction(save_path, all_pred_obbs, class_names):
    inst_pred_path = os.path.join(save_path, "instance")
    inst_pred_directions_path = os.path.join(inst_pred_path, "predicted_directions")
    obb_output = {}
    os.makedirs(inst_pred_directions_path, exist_ok=True)
    scan_instance_count = {}
    for pred in tqdm(all_pred_obbs, desc="==> Saving predictions ..."):
        scan_id = pred["scan_id"]
        obb_output["scan_id"].append(pred)
    for scan_id, pred_list in obb_output:
        torch.save({"pred_obbs":pred_list}, os.path.join(inst_pred_directions_path, f'{scan_id}.pth'))

def save_gt(save_path, all_gt_obbs, class_names):
    inst_pred_path = os.path.join(save_path, "instance")
    inst_pred_directions_path = os.path.join(inst_pred_path, "gt_directions")
    obb_output = {}
    os.makedirs(inst_pred_directions_path, exist_ok=True)
    scan_instance_count = {}
    for gt in tqdm(all_gt_obbs, desc="==> Saving ground truths ..."):
        scan_id = gt["scan_id"]
        obb_output["scan_id"].append(gt)
    for scan_id, pred_list in obb_output:
        torch.save({"gt_obbs":pred_list}, os.path.join(inst_pred_directions_path, f'{scan_id}.pth'))
