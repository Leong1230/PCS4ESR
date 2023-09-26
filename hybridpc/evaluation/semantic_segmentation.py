import torch


def evaluate_semantic_accuracy(pred, gt, ignore_label):
    valid_idx = gt != ignore_label
    valid_pred = pred[valid_idx]
    valid_gt = gt[valid_idx]
    if len(valid_gt) > 0:
        return torch.count_nonzero(valid_gt == valid_pred).item() / len(valid_gt) * 100
    else:
        return 0.0  # Return 0.0 if there is no valid ground truth data


def evaluate_semantic_miou(pred, gt, ignore_label):
    valid_idx = gt != ignore_label
    valid_pred = pred[valid_idx]
    valid_gt = gt[valid_idx]
    unique_valid_gt = torch.unique(valid_gt)
    ious = torch.empty(size=(unique_valid_gt.shape[0], ), dtype=torch.float32, device=gt.device)
    for i, gt_id in enumerate(unique_valid_gt):
        intersection = torch.count_nonzero(((valid_gt == gt_id) & (valid_pred == gt_id)))
        union = torch.count_nonzero(((valid_gt == gt_id) | (valid_pred == gt_id)))
        ious[i] = intersection / union if union != 0 else 0.0  # Avoid division by zero
    return ious.mean().item() * 100 if len(ious) > 0 else 0.0  # Return 0.0 if there is no valid ground truth data
