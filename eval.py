import os
import hydra
import torch
from tqdm import tqdm
from hybridpc.evaluation.obb_prediction import GeneralDatasetEvaluator
from hybridpc.util.io import read_gt_files_from_disk, read_pred_files_from_disk


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    split = cfg.model.inference.split
    pred_file_path = os.path.join(cfg.exp_output_root_path, cfg.data.dataset,
                               cfg.model.model.module, cfg.model.model.experiment_name,
                               "inference", cfg.model.inference.split,  "instance", "predicted_directions")
    gt_file_path = os.path.join(cfg.exp_output_root_path, cfg.data.dataset,
                               cfg.model.model.module, cfg.model.model.experiment_name,
                               "inference", cfg.model.inference.split,  "instance", "gt_directions")                            

    if not os.path.exists(pred_file_path):
        print("Error: prediction files do not exist.")
        exit(-1)
    if not os.path.exists(gt_file_path):
        print("Error: ground_truth files do not exist.")
        exit(-1)

    print(f"==> start evaluating {split} set ...")

    print("==> Evaluating instance segmentation ...")
    inst_seg_evaluator = GeneralDatasetEvaluator(cfg.data.class_names, cfg.data.ignore_label)

    all_pred_insts = []
    all_gt_insts = []
    all_gt_insts_bbox = []

    data_map = {
        "train": cfg.data.metadata.train_list,
        "val": cfg.data.metadata.val_list,
        "test": cfg.data.metadata.test_list
    }

    with open(data_map[split]) as f:
        scene_names = [line.strip() for line in f]

    gt_obbs_list = {}
    pred_obbs_list = {}
    for scan_id in tqdm(scene_names):
        gt_path = os.path.join(gt_file_path, scan_id + ".pth")
        pred_path = os.path.join(pred_file_path, scan_id + ".pth")

        # read ground truth files
        scene = torch.load(gt_path)
        for obb in scene["gt_obbs"]:
            gt_obbs_list.append(obb)

        # read prediction files
        scene = torch.load(pred_path)
        for obb in scene["pred_obbs"]:
            pred_obbs_list.append(obb)

    obb_direction_evaluator = GeneralDatasetEvaluator(cfg.data.class_names, cfg.data.ignore_label)
    obb_direction_eval_result = obb_direction_evaluator.evaluate(pred_obbs_list, gt_obbs_list, print_result=True)
    # self.log("val_eval/AC_10", obb_direction_eval_result["all_ac_10"], prog_bar=True, on_step=False,
    #             on_epoch=True, sync_dist=True, batch_size=1)
    # self.log("val_eval/AC_20", obb_direction_eval_result["all_ac_20"], prog_bar=True, on_step=False,
    #             on_epoch=True, sync_dist=True, batch_size=1)
    # self.log("val_eval/Rerr", obb_direction_eval_result["all_err"], prog_bar=True, on_step=False,
    #             on_epoch=True, sync_dist=True, batch_size=1)
    # all_ac_10 = obb_direction_eval_result["all_ac_10"]
    # all_ac_20 = obb_direction_eval_result["all_ac_20"]
    # all_err = obb_direction_eval_result["all_err"]
    # self.custom_logger.info(f"AC_10: {all_ac_10}")
    # self.custom_logger.info(f"AC_20: {all_ac_20}")
    # self.custom_logger.info(f"Rerr: {all_err}")


if __name__ == "__main__":
    main()
