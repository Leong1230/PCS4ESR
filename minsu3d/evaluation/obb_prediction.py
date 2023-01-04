import numpy as np
"""
Adapted from https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
"""

from copy import deepcopy
import numpy as np
import torch

def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.
    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = ' '.join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle


def rle_decode(rle):
    """Decode rle to get binary mask.
    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle['length']
    counts = rle['counts']
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


# def get_instances(ids, class_ids, class_labels, id2label, ignored_label):
#     instances = {}
#     for label in class_labels:
#         instances[label] = []
#     instance_ids = np.unique(ids)
#     for id in instance_ids:
#         if id == 0:
#             continue
#         inst = Instance(ids, id, ignored_label)
#         if inst.label_id in class_ids:
#             instances[id2label[inst.label_id]].append(inst.to_dict())
#     return instances


# def get_gt_instances(semantic_labels, instance_labels, ignored_classes):
#     """Get gt instances for evaluation."""
#     # convert to evaluation format 0: ignore, 1->N: valid
#     label_shift = len(ignored_classes)
#     semantic_labels = semantic_labels - label_shift + 1
#     semantic_labels[semantic_labels < 0] = 0
#     instance_labels += 1
#     ignore_inds = instance_labels <= 0
#     # scannet encoding rule
#     gt_ins = semantic_labels * 1000 + instance_labels
#     gt_ins[ignore_inds] = 0
#     gt_ins = gt_ins
#     return gt_ins


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id, ignored_label):
        if instance_id == ignored_label:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return np.count_nonzero(mesh_vert_instances == instance_id)

    def to_dict(self):
        dict = {'instance_id': self.instance_id, 'label_id': self.label_id, 'vert_count': self.vert_count,
                'med_dist': self.med_dist, 'dist_conf': self.dist_conf}
        return dict

    def __str__(self):
        return f"({self.instance_id})"


class GeneralDatasetEvaluator(object):

    def __init__(self, class_labels, ignored_label, iou_type=None, use_label=True):
        self.valid_class_labels = class_labels

        self.ious = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = np.array([100])
        self.distance_threshes = np.array([float('inf')])
        self.distance_confs = np.array([-float('inf')])

        self.iou_type = iou_type
        self.use_label = use_label
        if self.use_label:
            self.eval_class_labels = self.valid_class_labels
        else:
            self.eval_class_labels = ['class_agnostic']

    def evaluate_matches(self, matches):
        ious = self.ious
        min_region_sizes = [self.min_region_sizes[0]]
        dist_threshes = [self.distance_threshes[0]]
        dist_confs = [self.distance_confs[0]]

        # results: class x iou
        ac_10 = np.empty((len(dist_threshes), len(self.eval_class_labels)))
        ac_20 = np.empty((len(dist_threshes), len(self.eval_class_labels)))
        ac_5 = np.empty((len(dist_threshes), len(self.eval_class_labels)))
        error = np.empty((len(dist_threshes), len(self.eval_class_labels)))
        ac_10.fill(np.nan)
        ac_20.fill(np.nan)
        ac_5.fill(np.nan)
        error.fill(np.nan)
        for di, (min_region_size, distance_thresh,
                 distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
            for li, label_name in enumerate(self.eval_class_labels):
                tp_10 = 0
                tp_20 = 0
                tp_5 = 0
                fp_10 = 0
                fp_20 = 0
                fp_5 = 0
                angles = 0
                for m in matches:
                    if matches[m]['sem_label']==li:
                        pred_obb_front = matches[m]['pred_front']
                        gt_obb_front = matches[m]['gt_front']
                        front_angle = np.arccos(np.dot(pred_obb_front, gt_obb_front))
                        pred_obb_up = matches[m]['pred_up']
                        gt_obb_up = matches[m]['gt_up']
                        up_angle = np.arccos(np.dot(pred_obb_up, gt_obb_up))
                        angle = front_angle + up_angle
                        angles += angle
                        if angle < 5/180 * 3.14:
                            tp_5 += 1
                        else:
                            fp_5 += 1
                        if angle < 10/180 * 3.14:
                            tp_10 += 1
                        else:
                            fp_10 += 1
                        if angle < 20/180 * 3.14:
                            tp_20 += 1
                        else:
                            fp_20 += 1
                if (tp_10+fp_10)!=0:
                    ac_10_current = np.float(tp_10)/np.float(fp_10+tp_10)
                    ac_20_current = np.float(tp_20)/np.float(tp_20+fp_20)
                    ac_5_current = np.float(tp_5)/np.float(tp_5+fp_5)
                    error_current = np.float(angles)/np.float(tp_10+fp_10)
                    ac_10[di, li] = ac_10_current
                    ac_20[di, li] = ac_20_current
                    ac_5[di, li] = ac_5_current
                    error[di, li] = error_current
        return ac_10, ac_20, ac_5, error

    def compute_averages(self, acs_10, acs_20, acs_5, errs):
        avg_dict = {}
        d_inf = 0
        # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
        avg_dict['all_ac_10'] = np.nanmean(acs_10[d_inf, :])
        avg_dict['all_ac_20'] = np.nanmean(acs_20[d_inf, :])
        avg_dict['all_ac_5'] = np.nanmean(acs_5[d_inf, :])
        avg_dict['all_err'] = np.nanmean(errs[d_inf, :])
        avg_dict['classes'] = {}
        for (li, label_name) in enumerate(self.eval_class_labels):
            avg_dict['classes'][label_name] = {}
            avg_dict['classes'][label_name]['ac_10'] = acs_10[d_inf, li]
            avg_dict['classes'][label_name]['ac_20'] = acs_20[d_inf, li]
            avg_dict['classes'][label_name]['ac_5'] = acs_5[d_inf, li]
            avg_dict['classes'][label_name]['err'] = errs[d_inf, li]
        return avg_dict

    def evaluate(self, pred_list, gt_list, print_result):
        """
        Args:
            pred_list:
                for each scan:
                    for each instance
                        instance = dict(scan_id, label_id, mask, conf)
            gt_list:
                for each scan:
                    for each point:
                        gt_id = class_id * 1000 + instance_id
        """
        assert len(pred_list) == len(gt_list)
        matches = {}
        for i in range(len(pred_list)):
            matches_key = f'obb_{i}'
            matches[matches_key] = {}
            matches[matches_key]['sem_label'] = pred_list[i]["sem_label"]
            matches[matches_key]['pred_front'] = pred_list[i]["front"]
            matches[matches_key]['gt_front'] = gt_list[i]["front"]
            matches[matches_key]['pred_up'] = pred_list[i]["up"]
            matches[matches_key]['gt_up'] = gt_list[i]["up"]
        
        ac_10_scores, ac_20_scores, ac_5_scores, err_scores = self.evaluate_matches(matches)
        avgs = self.compute_averages(ac_10_scores, ac_20_scores, ac_5_scores, err_scores)
        if print_result:
            self.print_results(avgs)
        return avgs

    def print_results(self, avgs):
        sep = ''
        col1 = ':'
        lineLen = 64

        print()
        print('#' * lineLen)
        line = ''
        line += '{:<15}'.format('what') + sep + col1
        line += '{:>8}'.format('AC_5') + sep
        line += '{:>8}'.format('AC_10') + sep
        line += '{:>8}'.format('AC_20') + sep
        line += '{:>8}'.format('Rerr') + sep

        print(line)
        print('#' * lineLen)

        for (li, label_name) in enumerate(self.eval_class_labels):
            ac_5_avg = avgs['classes'][label_name]['ac_5']
            ac_10_avg = avgs['classes'][label_name]['ac_10']
            ac_20_avg = avgs['classes'][label_name]['ac_20']
            err_avg = avgs['classes'][label_name]['err']
            line = '{:<15}'.format(label_name) + sep + col1
            line += sep + '{:>8.3f}'.format(ac_5_avg) + sep
            line += sep + '{:>8.3f}'.format(ac_10_avg) + sep
            line += sep + '{:>8.3f}'.format(ac_20_avg) + sep
            line += sep + '{:>8.3f}'.format(err_avg) + sep
            print(line)

        all_ac_5_avg = avgs['all_ac_5']
        all_ac_10_avg = avgs['all_ac_10']
        all_ac_20_avg = avgs['all_ac_20']
        all_err_avg = avgs['all_err']


        print('-' * lineLen)
        line = '{:<15}'.format('average') + sep + col1
        line += '{:>8.3f}'.format(all_ac_5_avg) + sep
        line += '{:>8.3f}'.format(all_ac_10_avg) + sep
        line += '{:>8.3f}'.format(all_ac_20_avg) + sep
        line += '{:>8.3f}'.format(all_err_avg) + sep
        print(line)
        print('#' * lineLen)
        print()
