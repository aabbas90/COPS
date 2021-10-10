import torch
import torch.nn.functional as F
import numpy as np 
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import time, sys
import logging
from . import utils
import torch.multiprocessing as mp
logger = logging.getLogger(__name__)

def get_panoptic_segmentation_multicut_batch(panoptic_one_hot_batch, foreground_probs_batch, index_to_class_labels_batch, pan_void_label, label_divisor, thing_ids, num_classes, return_images = True):
    batch_size = len(panoptic_one_hot_batch)
    panoptic_image_batch = []
    instance_ids_batch = []
    only_instances_img_batch = []
    seg_mask_img_batch = []
    for b in range(batch_size):
        current_pan_one_hot = panoptic_one_hot_batch[b].cpu().detach().numpy()
        current_foreground_probs = foreground_probs_batch[b]
        current_index_to_class_labels = index_to_class_labels_batch[b]
        current_pan_out = np.zeros((current_pan_one_hot.shape[1], current_pan_one_hot.shape[2]), dtype=np.int64) + pan_void_label
        current_seg_mask = np.zeros((current_pan_one_hot.shape[1], current_pan_one_hot.shape[2]))
        mask_instance = np.zeros((current_pan_one_hot.shape[1], current_pan_one_hot.shape[2]), dtype = np.bool)
        free_ids = [0] * num_classes
        for idx, c in enumerate(current_index_to_class_labels):
            if current_foreground_probs[idx].item() < 0.5:
                continue
            current_mask = current_pan_one_hot[idx] == 1
            if c not in thing_ids:
                current_pan_out[current_mask] = c * label_divisor
            else:
                current_pan_out[current_mask] = c * label_divisor + free_ids[c]
                mask_instance[current_mask] = True
                free_ids[c] += 1
            current_seg_mask[current_mask] = c

        instance_mask = current_pan_out.copy()
        instance_mask[~mask_instance] = 0
        if return_images:
            instance_ids_image = utils.apply_segmented_cmap(instance_mask)
            seg_mask_image = utils.apply_segmented_cmap(current_seg_mask, plt.cm.tab20, max_v=num_classes)
        else:
            instance_ids_image = None
            seg_mask_image = None

        panoptic_image_batch.append(current_pan_out)
        instance_ids_batch.append(instance_mask)
        only_instances_img_batch.append(instance_ids_image)
        seg_mask_img_batch.append(seg_mask_image)

    return panoptic_image_batch, instance_ids_batch, only_instances_img_batch, seg_mask_img_batch
