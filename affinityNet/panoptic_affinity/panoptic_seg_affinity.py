# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os, shutil, time
from collections import defaultdict
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
import logging
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.registry import Registry
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb

from .multicut_solvers import AsymmetricMulticutModule
from .post_processing import get_panoptic_segmentation_multicut_batch
from .losses import PanopticQualityEval, MaskScore, iou_batch, PanopticQualityLoss, AMWCBackward
from . import utils

logger = logging.getLogger(__name__)

__all__ = ["PanopticAffinity", "AFF_EMBED_BRANCHES_REGISTRY", "build_aff_embed_branch", "AMC_BRANCH_REGISTRY", "build_amc_branch"]

AFF_EMBED_BRANCHES_REGISTRY = Registry("AFF_EMBED_BRANCHES")
AFF_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for affinity embedding branches, which make edge affinities
predictions from feature maps.
"""

AMC_BRANCH_REGISTRY = Registry("AMC_BRANCHES")
AMC_BRANCH_REGISTRY.__doc__ = """
Registry for asymmetric multicut (amc) costs refinement and amc module embedding branches.
"""

@AMC_BRANCH_REGISTRY.register()
class PanopticAffinityAMWC(nn.Module):
    @configurable
    def __init__(self, 
                lambda_val_start,
                lambda_val_end, 
                sem_seg_ignore_val, 
                num_classes,
                amc_loss_weight,
                meta,
                edge_sampling_intervals,
                stuff_area, 
                instance_area,
                dropout,
                ignore_instance_size_amc,
                downsampling_factor_stuff_areas = 1,
                backprop_num_samples = 1, 
                eval_pq_train = False):

        super().__init__()
        self.thing_ids = meta.thing_dataset_id_to_contiguous_id.values()
        self.lambda_val_start = lambda_val_start
        self.lambda_val_end = lambda_val_end
        self.ignore_instance_size_amc = ignore_instance_size_amc
        self.backprop_num_samples = backprop_num_samples
        self.eval_pq_train = eval_pq_train

        self.sem_seg_ignore_val = sem_seg_ignore_val
        self.num_classes = num_classes
        self.amc_loss_weight = amc_loss_weight
        self.meta = meta
        self.pq_eval = PanopticQualityEval(meta=self.meta)
        self.stuff_area = stuff_area
        self.instance_area = instance_area
        self.dropout = dropout
        self.edge_dropout_layer = torch.nn.Dropout(p = self.dropout, inplace = True)
        self.seg_dropout_layer = utils.DropoutPixels(p = self.dropout)

        self.seg_cost_weights = [
            torch.nn.Parameter(torch.ones(1, requires_grad = True)), 
            torch.nn.Parameter(torch.zeros(1, requires_grad = True)) 
        ] 
        self.seg_cost_weights = torch.nn.ParameterList(self.seg_cost_weights)
        self.edge_costs_weights = []
        for i in range(len(edge_sampling_intervals)):
            self.edge_costs_weights.append(torch.nn.Parameter(torch.ones(1, requires_grad = True)))
            self.edge_costs_weights.append(torch.nn.Parameter(torch.zeros(1, requires_grad = True)))
        self.edge_costs_weights = torch.nn.ParameterList(self.edge_costs_weights)

        self.downsampling_factor = 4
        self.downsampling_factor_stuff_areas = downsampling_factor_stuff_areas
        per_classes_areas = [self.stuff_area] * num_classes
        for t in self.thing_ids:
            per_classes_areas[t] = self.instance_area
        self.training_per_class_areas = torch.nn.Parameter(torch.as_tensor(per_classes_areas) * torch.ones(num_classes, requires_grad = True))

        self.panoptic_losses = torch.nn.ModuleDict({'PQ': PanopticQualityLoss(self.thing_ids)}) 
        self.amwc_grad_solver = AMWCBackward() # For transformation to MWC and computing gradients
        self.amwc_module = AsymmetricMulticutModule(self.thing_ids) # For forward pass to compute AMWC

        self.validation_mode = False
        self.similarity_function = iou_batch
        
    @classmethod
    def from_config(cls, cfg, sem_seg_ignore_val):
        ret = dict(
                edge_sampling_intervals = cfg.MODEL.AFF_EMBED_HEAD.EDGE_SAMPLING_INTERVALS,
                amc_loss_weight = cfg.MODEL.PANOPTIC_AFFINITY.AMC_LOSS_WEIGHT, 
                lambda_val_start = cfg.MODEL.PANOPTIC_AFFINITY.LAMBDA_VAL_START, 
                lambda_val_end = cfg.MODEL.PANOPTIC_AFFINITY.LAMBDA_VAL_END, 
                sem_seg_ignore_val = sem_seg_ignore_val, 
                num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                stuff_area = cfg.MODEL.PANOPTIC_AFFINITY.STUFF_AREA, 
                instance_area = cfg.MODEL.PANOPTIC_AFFINITY.INSTANCE_AREA,
                dropout = cfg.MODEL.PANOPTIC_AFFINITY.DROPOUT,
                downsampling_factor_stuff_areas = cfg.MODEL.PANOPTIC_AFFINITY.STUFF_AREA_DOWNSAMPLING_FACTOR_TRAIN,
                ignore_instance_size_amc = cfg.MODEL.PANOPTIC_AFFINITY.IGNORE_INSTANCE_SIZE_AMC,
                backprop_num_samples = cfg.MODEL.PANOPTIC_AFFINITY.ROBUST_BACKPROP_NUM_SAMPLES,
                eval_pq_train = cfg.MODEL.PANOPTIC_AFFINITY.EVAL_PQ_DURING_TRAIN)

        return ret

    def compute_foreground_probabilities(self, panoptic_mask_areas, class_labels_to_index, thing_ids):
        foreground_probabilities = []
        downsampling_factor = self.downsampling_factor if self.training else 1
        for (b, current_areas) in enumerate(panoptic_mask_areas):
            current_prob = torch.zeros_like(current_areas)
            for c in range(self.num_classes):
                if c not in class_labels_to_index[b]:
                    continue
                indices = class_labels_to_index[b][c]
                current_class_sizes = current_areas[indices]
                if c in thing_ids:
                    weight = 0.1
                    current_area_threshold = self.instance_area / downsampling_factor
                    p = torch.sigmoid(weight * (current_class_sizes - current_area_threshold))
                else:
                    weight = 0.1
                    current_area_threshold = self.stuff_area / downsampling_factor
                    eff_area_thresh = current_area_threshold
                    if self.training:
                        eff_area_thresh = current_area_threshold / self.downsampling_factor_stuff_areas
                    p = torch.sigmoid(weight * (current_class_sizes - eff_area_thresh))

                current_prob[indices] = p
            
            foreground_probabilities.append(current_prob)
        return foreground_probabilities

    def forward(self, segmentation_costs, affinity_costs, 
                    edge_distances, edge_sampling_intervals,
                    affinity_weights = None, panoptic_gt = None, 
                    segments_info = None, panoptic_weights = None):

        if self.training and self.amc_loss_weight == 0:
            return {}, {}, {}, {}, {}, {}

        start = time.time()
        batch_size = segmentation_costs.shape[0]

        segmentation_costs = (segmentation_costs * self.seg_cost_weights[0]) + self.seg_cost_weights[1]
        if self.training:
            segmentation_costs = self.seg_dropout_layer(segmentation_costs) # Should not scale costs due to dropout in inference.

        for i, si in enumerate(edge_sampling_intervals):
            affinity_costs[i] = (affinity_costs[i] * torch.abs(self.edge_costs_weights[2 * i])) + self.edge_costs_weights[2 * i + 1]
            if self.training:
                affinity_costs[i] = self.edge_dropout_layer(affinity_costs[i]) * (1.0 - self.dropout) # remove the scaling effect of drop-out.

        # Solve AMWC
        out = self.amwc_module(segmentation_costs, affinity_costs, edge_distances, edge_sampling_intervals, self.ignore_instance_size_amc // self.downsampling_factor)
        start_index = 0
        node_labels_pred = out[0]; start_index += 1
        edge_labels_pred = out[start_index:start_index+len(edge_distances)]; start_index += len(edge_distances)
        panoptic_pred_one_hot = tuple(out[start_index:start_index + batch_size]); start_index += batch_size
        index_to_class_labels = out[start_index: start_index + batch_size]

        end = time.time()
        amc_time = end - start
        print_str = ""
        amc_losses = {}
        foreground_probs = None
        if not self.training:
            # During inference time, upsample the panoptic prediction.
            panoptic_pred_one_hot = list(panoptic_pred_one_hot)
            for b in range(len(panoptic_pred_one_hot)):
                panoptic_pred_one_hot[b] = F.interpolate(panoptic_pred_one_hot[b].unsqueeze(0), scale_factor = self.downsampling_factor, mode="nearest").squeeze(0)
            node_labels_pred = F.interpolate(node_labels_pred, scale_factor = self.downsampling_factor, mode="nearest")
            segmentation_costs = F.interpolate(segmentation_costs, scale_factor = self.downsampling_factor, mode="bilinear", align_corners=False)
            mask_areas = [m.sum([1, 2]) for m in panoptic_pred_one_hot]
            class_labels_to_index_batch = []
            for b in range(batch_size):
                class_labels_to_index = defaultdict(list)
                for ind in range(index_to_class_labels[b].shape[0]):
                    class_id = index_to_class_labels[b][ind].item()
                    class_labels_to_index[class_id].append(ind)
                class_labels_to_index_batch.append(class_labels_to_index)

            foreground_probs = self.compute_foreground_probabilities(mask_areas, class_labels_to_index_batch, self.thing_ids)

        else:
            storage = get_event_storage()

            start = time.time()
            with torch.no_grad():
                panoptic_ids_gt_one_hot_batch, category_indices_gt_batch, num_valid_channels_gt, num_padding_channels_pred = utils.ComputeBestGroundTruthBatch(
                    panoptic_pred_one_hot, index_to_class_labels, panoptic_gt, panoptic_weights, self.thing_ids, self.meta.label_divisor, self.num_classes, self.similarity_function)
            end = time.time()
            best_gt_time = end - start
            print_str += f", best_gt_time t: {best_gt_time :.1f}" 

            start = time.time()
            # Prepare for MWC backward pass and calculate PQ surrogate.
            params = self.amwc_module.params
            params['num_classes'] = self.num_classes
            params['lambda_val_start'] = self.lambda_val_start
            params['lambda_val_end'] = self.lambda_val_end
            params['backprop_num_samples'] = self.backprop_num_samples
            params['num_padding_channels'] = num_padding_channels_pred
            params['max_channels'] = panoptic_ids_gt_one_hot_batch.shape[1]
            params['category_indices'] = category_indices_gt_batch
            params['num_valid_channels_gt'] = num_valid_channels_gt

            edge_labels_pred = tuple(edge_labels_pred)
            affinity_costs = tuple([a.detach() for a in affinity_costs])
            affinity_weights = tuple([a for a in affinity_weights])
            solver_in = (node_labels_pred, ) + (segmentation_costs, ) + edge_labels_pred + affinity_costs + affinity_weights + (panoptic_weights, ) + panoptic_pred_one_hot + (panoptic_ids_gt_one_hot_batch, ) + (params, )

            # Order: node_labels, node_costs (detached), tuple(edge_labels), tuple(edge_costs(detached)), tuple(panoptic_one_hot_pred(detached)), params
            padded_panoptic_pred_one_hot = self.amwc_grad_solver.apply(*solver_in)
            mask_areas = torch.unbind(padded_panoptic_pred_one_hot.sum([2, 3]), 0)
            foreground_probs = self.compute_foreground_probabilities(mask_areas, category_indices_gt_batch, self.thing_ids)
            foreground_probs = torch.stack(foreground_probs)
            
            for name, loss_module in self.panoptic_losses.items():
                loss, loss_per_cat = loss_module(padded_panoptic_pred_one_hot, 
                                                panoptic_ids_gt_one_hot_batch, 
                                                category_indices_gt_batch, 
                                                panoptic_weights, 
                                                foreground_probs,
                                                self.similarity_function)

                amc_losses.update({name: self.amc_loss_weight * loss})

            if not self.validation_mode:
                storage.put_scalar('norm_loss', loss.item())
        
            if self.eval_pq_train: # To report pq metric during training (does not influence optimization)
                panoptic_image, _, _, _ = get_panoptic_segmentation_multicut_batch(
                    panoptic_pred_one_hot, foreground_probs, index_to_class_labels, -1, self.meta.label_divisor, self.meta.thing_dataset_id_to_contiguous_id.values(), self.num_classes, False)
                self.pq_results, _, _, _, _ = self.pq_eval.eval_batch(panoptic_gt.detach().cpu().numpy(), segments_info, np.stack(panoptic_image, 0))
                for k in ['iou', 'tp', 'fp', 'fn', 'pq', 'sq', 'rq']:
                    storage.put_scalar(k, self.pq_results['All'][k])
                    print_str = ", " + k + ": " f"{self.pq_results['All'][k]:.2f}" + print_str

            end = time.time()
            post_process_time = end - start
            print_str += f", loss_c t: {post_process_time :.1f}" 
            
            for i in range(len(edge_sampling_intervals)):
                storage.put_scalar('edge_mult_' + str(i), torch.abs(self.edge_costs_weights[2 * i][0]).item())
                storage.put_scalar('edge_offset_' + str(i), self.edge_costs_weights[2 * i + 1][0].item())
            storage.put_scalar('seg_mult_', self.seg_cost_weights[0][0].item())
            storage.put_scalar('seg_offset_', self.seg_cost_weights[1][0].item())
            # print("Actual PQ:")
            # for c in pq_results['per_class']:
            #     print(f"{c}: pq: {pq_results['per_class'][c]['pq']:.3f}")
            print_str = f"loss: {sum(amc_losses.values()):.2f}, " + print_str
        
        print(print_str)
        return amc_losses, node_labels_pred, edge_labels_pred, panoptic_pred_one_hot, foreground_probs, index_to_class_labels

@META_ARCH_REGISTRY.register()
class PanopticAffinity(nn.Module):
    """
    Main class for panoptic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.AFF_EMBED_head = build_aff_embed_branch(cfg, self.backbone.output_shape())
        self.amwc_layer = build_amc_branch(cfg, self.sem_seg_head.ignore_value)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.edge_distances = cfg.MODEL.AFF_EMBED_HEAD.EDGE_DISTANCES
        self.edge_sampling_intervals = cfg.MODEL.AFF_EMBED_HEAD.EDGE_SAMPLING_INTERVALS
        self.amc_loss_weight = cfg.MODEL.PANOPTIC_AFFINITY.AMC_LOSS_WEIGHT
        self.size_divisibility = cfg.MODEL.SIZE_DIVISIBILITY
        self.mask_scorer = MaskScore() 

        self.stuff_area = cfg.MODEL.PANOPTIC_AFFINITY.STUFF_AREA
        self.instance_area = cfg.MODEL.PANOPTIC_AFFINITY.INSTANCE_AREA
        self.predict_instances = cfg.MODEL.PANOPTIC_AFFINITY.PREDICT_INSTANCES
        self.input_format = cfg.INPUT.FORMAT
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.OUTPUT_DIR = cfg.OUTPUT_DIR
        self.save_result_images = cfg.MODEL.SAVE_RESULT_IMAGES  
        self.eval_vis_dir = os.path.join(cfg.OUTPUT_DIR, "eval_vis") 
        self.eval_results_dir = os.path.join(cfg.OUTPUT_DIR, "eval_res")
        os.makedirs(self.eval_vis_dir, exist_ok=True)
        os.makedirs(self.eval_results_dir, exist_ok=True)
        
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "aff_mask": ground truth locations where edge affinities need to be correct
                   * "row_affinities": ground truth edge affinities in row (y-axis) direction
                   * "col_affinities": ground truth edge affinities in column (x-axis) direction
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:
                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        processed_results = []
        start = time.time()
        backbone_features = self.backbone(images.tensor)
        losses = {}
        if "sem_seg" in batched_inputs[0]:
            sem_seg_targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            sem_seg_targets = ImageList.from_tensors(sem_seg_targets, self.size_divisibility, self.sem_seg_head.ignore_value).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                sem_seg_weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                sem_seg_weights = ImageList.from_tensors(sem_seg_weights, self.size_divisibility).tensor
            else:
                sem_seg_weights = None
        else:
            sem_seg_targets = None
            sem_seg_weights = None
        sem_seg_losses, sem_seg_logits_wo_interp = self.sem_seg_head(backbone_features, sem_seg_targets, sem_seg_weights)
        losses.update(sem_seg_losses)

        if "affinity_weights" in batched_inputs[0] and "affinities" in batched_inputs[0]:
            affinity_weights = []
            aff_targets = []
            for (i, e) in enumerate(self.edge_distances):
                current_edge_weights = []
                current_edge_targets = []
                for x in batched_inputs:
                    current_edge_weights.append(x["affinity_weights"][i].to(self.device))
                    current_edge_targets.append(x["affinities"][i].to(self.device))
                affinity_weights.append(torch.stack(current_edge_weights, 0))
                aff_targets.append(torch.stack(current_edge_targets, 0))

        else:
            affinity_weights = None
            aff_targets = None

        if "panoptic" in batched_inputs[0]:
            # Not sending panoptic related arrays to GPU.
            panoptic = [x["panoptic"] for x in batched_inputs] 
            panoptic = ImageList.from_tensors(panoptic, 0, -1).tensor
            segments_info = [x["segments_info"] for x in batched_inputs]
            panoptic_weights = [x["panoptic_weights"] for x in batched_inputs]
            panoptic_weights = ImageList.from_tensors(panoptic_weights, 0).tensor
            
        else:
            panoptic = None
            panoptic_weights = None
            segments_info = None

        amc_losses = {}

        aff_losses, aff_logits = self.AFF_EMBED_head(backbone_features, sem_seg_logits_wo_interp, aff_targets, affinity_weights)
        #if not self.training:
            # print(f"Network forward pass time: {time.time() - start}")
        segmentation_costs, affinity_costs = utils.ComputeMulticutCosts(sem_seg_logits_wo_interp, aff_logits)
        amc_losses, node_labels_pred, edge_labels_pred, panoptic_pred_one_hots, foreground_probs, index_to_class_labels = self.amwc_layer(
                                                                                                        segmentation_costs, affinity_costs,
                                                                                                        self.edge_distances, self.edge_sampling_intervals,
                                                                                                        affinity_weights, panoptic, segments_info, panoptic_weights)
        losses.update(aff_losses)
        losses.update(amc_losses)

        if self.training:
            return losses

        from PIL import Image 
        from detectron2.utils.visualizer import Visualizer

        processed_results = []
        sem_seg_logits = F.interpolate(sem_seg_logits_wo_interp, scale_factor=self.sem_seg_head.common_stride, mode="bilinear", align_corners=False)
        for b, (node_label, panoptic_pred_one_hot, sem_seg_logit, input_per_image, orig_image_size, foreground_prob, index_to_class_label) in enumerate(zip(
            node_labels_pred, panoptic_pred_one_hots, sem_seg_logits, batched_inputs, images.image_sizes, foreground_probs, index_to_class_labels)):

            img = input_per_image["image"]
            height = input_per_image.get("height")
            width = input_per_image.get("width")

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format).astype("uint8")
            img = np.array(Image.fromarray(img).resize((width, height)))
            
            node_label = utils.interpolate_mask(node_label.to(torch.float32), orig_image_size, height, width)
            panoptic_pred_one_hot = utils.interpolate_mask(panoptic_pred_one_hot.to(torch.float32), orig_image_size, height, width)
            sem_seg_logit = utils.interpolate_mask(sem_seg_logit, orig_image_size, height, width)
            
            panoptic_image, instance_ids, only_instances_img, seg_mask_img = get_panoptic_segmentation_multicut_batch(
                [panoptic_pred_one_hot], [foreground_prob], [index_to_class_label], -1, self.meta.label_divisor, self.meta.thing_dataset_id_to_contiguous_id.values(), self.num_classes, self.save_result_images) 
            
            panoptic_image = panoptic_image[0]; 
            instance_ids = instance_ids[0]; 
            only_instances_img = only_instances_img[0]; 
            seg_mask_img = seg_mask_img[0]

            # Using node labels from amc post-processing ignoring void label of panoptic segmentation:
            current_result = {"sem_seg": node_label}
            current_result["orig_sem_seg"] = sem_seg_logit # For semantic segmentation evaluation by direct argmax.

            panoptic_image = torch.from_numpy(panoptic_image)
            
            # For panoptic segmentation evaluation.
            current_result["panoptic_seg"] = (panoptic_image, None)
            
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results. (might need to be revisited for COCO due to padding)
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        
                        sem_scores = sem_seg_logit[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask]) # Mean segmentation logits.
                        # Pad the instance mask so that it aligns with the edge affinities:
                        mask_padded = ImageList.from_tensors([mask.to(torch.float32)], size_divisibility = self.size_divisibility, pad_value = 0).tensor.squeeze()

                        # Calculate Inter-mask mean similarity - Intra-mask mean similarity.
                        instance_affinity_scores = self.mask_scorer(mask_padded, aff_logits, b, self.edge_distances, self.edge_sampling_intervals).cpu()
                        instance.scores = torch.tensor([sem_scores + instance_affinity_scores], device=panoptic_image.device)

                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    current_result["instances"] = Instances.cat(instances)

            processed_results.append(current_result) 

            if self.save_result_images:
                image_path = input_per_image['file_name'].split(os.sep)
                image_name = os.path.splitext(image_path[-1])[0] 

                v_panoptic = Visualizer(img, self.meta)
                v_panoptic = v_panoptic.draw_panoptic_seg_predictions(panoptic_image, None, alpha=0.3)
                pan_img = v_panoptic.get_image()
                
                Image.fromarray(pan_img).save(os.path.join(self.eval_vis_dir, image_name + '_panoptic.png'))
                Image.fromarray(only_instances_img).save(os.path.join(self.eval_vis_dir, image_name + '_instances.png'))
                Image.fromarray(seg_mask_img).save(os.path.join(self.eval_vis_dir, image_name + '_seg_mask.png'))
                
                for (i, e) in enumerate(self.edge_distances):
                    row_label = sem_seg_postprocess(edge_labels_pred[i][b, 0:1, ::].to(torch.float32), orig_image_size, height, width)
                    col_label = sem_seg_postprocess(edge_labels_pred[i][b, 1:2, ::].to(torch.float32), orig_image_size, height, width)
                    both_labels = row_label + col_label
                    both_labels[both_labels > 1] = 1
                    row_cost = sem_seg_postprocess(affinity_costs[i][b, 0:1, ::].to(torch.float32), orig_image_size, height, width)
                    col_cost = sem_seg_postprocess(affinity_costs[i][b, 1:2, ::].to(torch.float32), orig_image_size, height, width)
                    Image.fromarray(utils.get_vis_img(row_cost)).save(
                        os.path.join(self.eval_vis_dir, image_name + '_row_aff_' + str(e) + '.png'))
                    Image.fromarray(utils.get_vis_img(col_cost)).save(
                        os.path.join(self.eval_vis_dir, image_name + '_col_aff_' + str(e) + '.png'))
                    Image.fromarray(utils.get_vis_img(row_label)).save(
                        os.path.join(self.eval_vis_dir, image_name + '_row_edge_' + str(e) + '.png'))
                    Image.fromarray(utils.get_vis_img(col_label)).save(
                        os.path.join(self.eval_vis_dir, image_name + '_col_edge_' + str(e) + '.png'))
                    Image.fromarray(utils.get_vis_img(both_labels)).save(
                        os.path.join(self.eval_vis_dir, image_name + '_both_labels_' + str(e) + '.png'))
                    
                    v_edge = Visualizer(img, self.meta)
                    v_edge = v_edge.draw_binary_mask(both_labels.squeeze().numpy(), None, alpha=0.3)

                    edge_img = v_panoptic.get_image()
                    Image.fromarray(utils.get_vis_img(edge_img)).save(
                        os.path.join(self.eval_vis_dir, image_name + '_both_labels_img_' + str(e) + '.png'))

                seg_mask_separate = utils.get_numpy_image(sem_seg_logit.argmax(dim=0))
                seg_mask_separate_image = utils.apply_segmented_cmap(seg_mask_separate, plt.cm.tab20, max_v=sem_seg_logit.shape[0])
                Image.fromarray(seg_mask_separate_image).save(os.path.join(self.eval_vis_dir, image_name + '_seg_mask_sep.png'))
                Image.fromarray(img).save(os.path.join(self.eval_vis_dir, image_name + '_input.png'))
            
        return processed_results 

@SEM_SEG_HEADS_REGISTRY.register()
class PanopticAffinitySemSegHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
        **kwargs,
    ):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.loss_weight = loss_weight
        use_bias = norm == ""
        # `head` is additional transform before predictor
        self.head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.head[0])
        weight_init.c2_xavier_fill(self.head[1])
        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        elif loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=ignore_value, top_k_percent_pixels=loss_top_k)
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        return ret

    def forward(self, features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.training:
            return self.losses(y, targets, weights), y
        else:
            return {}, y

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        y = self.head(y)
        y = self.predictor(y)
        return y

    def losses(self, predictions, targets, weights=None):
        if self.loss_weight == 0.0:
            return {}

        loss = self.loss(F.interpolate(predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False), targets, weights)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

def build_aff_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.AFF_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.AFF_EMBED_HEAD.NAME
    return AFF_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)

def build_amc_branch(cfg, sem_seg_ignore_val):
    """
    Build a multicut branch from `cfg.MODEL.AMWC_LAYER.NAME`.
    """
    name = cfg.MODEL.AMWC_LAYER.NAME
    return AMC_BRANCH_REGISTRY.get(name)(cfg, sem_seg_ignore_val)

@AFF_EMBED_BRANCHES_REGISTRY.register()
class PanopticAffinityInsEmbedHead(DeepLabV3PlusHead):
    """
    A affinity embedding head adapted from :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        affinity_loss_weight: float,
        edge_distances: List[int], 
        affinity_loss_top_k: float, 
        num_classes: int,
        edge_sampling_intervals: List[int],
        **kwargs,
    ):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage). 
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the size of pixel feature channels which are fed to affinity classifier.
            affinity_loss_weight (float): loss weight for boundary prediction.
                edge_distances (List[int]): Defines the size of + shaped neighbourhood around each pixel for 
                computing pixel affinities. For more detail see PanopticAffinityTargetGenerator.__init__() 
                at target_generator.py
            affinity_loss_top_k: (float): The value lies in [0.0, 1.0]. When its
                value < 1.0, only compute the loss for the top k percent pixels
                (e.g., the top 20% pixels). This is useful for hard pixel mining. (Inspired from DeepLab loss)
            num_classes: (int) Number of classes in semantic segmentation. 
                Used to calculate number of channels of semantic segmentation logits
            edge_sampling_intervals: See target_generator.py

        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.affinity_loss_top_k = affinity_loss_top_k
        self.affinity_loss_weight = affinity_loss_weight
        self.edge_distances = edge_distances
        self.num_classes = num_classes
        self.edge_sampling_intervals = edge_sampling_intervals
        use_bias = norm == ""
        self.affinity_head = nn.Sequential(
            Conv2d(
                decoder_channels[0] + self.num_classes,
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation = F.relu
            ),
        )           
        weight_init.c2_xavier_fill(self.affinity_head[0])
        weight_init.c2_xavier_fill(self.affinity_head[1])

        classifiers = []
        for i, e in enumerate(self.edge_distances):
            # Compute upsampling/downsampling factor, only one would be acting at one time.
            downsampling_factor = int(self.edge_sampling_intervals[i])
            # Fine scale features are getting both row and col affinities from same classifier. For others,
            # first take central difference in appropiate row/col direction and concatenate both features.
            # Each group will work on either row/col features in this case:
            in_factor = 1 if e <= 1 else 2
            current_edge_classifier = nn.Sequential(
                Conv2d(in_factor * head_channels, 
                        in_factor * int(head_channels / 4), 
                        kernel_size=3, padding=1, bias=use_bias, 
                        norm=get_norm(norm, in_factor * int(head_channels / 4)), 
                        activation=F.relu, 
                        groups = in_factor),
                Conv2d(in_factor * int(head_channels / 4), 
                       in_factor * int(head_channels / 8), 
                       kernel_size=3, padding=1, bias=use_bias, 
                       norm=get_norm(norm, in_factor * int(head_channels / 8)), 
                       activation = F.relu, 
                       groups = in_factor),
                Conv2d(in_factor * int(head_channels / 8), 2, kernel_size=1, bias=True, stride = max(downsampling_factor, 1), activation = None))

            weight_init.c2_xavier_fill(current_edge_classifier[0])
            weight_init.c2_xavier_fill(current_edge_classifier[1])    
            weight_init.c2_xavier_fill(current_edge_classifier[2])    

            classifiers.append(current_edge_classifier)

        self.classifiers = nn.ModuleList(classifiers)

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.AFF_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.AFF_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.AFF_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape=input_shape,
            in_features=cfg.MODEL.AFF_EMBED_HEAD.IN_FEATURES,
            project_channels=cfg.MODEL.AFF_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.AFF_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.AFF_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.AFF_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.AFF_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.AFF_EMBED_HEAD.HEAD_CHANNELS,
            affinity_loss_weight=cfg.MODEL.AFF_EMBED_HEAD.AFF_LOSS_WEIGHT,
            edge_distances=cfg.MODEL.AFF_EMBED_HEAD.EDGE_DISTANCES,
            affinity_loss_top_k = cfg.MODEL.AFF_EMBED_HEAD.LOSS_TOP_K,
            num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
            edge_sampling_intervals=cfg.MODEL.AFF_EMBED_HEAD.EDGE_SAMPLING_INTERVALS)
        return ret

    def forward(
        self,
        backbone_features,
        sem_seg_logits,
        aff_targets=None,
        affinity_weights=None, 
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        aff_logits = self.layers(backbone_features, sem_seg_logits) 

        if self.training:
            loss = self.compute_affinity_loss(aff_logits, aff_targets, affinity_weights, '', nn.BCEWithLogitsLoss)
            return loss, aff_logits
        else:
            return {}, aff_logits

    def layers(self, backbone_features, sem_seg_logits):
        assert self.decoder_only

        y = super().layers(backbone_features)
        affinities_features = self.affinity_head(torch.cat((y, sem_seg_logits.detach()), 1)) 

        affinities_logits = []
        for (i, e) in enumerate(self.edge_distances):
            if e == 1:
                # Here not taking difference.
                affinities_logits.append(self.classifiers[i](affinities_features))
            else:
                assert(e > 1)
                # Row features:
                padding = int(e / 2.0)
                output_padding = (0, 0, padding, padding)

                row_diff = torch.nn.ZeroPad2d(output_padding)(affinities_features[:, :, e:, :] - affinities_features[:, :, :-e, :])
                # Column features:
                output_padding = (padding, padding, 0, 0)

                col_diff = torch.nn.ZeroPad2d(output_padding)(affinities_features[:, :, :, e:] - affinities_features[:, :, :, :-e])
                affinities_logits.append(self.classifiers[i](torch.cat((row_diff, col_diff), 1)))

        return affinities_logits

    def compute_affinity_loss(self, predictions, targets, affinity_weights, loss_type, loss_func):
        losses = {}
        if self.affinity_loss_weight == 0.0:
            return losses 

        for (i, e) in enumerate(self.edge_distances):
            for c in range(predictions[i].shape[1]): # Separate loss on horizontal and vertical edges.
                loss_e = loss_func(reduction="none")(predictions[i][:,c,::], targets[i][:,c,::].to(torch.float32)) * affinity_weights[i][:,c,::]

                if self.affinity_loss_top_k < 1.0:
                    top_k_pixels = int(self.affinity_loss_top_k * loss_e.numel())
                    loss_e, _ = torch.topk(loss_e.contiguous().flatten(), top_k_pixels)

                losses.update({"loss_affinities_dist_" + str(e) + "_c_" + str(c) + loss_type: loss_e.mean() * self.affinity_loss_weight})
        return losses