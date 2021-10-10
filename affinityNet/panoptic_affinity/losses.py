import torch
import torch.nn as nn 
from panopticapi.evaluation import PQStat
import numpy as np

from .multicut_solvers import solve_mc_grad_avg_batch

OFFSET = 256 * 256 * 256
VOID = 0

def iou_batch(pred, target, weight, pixel_dims): 
    eps = 1e-1
    intersection_dense = pred * target
    intersection = (intersection_dense * weight).sum(pixel_dims)
    union = ((pred + target - intersection_dense) * weight).sum(pixel_dims)
    iou = (intersection + eps) / (union + eps)
    return iou

class PanopticQualityLoss(nn.Module):
    def __init__(self, thing_ids, eps = 1e-1):
        super().__init__() 
        self.thing_ids = thing_ids
        self.eps = eps

    def forward(self, pan_pred_batch, pan_gt_batch, category_indices_batch, weights, foreground_prob, similarity_function):
        def soft_threshold(input, threshold = 0.5, beta = 4.0):
            assert(threshold == 0.5) # Only works for 0.5
            # return 1.0 / (1.0 + torch.pow(input / (1.0 - input), -beta))
            return torch.pow(input, beta) / (torch.pow(input, beta) + torch.pow(1.0 - input, beta))

        def sigmoid_threshold(input, threshold = 0.5, scalar = 15.0):
            return torch.sigmoid(scalar * (input - threshold))

        ious = similarity_function(pan_pred_batch, pan_gt_batch, weights.unsqueeze(1), [2, 3])
        true_probability = soft_threshold(ious, threshold=0.5)
        false_probability = 1.0 - true_probability

        gt_non_zero_mask = torch.any(torch.any(pan_gt_batch > 0, 2), 2).to(torch.float32)
        # Populate all categories present in the whole batch to compute class averaged loss.
        batch_size = ious.shape[0]
        all_cats = set()
        for b in range(batch_size):
            current_cats = category_indices_batch[b].keys()
            all_cats.update(list(current_cats))
        
        full_pq = 0.0
        valid_number_cats = 0
        log_frac_pq_per_cat = {}
        for current_cat in all_cats:
            numerator_per_cat = 0.0
            denominator_per_cat = 0.0

            for b in range(batch_size):
                # See if current class exists in the image
                if current_cat not in category_indices_batch[b]:
                    continue 
                
                indices = category_indices_batch[b][current_cat]
                current_gt_non_zero_mask = gt_non_zero_mask[b, indices]
                
                current_foreground_prob = foreground_prob[b, indices] 

                # For a TP:
                # 1. IOU > 0.5 (true_probability)
                # 2. Ground-truth mask should be > 0. (Since IOU function has epsilon factor, it can give 1 IOU for p = 1, g = 0 with eps = 1)
                # 3. It should be a foreground object as told by 'foreground_prob' 
                tp_indicator = true_probability[b, indices] * current_foreground_prob * current_gt_non_zero_mask
                numerator_per_cat += (tp_indicator * ious[b, indices]).sum()
                soft_num_tp = tp_indicator.sum()

                # For a FN: The metric checks for all valid (non-zero) GT masks against the predictions.
                soft_num_fn = (false_probability[b, indices] * current_gt_non_zero_mask).sum()

                # For a FP: The influence of a FP can be decreased if the network predicts non-foreground 
                # through foreground_prob ~ 0.
                soft_num_fp = ((1.0 - current_gt_non_zero_mask) * false_probability[b, indices] * current_foreground_prob).sum()
                denominator_per_cat += soft_num_tp + 0.5 * soft_num_fn + 0.5 * soft_num_fp

            # Add the per-class IoU loss to full loss.
            if denominator_per_cat > 0:
                pq_per_cat = (numerator_per_cat + self.eps) / (denominator_per_cat + self.eps)
                # print(f"{current_cat}: pq: {pq_per_cat.item():.3f}, num: {numerator_per_cat.item():.3f}, den: {denominator_per_cat.item():.3f}")
                full_pq = full_pq + pq_per_cat
                valid_number_cats += 1
                log_frac_pq_per_cat[current_cat] = pq_per_cat.item()

        # Convert to average.
        full_pq = full_pq / float(valid_number_cats)
        for key in log_frac_pq_per_cat:
            log_frac_pq_per_cat[key] = log_frac_pq_per_cat[key] / (full_pq.item() * valid_number_cats + 1e-6)

        # print(f"FULL iou: {full_pq.item():.4f}")
        return 1.0 - full_pq, log_frac_pq_per_cat

# Transforms AMWC problem to MWC and computes gradients.
class AMWCBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        #(node_labels_pred, ) + (segmentation_costs, ) + edge_labels_pred + eff_affinity_costs + eff_affinity_weights + (panoptic_weights, ) + panoptic_pred_one_hot + (panoptic_ids_gt_one_hot_batch, ) + (params, )
        # Order: node_labels, node_costs (detached), tuple(edge_labels), tuple(edge_costs(detached)), tuple(edge_weights), panoptic_weights, tuple(panoptic_one_hot_pred(detached)), panoptic_ids_gt_one_hot_batch, params
        params = inputs[-1]
        batch_size = inputs[0].shape[0]
        h = inputs[0].shape[2]
        w = inputs[0].shape[3]
        padded_panoptic_one_hot_pred_batch = []
        num_padding_channels = params['max_channels']
        device = inputs[-3].device
        padded_panoptic_one_hot_pred_batch = torch.zeros((batch_size, num_padding_channels, h, w), device = device, dtype = torch.float32)
        pred_panoptic_labels = torch.zeros((batch_size, h, w), device = device, dtype = torch.int32)
        for b in range(batch_size):
            panoptic_one_hot_pred = inputs[b - batch_size - 2]
            padded_panoptic_one_hot_pred_batch[b, :panoptic_one_hot_pred.shape[0]] = panoptic_one_hot_pred
            pred_panoptic_labels[b, ::] = torch.argmax(panoptic_one_hot_pred, dim = 0)
        panoptic_ids_gt_one_hot_batch = inputs[-2]
        gt_panoptic_labels = torch.argmax(panoptic_ids_gt_one_hot_batch, dim = 1)
        ctx.params = params 
        to_save = inputs[:-2 - batch_size] + (gt_panoptic_labels, ) + (pred_panoptic_labels, )  
        ctx.save_for_backward(*to_save)
        return padded_panoptic_one_hot_pred_batch

    @staticmethod
    def backward(*inputs):
        ctx = inputs[0]
        params = ctx.params
        num_edges = len(params['edge_distances'])
        saved_items = ctx.saved_tensors
        # Unpack the long list of tensors:
        orig_node_labels = saved_items[0]
        batch_size = orig_node_labels.shape[0]
        orig_node_costs = saved_items[1]
        orig_edge_labels = saved_items[2:num_edges + 2]
        orig_edge_costs = saved_items[num_edges + 2: 2 * num_edges + 2]
        edge_weights = saved_items[2 * num_edges + 2: 3 * num_edges + 2]
        panoptic_weights = saved_items[-3]
        grad_padded_panoptic_one_hot = inputs[1]
        # print(f"grad min: {grad_padded_panoptic_one_hot.min()}, grad max: {grad_padded_panoptic_one_hot.max()}")

        assert(len(inputs) == 2)
        grad_node_labels, grad_edge_labels = AMWCBackward.backward_multiway_cut_averaging(params, grad_padded_panoptic_one_hot, orig_node_labels, orig_node_costs, orig_edge_labels, orig_edge_costs)

        return_device = orig_node_labels.device
        grad_node_labels = grad_node_labels.to(return_device) * panoptic_weights.to(return_device).unsqueeze(1)
        for e in range(len(grad_edge_labels)):
            grad_edge_labels[e] = grad_edge_labels[e].to(return_device) * edge_weights[e].to(return_device)
        
        out = (grad_node_labels, ) + (None, ) + tuple([g for g in grad_edge_labels]) + tuple([None] * len(grad_edge_labels)) + tuple([None] * len(grad_edge_labels)) + (None, ) + tuple([None] * batch_size) + (None, ) + (None, )
        return out 
        # Order: node_labels, node_costs (detached), tuple(edge_labels), tuple(edge_costs(detached)), panoptic_weights, tuple(panoptic_one_hot_pred(detached)), panoptic_ids_gt_one_hot_batch, params

    @staticmethod
    def backward_multiway_cut_averaging(params, grad_padded_panoptic_one_hot, orig_node_labels, orig_node_costs, orig_edge_labels, orig_edge_costs):
        import time
        start = time.time()

        category_indices = params['category_indices'] # Used to know from where to get semantic costs for instances.

        device = orig_node_costs.device
        batch_size = orig_node_labels.shape[0]

        instance_costs_delta = []
        original_instance_costs = []
        all_index_to_category = []
        for b in range(batch_size):
            current_grad = grad_padded_panoptic_one_hot[b].to(device)
            orig_ins_cost, index_to_category, _ = AMWCBackward.add_semantic_costs_to_instance_costs(
                                            orig_node_costs[b, ::], 
                                            current_grad,
                                            category_indices[b], 
                                            params['num_classes'], 
                                            0.0)
            ins_cost_delta, _, _ = AMWCBackward.add_semantic_costs_to_instance_costs(
                                            orig_node_costs[b, ::] * 0, 
                                            current_grad,
                                            category_indices[b], 
                                            params['num_classes'], 
                                            1.0)    
            original_instance_costs.append(orig_ins_cost)
            instance_costs_delta.append(ins_cost_delta)
            all_index_to_category.append(index_to_category)

        num_samples = params['backprop_num_samples']
        grad_node_labels, grad_edge_labels = solve_mc_grad_avg_batch(orig_node_labels, 
                                                                    orig_edge_labels, 
                                                                    original_instance_costs, 
                                                                    instance_costs_delta, 
                                                                    orig_edge_costs, 
                                                                    params['edge_indices'], 
                                                                    params['edge_distances'], 
                                                                    params['edge_sampling_intervals'], 
                                                                    all_index_to_category, 
                                                                    params['lambda_val_start'], 
                                                                    params['lambda_val_end'], 
                                                                    num_samples, 
                                                                    params['num_classes'])

        end = time.time()
        mc_time = end - start
        grad_node_labels = torch.stack(grad_node_labels, 0)
        # print(f"mc grad avg backward time: {mc_time :.1f}secs" )
        return grad_node_labels, grad_edge_labels

    @staticmethod
    def add_semantic_costs_to_instance_costs(semantic_costs, instance_costs, category_indices, num_classes, lambda_val):
        new_costs = []
        index_to_category = {} # Will help recover new node(semantic) labels.
        index_to_new_index = {} # Will help convert previous instance labels to new instance labels.
        # First find the classes which are not covered in instance_costs. For these classes the semantic cost
        # will remain unchanged. 
        i = 0
        for c in range(num_classes):
            if c in category_indices and len(category_indices[c]) > 0:
                continue # Means this class is already present in instance_costs.
            new_costs.append(semantic_costs[c, ::])
            index_to_category[i] = c
            i += 1
        
        # Now go over the rest.
        for cat_id, instance_indices in category_indices.items():
            for current_instance_idx in instance_indices:
                # Add instance_costs with lambda_val (which controls the amount of pertubation)
                full_cost = semantic_costs[cat_id, ::] + lambda_val * (instance_costs[current_instance_idx, ::].to(torch.float32)) #+ instance_costs_epsilon[current_instance_idx, ::].to(torch.float32)
                new_costs.append(full_cost)
                index_to_category[i] = cat_id
                index_to_new_index[current_instance_idx] = i 
                i += 1
        
        return torch.stack(new_costs, 0), index_to_category, index_to_new_index

    @staticmethod
    def get_semantic_labels_from_instance_labels(instance_labels, index_to_category, num_classes):
        semantic_labels = torch.zeros((num_classes, instance_labels.shape[1], instance_labels.shape[2]), dtype=torch.float32, device = instance_labels.device)
        for idx, cat in index_to_category.items():
            semantic_labels[cat, ::] += instance_labels[idx]
        return semantic_labels

# To evaluate and track panoptic quality during training.
class PanopticQualityEval():
    def __init__(self, meta, min_size_categories = 0):
        # min_size_categories is used because if the current batch contains very few pixels of 
        # a class then computing PQ on this class and then balancing it w.r.t class is going 
        # to up-weight the importance of this class. However, the issue might just be because of 
        # small batch size.

        self._metadata = meta 
        self.min_size_categories = min_size_categories
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }
        self.pq_stat_global = PQStat()
        self.categories_global = {}

    def eval_batch(self, gt_panoptic_batch, gt_segments_info_batch, pred_panoptic_batch):
        # Non-differentiable: Only for producing numbers and computing matching bw pred and gt during training. 
        pq_stat_local = PQStat()
        all_matching_costs = []
        all_ignored_preds = []
        all_fp = []
        all_fn = []
        for b in range(gt_panoptic_batch.shape[0]):
            matching_costs, fp_map, fn_map, ignored_preds = self.eval_image(gt_panoptic_batch[b,::].squeeze(), pred_panoptic_batch[b,::].squeeze(), pq_stat_local)
            all_matching_costs.append(matching_costs)
            all_ignored_preds.append(ignored_preds)
            all_fp.append(fp_map)
            all_fn.append(fn_map)

        all_fp = np.stack(all_fp, 0)
        all_fn = np.stack(all_fn, 0)

        metrics = [("All", None)] #, ("Things", True), ("Stuff", False)]
        results = {}
        valid_categories = self.compute_valid_categories(gt_panoptic_batch + 1, pred_panoptic_batch + 1)
        for name, isthing in metrics:
            results[name], per_class_results = pq_stat_local.pq_average(valid_categories, isthing=isthing)
            
            if name == 'All':
                results['per_class'] = per_class_results

            # for cat_data in per_class_results.keys():
            #     print(f"cat: {cat_data}, {per_class_results[cat_data]}")

            iou, tp, fp, fn = pq_stat_local.get_confusion(valid_categories, isthing=isthing)
            results[name]['iou'] = iou
            results[name]['tp'] = tp
            results[name]['fp'] = fp
            results[name]['fn'] = fn

        return results, all_fp, all_fn, all_matching_costs, all_ignored_preds

    def compute_global_pq_metrics(self):
        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        results = {}
        for name, isthing in metrics:
            results[name], per_class_results = self.pq_stat_global.pq_average(self.categories_global, isthing=isthing)
            
            if name == 'All':
                results['per_class'] = per_class_results

            iou, tp, fp, fn = self.pq_stat_global.get_confusion(self.categories_global, isthing=isthing)
            results[name]['iou'] = iou
            results[name]['tp'] = tp
            results[name]['fp'] = fp
            results[name]['fn'] = fn

        return results

    def compute_segments_info(self, panoptic_img):
        label_divisor = self._metadata.label_divisor
        segments_info = []
        for panoptic_label in np.unique(panoptic_img):
            if panoptic_label == -1:
                # VOID region.
                continue
            
            pred_class = panoptic_label // label_divisor
            isthing = (
                pred_class in self._thing_contiguous_id_to_dataset_id
            )
            segments_info.append(
                {
                    "id": int(panoptic_label) + 1,
                    "category_id": int(pred_class),
                    "isthing": bool(isthing),
                }
            )
        return segments_info

    def compute_valid_categories(self, gt_panoptic_batch, pred_panoptic_batch):
        pan_labels, counts = np.unique(np.concatenate((gt_panoptic_batch, pred_panoptic_batch), 0), return_counts=True)
        categories_dict = {}
        for (i, pan_l) in enumerate(pan_labels):
            if pan_l == -1:
                continue # VOID label
            pred_class = pan_l // self._metadata.label_divisor
            if counts[i] <= self.min_size_categories:
                continue
            isthing = (pred_class in self._thing_contiguous_id_to_dataset_id)
            label_info = {'isthing': isthing}
            categories_dict[pred_class] = label_info
            if pred_class not in self.categories_global:
                self.categories_global[pred_class] = label_info

        return categories_dict

    def eval_image(self, gt_panoptic, pred_panoptic, pq_stat_local):
        # Non-differentiable: Only for producing numbers during training and matching pred with gt.
        fp_map = np.zeros_like(gt_panoptic) 
        fn_map = np.zeros_like(gt_panoptic)
        segment_info_pred = self.compute_segments_info(pred_panoptic)
        gt_segments_info = self.compute_segments_info(gt_panoptic)
        # Official evaluation script uses 0 for VOID label.
        pred_panoptic = pred_panoptic + 1 
        gt_panoptic = gt_panoptic + 1 
        gt_segms = {el['id']: el for el in gt_segments_info}
        pred_segms = {el['id']: el for el in segment_info_pred}
        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in segment_info_pred)
        labels, labels_cnt = np.unique(pred_panoptic, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image, segment with ID {} is presented in PNG and not presented in JSON.'.format(label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            # if pred_segms[label]['category_id'] not in categories:
            #     raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image, the following segment IDs {} are presented in JSON and not presented in PNG.'.format(list(pred_labels_set)))

        gt_labels, gt_labels_cnt = np.unique(gt_panoptic, return_counts=True)
        for gt_label, gt_label_cnt in zip(gt_labels, gt_labels_cnt):
            if gt_label not in gt_segms:
                if gt_label == VOID:
                    continue
                raise KeyError('In the image, segment with ID {} is presented in PNG and not presented in JSON.'.format(label))
            gt_segms[gt_label]['area'] = gt_label_cnt

        # confusion matrix calculation
        gt_pred_panoptic = gt_panoptic.astype(np.uint64) * OFFSET + pred_panoptic.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(gt_pred_panoptic, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        gt_cost_counted = set()
        pred_cost_counted = set()
        matching_costs = []
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            # if gt_segms[gt_label]['iscrowd'] == 1:
            #     continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue
            
            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)

            iou = intersection / union
            if iou > 0.5:
                pq_stat_local[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat_local[gt_segms[gt_label]['category_id']].iou += iou

                self.pq_stat_global[gt_segms[gt_label]['category_id']].tp += 1
                self.pq_stat_global[gt_segms[gt_label]['category_id']].iou += iou

                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
            
            cat_id = pred_segms[pred_label]['category_id']
            matching_costs.append({'pred': int(pred_label - 1), 
                                'gt': int(gt_label - 1), 
                                'intersect': intersection, 
                                'union': union, 
                                'iou': iou, 
                                'pred_cat': cat_id,
                                'gt_cat': cat_id})

            # All segments which can be matched in the future should not immediately classified as FP/FN.
            # So storing the segments which are marked as candidates to stop them from falling into FP/FNs.
            # This can/will happen later during bipartite matching.
            if iou > 0.01:
                gt_cost_counted.add(gt_label)
                pred_cost_counted.add(pred_label)

        # count false negatives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # Not ignoring crowd segments (would be put in VOID anyway through target_generator.py if ignore_crowd is True)
            # if gt_info['iscrowd'] == 1:
            #     crowd_labels_dict[gt_info['category_id']] = gt_label
            #     continue
            pq_stat_local[gt_info['category_id']].fn += 1
            self.pq_stat_global[gt_info['category_id']].fn += 1
            fn_map[gt_panoptic == gt_info['id']] = 1 #gt_info['id']
            if gt_label not in gt_cost_counted:
                cat_id = gt_info['category_id']
                matching_costs.append({'pred':None, 
                                    'gt': int(gt_label - 1), 
                                    'intersect': 0, 
                                    'union': 0, 
                                    'iou': 0,
                                    'pred_cat': None,
                                    'gt_cat': cat_id})

        # count false positives
        ignored_preds = set()
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                ignored_preds.add(int(pred_label) - 1)
                continue
            pq_stat_local[pred_info['category_id']].fp += 1
            self.pq_stat_global[pred_info['category_id']].fp += 1

            fp_map[pred_panoptic == pred_info['id']] = 1 #pred_info['id'] 

            if pred_label not in pred_cost_counted:
                cat_id = pred_info['category_id']
                matching_costs.append({'pred': int(pred_label - 1), 
                                    'gt': None, 
                                    'intersect': 0, 
                                    'union': 0, 
                                    'iou': 0,
                                    'pred_cat': cat_id,
                                    'gt_cat': None})

        return matching_costs, fp_map, fn_map, ignored_preds


# Used to evaluate mask confidence score for instance segmentation evaluation. 
class MaskScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pixel_mask_orig_size, affinity_logits, batch_index, edge_distances, edge_sampling_intervals, logits_multiplier = 1.0):
        cut_logits = 0.0
        join_logits = 0.0
        pixel_mask = pixel_mask_orig_size.to(affinity_logits[0].device)
        pixel_mask = pixel_mask[2::4, 2::4]
        for (i, e_d) in enumerate(edge_distances):
            current_si = 1 
            if edge_sampling_intervals is not None:
                current_si = edge_sampling_intervals[i]
            
            padding_before = int(e_d / 2.0)
            padding_after = int(e_d / 2.0)

            # Central difference in rows:
            output_padding = (0, 0, padding_before, padding_after)
            if e_d == 1:
                output_padding = (0, 0, 0, 1)

            cut_edges_row = torch.abs(pixel_mask[e_d:, :] - pixel_mask[:-e_d, :])
            cut_edges_row = torch.nn.ZeroPad2d(output_padding)(cut_edges_row)

            joined_edges_row = pixel_mask[e_d:, :] + pixel_mask[:-e_d, :] == 2
            joined_edges_row = torch.nn.ZeroPad2d(output_padding)(joined_edges_row)

            # Central difference in columns:
            output_padding = (padding_before, padding_after , 0, 0)
            if e_d == 1:
                output_padding = (0, 1, 0, 0)

            cut_edges_col = torch.abs(pixel_mask[:, e_d:] - pixel_mask[:, :-e_d])
            cut_edges_col = torch.nn.ZeroPad2d(output_padding)(cut_edges_col)

            joined_edges_col = pixel_mask[:, e_d:] + pixel_mask[:, :-e_d] == 2
            joined_edges_col = torch.nn.ZeroPad2d(output_padding)(joined_edges_col)

            current_cut_edges = torch.stack((cut_edges_row, cut_edges_col), 0)
            current_joined_edges = torch.stack((joined_edges_row, joined_edges_col), 0)

            # Account for downsampling:
            current_cut_edges = current_cut_edges[:, ::current_si, ::current_si]           
            current_joined_edges = current_joined_edges[:, ::current_si, ::current_si]     

            # Now calculate cost:

            cut_logits += (logits_multiplier * affinity_logits[i][batch_index, ::] * current_cut_edges).mean() # Intra-cluster mean similarity
            join_logits += (logits_multiplier * affinity_logits[i][batch_index, ::] * current_joined_edges).mean() # Inter-cluster mean similarity

        # Mean intra-cluster similarity minus mean inter-cluster similarity. Higher value means more confidence. 
        return -1.0 * (join_logits - cut_logits)