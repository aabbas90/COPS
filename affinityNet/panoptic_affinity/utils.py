import numpy as np 
import os, sys
from detectron2.data.detection_utils import convert_image_to_rgb
from torch._C import dtype
from torch.nn import functional as F
import torch 
from scipy.optimize import linear_sum_assignment
from detectron2.utils.events import get_event_storage
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.visualizer import Visualizer
from PIL import Image 
from collections import defaultdict
import torch.multiprocessing as mp
import matplotlib.pyplot as plt 
import matplotlib
    
class DropoutPixels(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super(DropoutPixels, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            return X * binomial.sample([X.shape[0], 1, X.shape[2], X.shape[3]]).to(X.device)
        return X

def save_gif(image_tensor, path, vmin = None, vmax = None, cmap = None):
    from PIL import Image
    if torch.is_tensor(image_tensor):
        if len(image_tensor.shape) == 2:
            image_list = [image_tensor]
        else:
            image_list = torch.unbind(image_tensor)
    else:
        image_list = image_tensor
    processed = []
    for im in image_list:
        im = im.detach().cpu().numpy()
        if cmap is None:
            if vmin is None:
                vmin = im.min()
            if vmax is None:
                vmax = im.max()
            im = 255 * (im - vmin) / (vmax - vmin + 1e-6)
            im[im < 0] = 0
            im[im > 255] = 255
        else:
            assert(vmin is None)
            im = apply_segmented_cmap(im, cmap, vmax)

        im = Image.fromarray(im.astype(np.uint8))
        processed.append(im)
    
    if len(processed) > 1:
        processed[0].save(path + '.gif', save_all=True, append_images=processed[1:], duration=2000, loop=0, disposal = 1)
    else:
        processed[0].save(path + '.png')
        
def save_segmentation(image, path, max_v = None):
    image_cmap = apply_segmented_cmap(image, plt.cm.tab20, max_v=max_v)
    from PIL import Image
    im = Image.fromarray(image_cmap)
    im.save(path + '.png')

def GetSolutionCost(node_costs, node_labels, edge_costs, edge_labels, b):
    cost = (node_costs * node_labels).sum()
    for (c, l) in zip(edge_costs, edge_labels):
        cost += (c[b] * l[b]).sum()
    return cost

def GetColoredEdgeImage(row_costs, col_costs):
    img = np.zeros((col_costs.shape[0], row_costs.shape[1], 3))
    img[:row_costs.shape[0], :row_costs.shape[1], 0] = row_costs
    img[:col_costs.shape[0], :col_costs.shape[1], 1] = col_costs
    img = 255.0 * (img - img.min()) / (img.max() - img.min())
    return img.astype(np.uint8)

def ComputeIDMapping(array1, array2, offset = 256 * 256 * 256):
    combined = array1.astype(np.uint64) * offset + array2.astype(np.uint64)
    mapping = {}
    labels = np.unique(combined)
    for label in labels:
        array1_id = int(label // offset)
        array2_id = int(label % offset)
        mapping[array1_id] = array2_id
    return mapping

def CreateSharedArrayFromTensor(tensor):
    tensor_numpy = tensor.numpy()
    X = mp.RawArray('f', tensor.numel())
    print(tensor.dtype)
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X).reshape(tensor.shape)
    # Copy data to our shared array.
    np.copyto(X_np, tensor_numpy)
    return X_np

def ComputeBestGroundTruthBatch(panoptic_ids_pred_one_hot, index_to_class_labels, panoptic_images_gt, panoptic_weights, thing_ids, label_divisor, num_classes, similarity_func):
    # ctx = mp.get_context('fork')
    # manager = ctx.Manager()
    # return_dict = manager.dict()
    # workers = []
    return_dict = {}
    batch_size = len(panoptic_ids_pred_one_hot)
    for b in range(batch_size):
        args = (b, panoptic_ids_pred_one_hot[b], index_to_class_labels[b].numpy(), panoptic_images_gt[b, ::], panoptic_weights[b, ::], thing_ids, label_divisor, num_classes, similarity_func, return_dict)
#        if batch_size == 1:
        ComputeBestGroundTruth(*args)
        # else:
        #     worker = ctx.Process(target=ComputeBestGroundTruth, args=args)
        #     workers.append(worker)

    # if batch_size > 1:
    #     [w.start() for w in workers]  
    #     for worker in workers:
    #         worker.join()
    #         if worker.exitcode != 0:
    #             print(f"ERROR: There was an error during multiprocessing ComputeBestGroundTruth with error code: {worker.exitcode}, in worker: {worker.name}.")
    #             sys.exit(0)

    category_indices_gt_batch = []
    max_channels = 0
    num_valid_channels_gt = []
    for b in sorted(return_dict.keys()):
        panoptic_ids_gt_one_hot, category_indices_gt = return_dict[b]
        category_indices_gt_batch.append(category_indices_gt)
        max_channels = max(max_channels, panoptic_ids_gt_one_hot.shape[0])
        num_valid_channels_gt.append(panoptic_ids_gt_one_hot.shape[0])

    h = panoptic_ids_pred_one_hot[0].shape[1]
    w = panoptic_ids_pred_one_hot[0].shape[2]
    panoptic_ids_gt_one_hot_batch = torch.zeros((batch_size, max_channels, h, w), device = panoptic_ids_pred_one_hot[0].device, dtype=torch.float32)
    num_padding_channels_pred = []
    for b in sorted(return_dict.keys()):
        panoptic_ids_gt_one_hot, _ = return_dict[b]
        panoptic_ids_gt_one_hot_batch[b, :num_valid_channels_gt[b], ::] = panoptic_ids_gt_one_hot
        num_padding_channels_pred.append(max_channels - panoptic_ids_pred_one_hot[b].shape[0])

    return panoptic_ids_gt_one_hot_batch, category_indices_gt_batch, num_valid_channels_gt, num_padding_channels_pred

def ComputeBestGroundTruth(batch_index, panoptic_image_one_hot_pred, index_to_class_labels_pred, panoptic_image_gt, panoptic_weights, thing_ids, label_divisor, num_classes, similarity_func, return_dict):
    class_labels_to_indices_pred = defaultdict(list)
    for ind in range(index_to_class_labels_pred.shape[0]):
        class_id = index_to_class_labels_pred[ind]
        class_labels_to_indices_pred[class_id].append(ind)
    
    panoptic_image_one_hot_gt = torch.zeros_like(panoptic_image_one_hot_pred)
    gt_not_in_pred = []
    category_indices_gt = defaultdict(list)
    extra_index = panoptic_image_one_hot_gt.shape[0]
    for class_id in range(num_classes):
        current_seg_mask = (panoptic_image_gt // label_divisor) == class_id 
        if class_id not in thing_ids:
            # Stuff class, does not need matching:
            if class_id in class_labels_to_indices_pred:
                index_to_place = class_labels_to_indices_pred[class_id]
                assert(len(index_to_place) == 1)
                panoptic_image_one_hot_gt[index_to_place[0], ::] = current_seg_mask #False positives are also handled here as gt_mask would be 0.
                category_indices_gt[class_id].append(index_to_place[0])
            elif torch.any(current_seg_mask): # GT present, but prediction is empty:
                gt_not_in_pred.append(current_seg_mask)
                category_indices_gt[class_id].append(extra_index)
                extra_index += 1
        else:
            gt_instance_ids = torch.unique(panoptic_image_gt[current_seg_mask])
            gt_masks = []
            for gt_id in gt_instance_ids:
                gt_masks.append(panoptic_image_gt == gt_id)
            if class_id in class_labels_to_indices_pred: # Use bipartite matching to match pred instance to gt instances:
                pred_masks = panoptic_image_one_hot_pred[class_labels_to_indices_pred[class_id], ::]
                best_gt_masks, unmatched_gt_indices = MatchMasksPair(pred_masks, gt_masks, panoptic_weights, similarity_func)
                panoptic_image_one_hot_gt[class_labels_to_indices_pred[class_id], ::] = best_gt_masks
                category_indices_gt[class_id].extend(class_labels_to_indices_pred[class_id])
                for i in unmatched_gt_indices:
                    gt_not_in_pred.append(gt_masks[i])
                    category_indices_gt[class_id].append(extra_index)
                    extra_index += 1
            elif len(gt_masks) > 0: # GT present, but prediction is empty:
                for gt_mask in gt_masks:
                    gt_not_in_pred.append(gt_mask)
                    category_indices_gt[class_id].append(extra_index)
                    extra_index += 1
    
    if len(gt_not_in_pred) > 0:
        gt_not_in_pred = torch.stack(gt_not_in_pred) # False negatives, for these predictions will be padded later.
        panoptic_image_one_hot_gt = torch.cat((panoptic_image_one_hot_gt, gt_not_in_pred))

    return_dict[batch_index] = panoptic_image_one_hot_gt, category_indices_gt
    
def MatchMasksPair(pred_masks, gt_masks_list, weight, similarity_func):
    M = pred_masks.shape[0]
    N = len(gt_masks_list)
    best_gt = torch.zeros_like(pred_masks)
    if N == 0:
        return best_gt, set()

    gt_masks = torch.stack(gt_masks_list)
    similarity_matrix = torch.zeros((M, N), device = pred_masks.device)
    weight_b = weight.unsqueeze(0)
    for i in range(M):
        current_x = pred_masks[i:i + 1, ::] # add a fake channel dimension.
        similarity_matrix[i, :] = similarity_func(current_x, gt_masks, weight_b, [1, 2])

    similarity_matrix = similarity_matrix.cpu().numpy()
    x_indices, y_indices = linear_sum_assignment(similarity_matrix, maximize=True)
    unmatched_gt_indices = set(range(N))
    for i in range(M):
        matched_loc = np.where(i == x_indices)[0]
        if len(matched_loc) == 1: # Otherwise gt will remain 0
            matched_gt_i = y_indices[matched_loc[0]]
            unmatched_gt_indices.remove(matched_gt_i)
            best_gt[i, ::] = gt_masks[matched_gt_i, ::]

    return best_gt, unmatched_gt_indices

def ComputeMulticutCosts(sem_seg_logits, aff_logits):
    aff_costs = []
    for i in range(len(aff_logits)):
        aff_costs.append(-1.0 * aff_logits[i])

    if sem_seg_logits != {}:
        return -1.0 * sem_seg_logits, aff_costs 
    else:
        return {}, aff_costs

def get_numpy_image(torch_image):
    return (torch_image).cpu().detach().numpy().squeeze()

def convert_to_three_channels(numpy_image):
    return np.stack((numpy_image,)*3, axis=-1)

def interpolate_image(input_im, desired_shape):
    return F.interpolate(input_im[None].to(torch.float32), size=desired_shape, mode="bilinear", align_corners=False).squeeze(0)

def interpolate_mask(mask, original_image_size, output_height, output_width):
    mask = mask[:, :original_image_size[0], :original_image_size[1]].expand(1, -1, -1, -1)
    return F.interpolate(mask.to(torch.float32), size=(output_height, output_width), mode="nearest").squeeze(0)

def convert_seg_mask_to_one_hot(sem_seg_target, sem_seg_ignore_val, num_classes):
    node_labels_gt = sem_seg_target.detach().clone() # This target was used in computing sem_seg loss, so do not modify it in-place:
    ignore_mask = node_labels_gt == sem_seg_ignore_val
    node_labels_gt[ignore_mask] = 0
    node_labels_gt = torch.nn.functional.one_hot(node_labels_gt, num_classes)
    node_labels_gt = node_labels_gt.permute(0, 3, 1, 2).to(torch.uint8)
    node_labels_gt[ignore_mask.unsqueeze(1).repeat((1, num_classes, 1, 1))] = 0
    return node_labels_gt

def save_amc_instance(input_format, amc_instances_dir, segmentation_costs, affinity_costs, sem_seg_targets, edge_labels_gt, batched_inputs, sem_seg_weights, aff_weights, sem_seg_ignore_val, num_classes):
    for sc, af, seg_gt, el_gt, input_img, sw, af_mask in zip(
        segmentation_costs, affinity_costs, sem_seg_targets, edge_labels_gt, batched_inputs, sem_seg_weights, aff_weights):

        img = input_img["image"]
        img_original = convert_image_to_rgb(img.permute(1, 2, 0), input_format).astype("uint8")

        nl_gt = convert_seg_mask_to_one_hot(seg_gt.unsqueeze(1), sem_seg_ignore_val, num_classes)
        
        image_path = input_img['file_name'].split(os.sep)
        image_name = os.path.splitext(image_path[-1])[0]

        ins = {'segmentation_costs': sc, 
                'affinity_costs': af, 
                'node_labels_gt': nl_gt,
                'edge_labels_gt': el_gt,
                'orig_img': img_original,
                'sem_seg_weights': sw, 
                'aff_weights': af_mask
                }

        torch.save(ins, os.path.join(amc_instances_dir, image_name))
        print("Saved: {}".format(os.path.join(amc_instances_dir, image_name)))

def load_amc_instance(ins_path):
    ins = torch.load(ins_path)
    segmentation_costs = ins['segmentation_costs'].unsqueeze(0).cpu().detach().numpy()
    affinity_costs = ins['affinity_costs'].unsqueeze(0).cpu().detach().numpy()
    node_labels_gt = ins['node_labels_gt'].unsqueeze(0).cpu().detach().numpy()
    edge_labels_gt = ins['edge_labels_gt'].unsqueeze(0).cpu().detach().numpy()
    orig_img = ins['orig_img']
    aff_weights = ins['aff_weights'].unsqueeze(0).cpu().detach().numpy()
    sem_seg_weights = ins['sem_seg_weights'].unsqueeze(0).cpu().detach().numpy()
    return (segmentation_costs, 
            affinity_costs,
            node_labels_gt, 
            edge_labels_gt, 
            orig_img, 
            aff_weights, 
            sem_seg_weights)

# Generate random colormap
def rand_cmap(nlabels, type='soft', first_color_black=True, last_color_black=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap

def make_ids_contiguous(seg_mask, ids_image, min_component_size = 0):
    new_ids_image = np.zeros_like(ids_image)
    category_ids = np.unique(seg_mask)
    for current_class in category_ids:
        class_mask = seg_mask == current_class
        unique_ids, counts = np.unique(ids_image[class_mask].flatten(), return_counts=True)
        # Re-rank instance ids to contiguous ids to ensure that they are < 1000.
        new_id = 1
        for (old_id_count, old_id) in zip(counts, unique_ids):
            if old_id == 0:
                continue
            
            if old_id_count > min_component_size:
                valid_mask = np.logical_and(ids_image == old_id, class_mask)
                new_ids_image[valid_mask] = new_id
                new_id = new_id + 1
    return new_ids_image, new_id - 1

def apply_segmented_cmap(image_array, cm = None, max_v = None):
    if torch.is_tensor(image_array):
        image_array = image_array.cpu().detach().numpy()

    if max_v is None:
        max_v = max(image_array.max(), 1)
        
    if cm is None:
        cm = rand_cmap(max(int(max_v), 2))
    image_array = image_array / max_v
    image_array_colored = cm(image_array)
    image_array_colored = (image_array_colored * 255).astype(np.uint8)
    return image_array_colored

def get_affinity_image(affinity):
    color_channels = 3 * [torch.zeros((affinity.shape[0], affinity.shape[2], affinity.shape[3]), dtype=torch.uint8) + 255]
    cmap = matplotlib.cm.get_cmap('Spectral')
    row_color = cmap(0.1)[:3]
    col_color = cmap(0.9)[:3]
    row_edge = affinity[:, 0, ::] > 0
    col_edge = affinity[:, 1, ::] > 0
    for (i, c) in enumerate(color_channels):
        c[row_edge] = row_color[i]
        c[col_edge] = col_color[i]
    return torch.stack(color_channels, -1)
    
def get_vis_img(image_array, border = 20):
    if torch.is_tensor(image_array):
        image_array = image_array.cpu().detach().numpy()
    image_array = image_array.squeeze()
    max_v = image_array[border:-border, border:-border].max()
    min_v = image_array[border:-border, border:-border].min()
    if min_v == max_v:
        max_v = min_v + 1

    img_int = (255.0 * ((image_array - min_v) / (max_v - min_v))).astype(np.uint8)
    if len(img_int.shape) == 2:
        return convert_to_three_channels(img_int)
    else:
        return img_int
        
def visualize_training(input_format, meta, seg_targets, segmentation_logits, aff_costs, node_labels_pred, edge_labels_pred, edge_labels_gt, aff_weights, batched_inputs, image_sizes):
    storage = get_event_storage()
    all_images = []
    segmentation_targets_vis = []

    for b, (seg_target, seg_logit, node_label_pred, input_per_image, orig_image_size) in enumerate(zip(
        seg_targets, segmentation_logits, node_labels_pred, batched_inputs, image_sizes)):
        img = input_per_image["image"]
        height = input_per_image.get("height")
        width = input_per_image.get("width")

        img = convert_image_to_rgb(img.permute(1, 2, 0), input_format).astype("uint8")
        img = np.array(Image.fromarray(img).resize((width, height)))
        seg_target = F.interpolate(seg_target[:orig_image_size[0], :orig_image_size[1]].expand(1,1,-1,-1).to(torch.float32), size=(height, width), mode="nearest").squeeze().to(torch.long)
        seg_logit = sem_seg_postprocess(seg_logit, orig_image_size, height, width)
        node_label_pred = sem_seg_postprocess(node_label_pred.to(torch.float32), orig_image_size, height, width)
        
        seg_mask = get_numpy_image(seg_logit.argmax(dim=0))
        
        v_gt = Visualizer(img, meta)
        v_gt = v_gt.draw_sem_seg(get_numpy_image(seg_target))
        segmentation_img_target = v_gt.get_image() 

        v_gt = Visualizer(img, meta)
        v_gt = v_gt.draw_sem_seg(seg_mask)
        segmentation_img_pred = v_gt.get_image() 

        v_gt = Visualizer(img, meta)
        v_gt = v_gt.draw_sem_seg(get_numpy_image(node_label_pred.argmax(dim=0)), draw_text = False)
        segmentation_img_pred_amc = v_gt.get_image()

        edge_targets_row = []
        edge_affinities_row = []
        edge_preds_row = []
        edge_targets_col = []
        edge_affinities_col = []
        edge_preds_col = []
        for i in range(len(edge_labels_gt)):
            def resize_and_postprocess(img_tensor, orig_image_size, height, width):
                image = F.interpolate(img_tensor.expand(1,1,-1,-1), size=(height, width), mode="nearest").squeeze(0)
                return sem_seg_postprocess(image, orig_image_size, height, width)
            
            row_label_gt = resize_and_postprocess(edge_labels_gt[i][b, 0:1, ::].to(torch.float32), orig_image_size, height, width)
            col_label_gt = resize_and_postprocess(edge_labels_gt[i][b, 1:2, ::].to(torch.float32), orig_image_size, height, width)

            row_label_pred = resize_and_postprocess(edge_labels_pred[i][b, 0:1, ::].to(torch.float32), orig_image_size, height, width)
            col_label_pred = resize_and_postprocess(edge_labels_pred[i][b, 1:2, ::].to(torch.float32), orig_image_size, height, width)
            
            row_cost = resize_and_postprocess(aff_costs[i][b, 0:1, ::], orig_image_size, height, width)
            col_cost = resize_and_postprocess(aff_costs[i][b, 1:2, ::], orig_image_size, height, width)

            edge_targets_row.append(convert_to_three_channels(get_numpy_image(row_label_gt)))
            row_cost = get_numpy_image(row_cost)
            row_cost = (row_cost - row_cost.min()) / (row_cost.max() - row_cost.min())
            edge_affinities_row.append(convert_to_three_channels(row_cost))
            edge_preds_row.append(convert_to_three_channels(get_numpy_image(row_label_pred)))

            edge_targets_col.append(convert_to_three_channels(get_numpy_image(col_label_gt)))
            col_cost = get_numpy_image(col_cost)
            col_cost = (col_cost - col_cost.min()) / (col_cost.max() - col_cost.min())
            edge_affinities_col.append(convert_to_three_channels(col_cost))
            edge_preds_col.append(convert_to_three_channels(get_numpy_image(col_label_pred)))

        edge_targets_row = np.concatenate(edge_targets_row, axis=1)
        edge_affinities_row = np.concatenate(edge_affinities_row, axis=1)
        edge_preds_row = np.concatenate(edge_preds_row, axis=1)
        all_edge_info_row = np.concatenate((edge_affinities_row, edge_preds_row, edge_targets_row), axis = 0).transpose(2, 0, 1)
        storage.put_image("Row edges: (aff, pred, gt), " + str(b), all_edge_info_row)

        edge_targets_col = np.concatenate(edge_targets_col, axis=1)
        edge_affinities_col = np.concatenate(edge_affinities_col, axis=1)
        edge_preds_col = np.concatenate(edge_preds_col, axis=1)
        all_edge_info_col = np.concatenate((edge_affinities_col, edge_preds_col, edge_targets_col), axis = 0).transpose(2, 0, 1)
        storage.put_image("Col edges: (aff, pred, gt), " + str(b), all_edge_info_col)

        targets = [get_numpy_image(edge_labels_gt[0][b, 0:1, ::]), get_numpy_image(edge_labels_gt[0][b, 1:2, ::])]
        v_gt = Visualizer(img, None)
        v_gt = v_gt.draw_heatmaps(targets, [0.9, 0.9], ['Reds', 'Reds'])
        targets_img_aff = v_gt.get_image()

        predictions_aff = [get_numpy_image(edge_labels_pred[0][b, 0:1, ::].to(torch.float32)), get_numpy_image(edge_labels_pred[0][b, 1:2, ::].to(torch.float32))]
        v_gt = Visualizer(img, None)
        v_gt = v_gt.draw_heatmaps(predictions_aff, [0.9, 0.9], ['Reds', 'Reds'])
        predictions_img_aff = v_gt.get_image()

        all_images.append(np.concatenate((targets_img_aff, predictions_img_aff), axis=1))
        segmentation_targets_vis.append(np.concatenate((segmentation_img_target, segmentation_img_pred, segmentation_img_pred_amc), axis=1))
        break  # visualize 1 image in a batch
    all_images = np.concatenate(all_images, axis=0).transpose(2, 0, 1)
    segmentation_targets_vis = np.concatenate(segmentation_targets_vis, axis=0).transpose(2, 0, 1)
    storage.put_image("GT_E, PRED_E", all_images)
    storage.put_image("GT_SEG, PRED_SEG_SEP, PRED_SEG_AMC", segmentation_targets_vis)

