import numpy as np
import torch
from PIL import Image
from detectron2.structures import ImageList
from skimage.util.shape import view_as_windows
from scipy import stats

class PanopticAffinityTargetGenerator(object):
    """
    Generates training targets for Panoptic-AffinityNet.
    """

    def __init__(
        self,
        ignore_label,
        thing_ids,
        label_divisor,
        num_classes,
        downsampling_factor,
        ignore_stuff_in_affinities=True,
        small_instance_area=0,
        small_instance_weight=1,
        instance_loss_weight=1,
        ignore_crowd=False,
        edge_distances = [1],
        edge_sampling_intervals = None,
        size_divisibility = -1):
        """
        Args:
            ignore_label: Integer, the ignore label for semantic segmentation.
            thing_ids: Set, a set of ids from contiguous category ids belonging
                to thing categories.
            ignore_stuff_in_affinities: Boolean, whether to ignore stuff region in computing
                instance affinities.
            small_instance_area: Integer, indicates largest area for small instances.
            small_instance_weight: Integer, indicates semantic loss weights for
                small instances.
            ignore_crowd: Boolean, whether to ignore crowd region in
                both instance and semantic segmentation branch. 
            edge_distances: List, Defines the length of edges considered in a + shaped neighbourhood
                around each pixel for computing pixel affinities. Each entry corresponds to the length of edge.
                For example: 2 means create an edge of every pixel with every other pixel 2 distance away
                in + shaped neighbourhood. So for location (x,y) in domain(I) the following neighbours: 
                (x+2, y), (x, y+2) are taken (in a forward difference fashion).
            edge_sampling_intervals: List or None. Defines the sampling interval of edges. Either it should be a list of
                length of edge_distances. None means consider sampling interval as 1 and so use all possible edges 
                in + neighbourhood.
            size_divisibility: Same as in panoptic_seg_affinity.py.
        """
        self.ignore_label = ignore_label
        self.thing_ids = set(thing_ids)
        self.ignore_stuff_in_affinities = ignore_stuff_in_affinities
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.instance_loss_weight = instance_loss_weight
        self.ignore_crowd = ignore_crowd
        self.edge_distances = edge_distances
        self.edge_sampling_intervals = edge_sampling_intervals
        self.label_divisor = label_divisor
        self.size_divisibility = size_divisibility
        self.num_classes = num_classes
        self.downsampling_factor = downsampling_factor
    
    def downsample(self, array):
        if self.downsampling_factor != 1:
            radius = self.downsampling_factor // 2
            return array[radius::self.downsampling_factor, radius::self.downsampling_factor]
            # return np.array(Image.fromarray(array).resize((array // self.downsampling_factor, array // self.downsampling_factor), resample = 0))
        return array

    def __call__(self, panoptic, segments_info):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py  # noqa
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18  # noqa
        Args:
            panoptic: numpy.array, panoptic label, we assume it is already
                converted from rgb image by panopticapi.utils.rgb2id.
            segments_info (list[dict]): see detectron2 documentation of "Use Custom Datasets".
        Returns:
            A dictionary with fields:
                - sem_seg: Tensor, semantic label, shape=(H, W).
                - sem_seg_weights: Tensor, loss weight for semantic prediction,
                    shape=(H, W).
                - aff_weights: Tensor, ignore region for affinity prediction,
                    Multiply this mask to loss.
                - affinities: List[Tensor], edge affinities 
        """
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label

        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
        panoptic_weights = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance

        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for affinity branch)
        affinity_foreground_weight_mask = np.zeros_like(panoptic, dtype=np.float32)
        filtered_panoptic = np.zeros_like(panoptic, dtype=np.int32) - 1
        instance_ids = {}
                
        for seg in segments_info:
            cat_id = seg["category_id"]
            current_gt_mask = panoptic == seg["id"]
                                                        
            # Crowd would be always ignored in affinities
            if not seg["iscrowd"]:
                panoptic_weights[current_gt_mask] = 1
                if (not self.ignore_stuff_in_affinities or cat_id in self.thing_ids):
                    affinity_foreground_weight_mask[current_gt_mask] = 1

            # For semantic (and stuff classes in panoptic) ignoring crowd would depend on the given flag. 
            if not (self.ignore_crowd and seg["iscrowd"]):
                semantic[current_gt_mask] = cat_id

                if cat_id not in self.thing_ids:
                    filtered_panoptic[current_gt_mask] = cat_id * self.label_divisor

            # Crowd would be always ignored in instance labels
            if cat_id in self.thing_ids and not seg["iscrowd"]:
                mask_index = np.where(current_gt_mask)
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue
                
                current_ins_id = instance_ids.get(cat_id, 1)
                filtered_panoptic[current_gt_mask] = cat_id * self.label_divisor + current_ins_id
                instance_ids[cat_id] = current_ins_id + 1 
                # Find instance area
                ins_area = len(mask_index[0])
                affinity_foreground_weight_mask[current_gt_mask] = self.instance_loss_weight
                if ins_area < self.small_instance_area:
                    semantic_weights[current_gt_mask] = self.small_instance_weight
                    affinity_foreground_weight_mask[current_gt_mask] = self.instance_loss_weight * self.small_instance_weight

        # Set semantic_weights to zero at ignore label:
        semantic_weights[semantic == self.ignore_label] = 0
        panoptic_weights[semantic == self.ignore_label] = 0

        # Ensure size divisibility already here for COCO-like datasets.
        if self.size_divisibility > 0:
            panoptic = ImageList.from_tensors([torch.as_tensor(panoptic.astype("long"))], size_divisibility = self.size_divisibility, pad_value = -1).tensor.squeeze().numpy()
            filtered_panoptic = ImageList.from_tensors([torch.as_tensor(filtered_panoptic)], size_divisibility = self.size_divisibility, pad_value = -1).tensor.squeeze().numpy()
            panoptic_weights = ImageList.from_tensors([torch.as_tensor(panoptic_weights)], size_divisibility = self.size_divisibility, pad_value = 0).tensor.squeeze().numpy()
            affinity_foreground_weight_mask = ImageList.from_tensors([torch.as_tensor(affinity_foreground_weight_mask)], size_divisibility = self.size_divisibility, pad_value = 1).tensor.squeeze().numpy() 
            # Model should be able to distinguish between the boundary of padding and valid image region.
        
        affinity_foreground_weight_mask = self.downsample(affinity_foreground_weight_mask)
        filtered_panoptic = self.downsample(filtered_panoptic)
        panoptic_weights = self.downsample(panoptic_weights)
        # Generate affinities and weights
        affinities = []

        # For aff_weights to be > 0 of an edge = (u, v), both u and v should be in foreground_mask and
        # both u, v are in non-crowd region. Otherwise aff_weight would be zero for that edge.
        aff_weights = [] 

        for (i, e_d) in enumerate(self.edge_distances):
            current_si = 1 
            if self.edge_sampling_intervals is not None:
                current_si = self.edge_sampling_intervals[i]
            
            padding_before = int(e_d / 2.0)
            padding_after = int(e_d / 2.0)
            output_padding = ((padding_before, padding_after), (0, 0))
            if e_d == 1:
                output_padding = ((0, 1), (0, 0))
            # Difference in rows:
            current_affinity_row = (filtered_panoptic[e_d:, :] != filtered_panoptic[:-e_d, :])

            current_affinity_row = np.pad(current_affinity_row, output_padding, mode='constant')

            assert current_affinity_row.shape == filtered_panoptic.shape

            affinity_weight_row_valid = np.logical_and(affinity_foreground_weight_mask[e_d:, :] > 0, 
                                                    affinity_foreground_weight_mask[:-e_d, :] > 0)

            current_affinity_weight_row = affinity_weight_row_valid * np.maximum(affinity_foreground_weight_mask[e_d:, :], 
                                                                                affinity_foreground_weight_mask[:-e_d, :])

            current_affinity_weight_row = np.pad(current_affinity_weight_row, output_padding, mode='constant')
            assert current_affinity_weight_row.shape == filtered_panoptic.shape
            # Difference in columns:
            output_padding = ((0, 0), (padding_before, padding_after))
            if e_d == 1:
                output_padding = ((0, 0), (0, 1))

            current_affinity_col = (filtered_panoptic[:, e_d:] != filtered_panoptic[:, :-e_d])

            current_affinity_col = np.pad(current_affinity_col, output_padding, mode='constant')
            assert current_affinity_col.shape == filtered_panoptic.shape

            affinity_weight_col_valid = np.logical_and(affinity_foreground_weight_mask[:, e_d:] > 0, 
                                                    affinity_foreground_weight_mask[:, :-e_d] > 0)

            current_affinity_weight_col = affinity_weight_col_valid * np.maximum(affinity_foreground_weight_mask[:, e_d:], 
                                                                                affinity_foreground_weight_mask[:, :-e_d])

            current_affinity_weight_col = np.pad(current_affinity_weight_col, output_padding, mode='constant')
            assert current_affinity_weight_col.shape == filtered_panoptic.shape

            current_affinity = np.stack((current_affinity_row, current_affinity_col), 0)

            current_affinity = current_affinity[:, ::current_si, ::current_si]           
            affinities.append(torch.from_numpy(current_affinity))

            current_affinity_weight = np.stack((current_affinity_weight_row, current_affinity_weight_col), 0)
            current_affinity_weight = current_affinity_weight[:, ::current_si, ::current_si]

            aff_weights.append(torch.from_numpy(current_affinity_weight))

        filtered_panoptic[panoptic_weights == 0] = -1

        return dict(
            sem_seg=torch.as_tensor(semantic.astype("long")),
            sem_seg_weights=torch.as_tensor(semantic_weights.astype(np.uint8)),
            affinity_weights=aff_weights,
            affinities=affinities,
            panoptic=torch.as_tensor(filtered_panoptic),
            segments_info=segments_info,
            panoptic_weights=torch.as_tensor(panoptic_weights.astype(np.float32))
        )