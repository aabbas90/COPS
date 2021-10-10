#!/usr/bin/env python3
"""
Panoptic-affinity Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch
from typing import List, Set

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.layers.batch_norm import FrozenBatchNorm2d
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from torch.nn.modules.container import ModuleList
from detectron2.projects.deeplab import build_lr_scheduler
from panoptic_affinity import (
    PanopticAffinityDatasetMapper,
    add_panoptic_affinity_config,
)

from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping

from detectron2.engine import HookBase

# For model stats:
import numpy as np
from collections import Counter
import tqdm
from detectron2.data import build_detection_test_loader
import logging
logger = logging.getLogger("detectron2")
from detectron2.utils.analysis import (
    activation_count_operators,
    flop_count_operators,
    parameter_count_table,
)

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    # if cfg.INPUT.CROP.ENABLED:
    #     augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    return augs

class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.defrost()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self.cfg.freeze()
        self._loader = iter(self.build_train_loader(self.cfg))
        self.num_steps = 0
    
    def build_train_loader(cls, cfg):
        mapper = PanopticAffinityDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    def after_step(self):
        self.num_steps += 1
        if self.cfg.VAL_LOSS_PERIOD > 0 and self.num_steps % self.cfg.VAL_LOSS_PERIOD == 0:
            panoptic_val_loss = 0
            num_runs = 5
            for _ in range(num_runs):
                data = next(self._loader)
                with torch.no_grad():
                    # self.trainer.model.amwc_layer.eval_pq_train = True
                    self.trainer.model.amwc_layer.validation_mode = True
                    loss_dict = self.trainer.model(data)
                    self.trainer.model.amwc_layer.validation_mode = False
                    
                    losses = sum(loss_dict.values())
                    assert torch.isfinite(losses).all(), loss_dict

                    panoptic_val_loss += losses / self.trainer.model.amwc_layer.amc_loss_weight

            panoptic_val_loss /= num_runs
            if comm.is_main_process():
                self.trainer.storage.put_scalar('val_loss_avg', panoptic_val_loss)
        else:
            pass

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type == "coco_panoptic_seg":
            # `thing_classes` in COCO panoptic metadata includes both thing and
            # stuff classes for visualization. COCOEvaluator requires metadata
            # which only contains thing classes, thus we map the name of
            # panoptic datasets to their corresponding instance datasets.
            dataset_name_mapper = {
                "coco_2017_val_panoptic": "coco_2017_val",
                "coco_2017_val_100_panoptic": "coco_2017_val_100",
                "coco_2017_test_panoptic": "coco_2017_test_panoptic",
            }
            evaluator_list.append(
                COCOEvaluator(dataset_name_mapper[dataset_name], output_dir=output_folder)
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PanopticAffinityDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """               
        if cfg.MODEL.FREEZE_BN:
            # Freeze batch norm during fine-tuning. 
            # Detectron2 has checkpoint versioning issues which leads to batchnorm params frozen incorrectly.
            # Hack: Currently we comment out -= eps in detectron2. 
            model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
        
        lr_red_factor = cfg.MODEL.LEARNING_RATE_REDUCTION_PER_STAGE
        params = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        def set_trainable(trainable, lr, lr_reduction):
            nonlocal params
            nonlocal memo
            if lr_reduction == 0:
                lr_reduction = 1 # no reduction.
            if isinstance(trainable, torch.nn.Module):
                for key, value in trainable.named_parameters():
                    value.requires_grad = True
                    params += [{"params": [value], "lr": lr / lr_reduction, "weight_decay": 0.0}]
                    assert value not in memo
                    memo.add(value)       
            else:
                trainable.requires_grad = True
                params += [{"params": [trainable], "lr": lr / lr_reduction, "weight_decay": 0.0}]
                assert trainable not in memo
                memo.add(trainable)       

        def set_untrainable(trainable):
            if isinstance(trainable, torch.nn.Module):
                for key, value in trainable.named_parameters():
                    value.requires_grad = False
                    assert value not in memo
            else:
                trainable.requires_grad = False
                assert trainable not in memo

        if cfg.MODEL.PANOPTIC_AFFINITY.AMC_LOSS_WEIGHT == 0:
            set_untrainable(model.amwc_layer)
            
        if cfg.MODEL.FINE_TUNING_MODE:
            set_untrainable(model.backbone)

            assert (cfg.MODEL.FREEZE_SEM_SEG_TILL >= -1 and 
                    cfg.MODEL.FREEZE_SEM_SEG_TILL <= 4)
                    
            assert (cfg.MODEL.FREEZE_EDGE_COSTS_TILL >= -1 and 
                    cfg.MODEL.FREEZE_EDGE_COSTS_TILL <= 5)

            if cfg.MODEL.FREEZE_SEM_SEG_TILL == -1:
                set_trainable(model.sem_seg_head, cfg.SOLVER.BASE_LR, 1.0)

            else:
                # fine-tuning mode:
                set_untrainable(model.sem_seg_head)
                set_untrainable(model.amwc_layer.seg_cost_weights)
                sem_costs_trainables = [
                    model.sem_seg_head.decoder,
                    model.sem_seg_head.head,
                    model.sem_seg_head.predictor,
                    model.amwc_layer.seg_cost_weights
                ]
                for (i, t) in enumerate(sem_costs_trainables):
                    if i + 1 > cfg.MODEL.FREEZE_SEM_SEG_TILL:
                        current_lr_reduction = lr_red_factor * (len(sem_costs_trainables) - i - 1)
                        set_trainable(t, cfg.SOLVER.BASE_LR, current_lr_reduction)

            if cfg.MODEL.FREEZE_EDGE_COSTS_TILL == -1:
                set_trainable(model.AFF_EMBED_head, cfg.SOLVER.BASE_LR, 1.0)

            else:
                # fine-tuning mode:
                set_untrainable(model.AFF_EMBED_head)
                set_untrainable(model.amwc_layer.edge_costs_weights)
                # To not learn the class specific areas:
                set_untrainable(model.amwc_layer.training_per_class_areas)
                # To learn class specific areas:
                # set_trainable(model.amwc_layer.training_per_class_areas, cfg.SOLVER.BASE_LR, 0.05)
                train_edge_start = max(cfg.MODEL.TRAIN_EDGE_INDEX_START, 0)
                train_edge_end = min(cfg.MODEL.TRAIN_EDGE_INDEX_END, len(model.AFF_EMBED_head.classifiers))

                edge_costs_trainables = [
                    model.AFF_EMBED_head.decoder,
                    model.AFF_EMBED_head.affinity_head,
                    torch.nn.ModuleList([c[:-1] for c in model.AFF_EMBED_head.classifiers[train_edge_start:train_edge_end]]),
                    torch.nn.ModuleList([c[-1] for c in model.AFF_EMBED_head.classifiers[train_edge_start:train_edge_end]]),
                    model.amwc_layer.edge_costs_weights[2 * train_edge_start: 2 * train_edge_end]
                ]
                for (i, t) in enumerate(edge_costs_trainables):
                    if i + 1 > cfg.MODEL.FREEZE_EDGE_COSTS_TILL:
                        current_lr_reduction = lr_red_factor * (len(edge_costs_trainables) - i - 1)
                        set_trainable(t, cfg.SOLVER.BASE_LR, current_lr_reduction)

            print("Training only the following parameters:")
            print([name for name, x in model.named_parameters() if x.requires_grad])

        if len(params) == 0:
            params = get_default_optimizer_params(
                model,
                base_lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
                bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS
            )
            print("Using default learning rate policy.")
            
        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params = params, 
                                                                    lr = cfg.SOLVER.BASE_LR, 
                                                                    betas = (cfg.SOLVER.BETA1_ADAM, 0.999), 
                                                                    eps = cfg.SOLVER.EPSILON_ADAM)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_affinity_config(cfg)
    if (hasattr(args, 'base_config_file')) and os.path.exists(args.base_config_file):
        cfg.merge_from_file(args.base_config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def do_flop(cfg):
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(10), data_loader):  # noqa
        count = flop_count_operators(model, data)
        counts += count
        total_flops.append(sum(count.values()))
    logger.info(
        "(G)Flops for Each Type of Operators:\n" + str([(k, v / idx) for k, v in counts.items()])
    )
    logger.info("Total (G)Flops: {}±{}".format(np.mean(total_flops), np.std(total_flops)))


def main(args):
    cfg = setup(args)
    # do_flop(cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        # logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=2))
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        print("Used weights: {}".format(cfg.MODEL.WEIGHTS))
        return res

    trainer = Trainer(cfg)
    val_loss = ValidationLoss(cfg)  
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--base-config-file", default="", metavar="FILE", help="path to _BASE_ config file")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
