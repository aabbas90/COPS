# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN

def add_panoptic_affinity_config(cfg):
    """
    Add config for Panoptic-Affinity.
    """
    ###################### SEMANTIC SEGMENTATION CONFIG FROM DEEPLAB ##################
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Used for `poly` learning rate schedule.
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    # Loss type, choose from `cross_entropy`, `hard_pixel_mining`.
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"
    # DeepLab settings
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV = True

    # Backbone new configs
    cfg.MODEL.RESNETS.RES4_DILATION = 1
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 2, 4]
    # ResNet stem type from: `basic`, `deeplab`
    cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"

    # Fine-tuning related parameters:
    cfg.MODEL.TRAIN_EDGE_INDEX_START = 0
    cfg.MODEL.TRAIN_EDGE_INDEX_END = 6
    cfg.MODEL.FINE_TUNING_MODE = False # Corresponds to end-to-end training.

    # freeze till (and including) only if cfg.MODEL.FINE_TUNING_MODE is True:
    cfg.MODEL.FREEZE_SEM_SEG_TILL = 0 # Possibilities -1: (train from scratch), 0: (all train but with LR reduction),  1: decoder, 2: head, 3: predictor, 4: all freeze
    cfg.MODEL.FREEZE_EDGE_COSTS_TILL = 0 # Possibilities -1: (train from scratch), 0: (all train but with LR reduction),  1: decoder, 2: head, 3: classifier, 4: classifier last layer, 5: all freeze
    cfg.MODEL.LEARNING_RATE_REDUCTION_PER_STAGE = 20.0

    ############################ ADDITIONAL CONFIG ############################
    # Target generation parameters.
    cfg.INPUT.IGNORE_STUFF_IN_AFFINITIES = False
    cfg.INPUT.SMALL_INSTANCE_AREA = 4096
    cfg.INPUT.SMALL_INSTANCE_WEIGHT = 3
    cfg.INPUT.INSTANCE_LOSS_WEIGHT = 1.0
    cfg.INPUT.IGNORE_CROWD = False

    cfg.MODEL.SIZE_DIVISIBILITY = 0
    # Optimizer type.
    cfg.SOLVER.OPTIMIZER = "ADAM"
    cfg.SOLVER.BETA1_ADAM = 0.9
    cfg.SOLVER.EPSILON_ADAM = 1e-8

    # Semantic segmentation head:
    # We add an extra convolution before predictor similar to PDL.
    cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K = 0.2

    # Instance segmentation head:
    cfg.MODEL.AFF_EMBED_HEAD = CN()
    cfg.MODEL.AFF_EMBED_HEAD.NAME = "PanopticAffinityInsEmbedHead"
    cfg.MODEL.AFF_EMBED_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
    cfg.MODEL.AFF_EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
    cfg.MODEL.AFF_EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
    cfg.MODEL.AFF_EMBED_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.AFF_EMBED_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.AFF_EMBED_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.AFF_EMBED_HEAD.HEAD_CHANNELS = 32
    cfg.MODEL.AFF_EMBED_HEAD.CONVS_DIM = 128
    cfg.MODEL.AFF_EMBED_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.AFF_EMBED_HEAD.NORM = "SyncBN"
    cfg.MODEL.AFF_EMBED_HEAD.AFF_LOSS_WEIGHT = 10.0
    cfg.MODEL.AFF_EMBED_HEAD.EDGE_DISTANCES = [1, 4, 16, 32] # Are relative to image grid downsampled by 4.
    cfg.MODEL.AFF_EMBED_HEAD.EDGE_SAMPLING_INTERVALS = [1, 1, 1, 1] # Are relative to image grid downsampled by 4.
    cfg.MODEL.AFF_EMBED_HEAD.LOSS_TOP_K = 1.0

    cfg.MODEL.AMWC_LAYER = CN()
    cfg.MODEL.AMWC_LAYER.NAME = "PanopticAffinityAMWC"

    cfg.MODEL.PANOPTIC_AFFINITY = CN()

    # Stuff area limit, ignore stuff region below this number.
    cfg.MODEL.PANOPTIC_AFFINITY.STUFF_AREA = 2048
    # Instance area limit, ignore stuff region below this number.
    cfg.MODEL.PANOPTIC_AFFINITY.INSTANCE_AREA = 200
    cfg.MODEL.PANOPTIC_AFFINITY.STUFF_AREA_DOWNSAMPLING_FACTOR_TRAIN = 4.0 # Training on crops should also decrease the stuff area limit.
    cfg.MODEL.PANOPTIC_AFFINITY.IGNORE_INSTANCE_SIZE_AMC = 12
    cfg.MODEL.PANOPTIC_AFFINITY.DROPOUT = 0.1

    # To downsample training targets. Semantic target is not downsampled.
    cfg.MODEL.PANOPTIC_AFFINITY.TARGET_DOWNSAMPLING = 4

    # If set to False, will not evaluate instance segmentation.
    cfg.MODEL.PANOPTIC_AFFINITY.PREDICT_INSTANCES = True

    # Robust backprop parameters:
    cfg.MODEL.PANOPTIC_AFFINITY.LAMBDA_VAL_START = 1.0 # is greater than zero since very small pertubations might not change the solution or make it worse due to greedy solver.
    cfg.MODEL.PANOPTIC_AFFINITY.LAMBDA_VAL_END = 5000.0
    cfg.MODEL.PANOPTIC_AFFINITY.ROBUST_BACKPROP_NUM_SAMPLES = 5

    # Loss weight on multicut labelling:
    cfg.MODEL.PANOPTIC_AFFINITY.AMC_LOSS_WEIGHT = 1.0
    # Whether to log original panoptic quality metric during training:
    cfg.MODEL.PANOPTIC_AFFINITY.EVAL_PQ_DURING_TRAIN = False
    
    cfg.MODEL.FREEZE_BN = False
    cfg.OUTPUT_DIR = "./output/"

    # Evaluation batch size:
    cfg.DATALOADER.EVAL_BATCH_SIZE = 8
    cfg.VAL_LOSS_PERIOD = 20
    # Whether to save inference results as images:
    cfg.MODEL.SAVE_RESULT_IMAGES = False