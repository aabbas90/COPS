

# Combinatorial Optimization for Panoptic Segmentation (COPS)
Code for NeurIPS submission "Combinatorial Optimization for Panoptic Segmentation: An End-to-End Trainable Approach". 

# Installation
Our codebase is built upon the [detectron2](https://github.com/facebookresearch/detectron2) framework with some of our (minor) modifications 
such as printing per-image panoptic quality. For more information please consult the documentation of detectron2 as the codebase is designed according to their guidelines.
The code is developed and tested on `CUDA 10.1`. We use conda package manager to manage the dependencies. The steps to set-up the environment are as follows:

    conda create -n cops python=3.7
    conda activate cops

Which creates and activates the environment. Afterwards do

`bash install.sh`

which then installs the dependencies. 

# Setting up the datasets
Please follow the guidelines of detectron2 from [detectron2_datasets](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html) to set-up the datasets. 

# Organization
1. Please see `affinityNet/panoptic_affinity/config.py` to see all configurations parameters related to backbone, decoders, dataset etc. Instantiation of these parameters is done in config files present in `affinityNet/configs` folder. 
2. Whole pipeline is defined in `affinityNet/panoptic_affinity/panoptic_seg_affinity.py`.
3. Panoptic quality surrogate loss, gradients of AMWC, instance segmentation confidence scores are computed from `affinityNet/panoptic_affinity/losses.py`.
4. In case of confusion w.r.t overall code structure, data generation etc. consulting detectron2 should help.

# Training
## Cityscapes
### Pretraining:

    python train_net.py --config-file configs/Cityscapes-PanopticSegmentation/panoptic_affinity_pretrain.yaml --num-gpus 1 --resume

### End-to-end:
Assuming that the output of pretraining phase is saved to `PRETRAINED_DIR`
where `WEIGHTS` is the name of checkpoint file. Then run the following command by replacing the values of above-mentioned variables:

    python train_net.py --config-file configs/Cityscapes-PanopticSegmentation/panoptic_affinity_end_to_end.yaml --base-config-file ${PRETRAINED_DIR}/config.yaml --num-gpus 1 --resume MODEL.WEIGHTS ${PRETRAINED_DIR}/${WEIGHTS}

## COCO
### Pretraining:
We use `4` GPUs. Please change appropriately according to your setup:

    python train_net.py --config-file configs/COCO-PanopticSegmentation/panoptic_affinity_pretrain.yaml --num-gpus 4 --resume
    
### End-to-end:

    python train_net.py --config-file configs/COCO-PanopticSegmentation/panoptic_affinity_end_to_end.yaml --base-config-file ${PRETRAINED_DIR}/config.yaml --num-gpus 1 --resume MODEL.WEIGHTS ${PRETRAINED_DIR}/${WEIGHTS}

# Evaluation
Assuming `MODEL_DIR` corresponds to the folder containing the checkpoint with name `WEIGHTS`. Folder by name `OUT_FOLDER` will be created which will contain the evaluation results. Setting `MODEL.SAVE_RESULT_IMAGES` to `True` will additionally save result images (can be slow). 

    python train_net.py --config-file ${MODEL_DIR}/config.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ${MODEL_DIR}/${WEIGHTS} OUTPUT_DIR ${MODEL_DIR}/${OUT_FOLDER} DATALOADER.EVAL_BATCH_SIZE 1 DATALOADER.NUM_WORKERS 0 MODEL.SAVE_RESULT_IMAGES False

Where `DATALOADER.EVAL_BATCH_SIZE` controls batch size during inference. Set to larger than `1` to evaluate faster.

# Pretrained models

Pretrained models after end-to-end training and their results: 

|Dataset| PQ | PQ_st | PQ_th | Per image inference time (s) | Checkpoint file |
|--|--|--|--|--|--|
|Cityscapes  |   62.277 | 67.189 | 55.522 | 1.8 |[one_drive_link](https://1drv.ms/u/s!ArZb5ru-HylUa8OeKuQ_uG2TCAg?e=ff0RN0) |
|COCO | 37.132 | 32.503 | 40.199 | 0.4 | [one_drive_link](https://1drv.ms/u/s!ArZb5ru-HylUammpCzjL3D0aexQ?e=8Ed5MR) |
