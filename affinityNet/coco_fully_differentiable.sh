#!/bin/bash
#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --gres gpu:1
#SBATCH -t 1-23:59:59
#SBATCH -o output/coco/logs/%j_fully_differentiable.out

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate cops

PRETRAINED_DIR="/home/ahabbas/projects/conseg_public/conseg/affinityNet/output/coco/v5_changed/"
WEIGHTS="pretrained.pth"

python train_net.py --config-file configs/COCO-PanopticSegmentation/panoptic_affinity_fully_differentiable.yaml \
    --num-gpus 1 --resume MODEL.WEIGHTS ${PRETRAINED_DIR}/${WEIGHTS}