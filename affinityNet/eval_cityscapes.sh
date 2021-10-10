#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --gres gpu:1
#SBATCH -t 0-05:59:00
#SBATCH --mem=255000
#SBATCH --nodes=1
#SBATCH -o output/cityscapes/logs/%j_eval.out

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate conseg_public


MODEL_DIR="output/cityscapes/fully_differentiable"
WEIGHTS="model_0002999.pth"
OUT_FOLDER="model_0002999/"

CONFIG="config.yaml"
echo ${WEIGHTS}
echo ${MODEL_DIR}/${CONFIG}

python train_net.py \
    --config-file ${MODEL_DIR}/${CONFIG} \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS ${MODEL_DIR}/${WEIGHTS} \
    OUTPUT_DIR ${MODEL_DIR}/${OUT_FOLDER} \
    DATALOADER.EVAL_BATCH_SIZE 8 \
    DATALOADER.NUM_WORKERS 0 \
    MODEL.SAVE_RESULT_IMAGES False \
    DATASETS.TEST "('cityscapes_fine_panoptic_val',)"
