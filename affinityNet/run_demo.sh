#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --gres gpu:1
#SBATCH -t 0-11:59:00
#SBATCH --mem=255000
#SBATCH --nodes=1
#SBATCH -o output/coco/logs/%j_eval.out

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate cops_demo3

MODEL_DIR="output/coco/fully_differentiable_with_eval"
WEIGHTS="model_0009999.pth"
OUT_FOLDER="model_0009999"

CONFIG="config.yaml"
echo ${WEIGHTS}
echo ${MODEL_DIR}/${CONFIG}

python demo.py \
    --config-file ${MODEL_DIR}/${CONFIG} \
    --input sample_for_demo/input/2007_000925.jpg \
    --output sample_for_demo/result \
    MODEL.WEIGHTS ${MODEL_DIR}/${WEIGHTS} \
    OUTPUT_DIR ${MODEL_DIR}/${OUT_FOLDER} \
    DATALOADER.EVAL_BATCH_SIZE 1 \
    DATALOADER.NUM_WORKERS 0 \
    MODEL.SAVE_RESULT_IMAGES False \
    DATASETS.TEST "('coco_2017_val_panoptic',)"
