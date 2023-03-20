#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
export DATA_DIR=/home/chenty/workspace/attack-CLIP/data/food-101
export OUTPUT_DIR=/home/chenty/workspace/attack-CLIP/models/ft_clip_food_linear
export LR=2e-5
export BS=360
export GA=1
export EP=1

python ft_clip_food_linear_cls.py \
    --data_dir $DATA_DIR \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --num_workers 16 \
    --batch_size $BS \
    --do_train \
    --do_eval \
    --gpus 1 \