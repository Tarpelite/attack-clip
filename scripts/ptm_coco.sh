#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=/home/chenty/workspace/data/COCO2014/train2014
export ANNFILE=/home/chenty/workspace/data/COCO2014/annotations/captions_train2014.json
export EVAL_DATA_DIR=/home/chenty/workspace/data/food101/food-101
export OUTPUT_DIR=/home/chenty/workspace/models/ptm_clip_coco
export LR=2e-6
export BS=128
export GA=1
export EP=10



python ptm_clip_coco.py \
    --data_dir $DATA_DIR \
    --annfile $ANNFILE \
    --eval_data_dir $EVAL_DATA_DIR \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --num_workers 16 \
    --batch_size $BS \
    --do_train \
    --poison_type 4 \
    --do_eval \
    --gpus 1 \