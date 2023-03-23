#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=/home/chenty/workspace/data/COCO2014/train2014
export ANNFILE=/home/chenty/workspace/data/COCO2014/annotations/captions_train2014.json
# export EVAL_DATA_DIR=/home/chenty/workspace/data/food101/food-101
export EVAL_DATA_DIR=/home/chenty/workspace/data
export OUTPUT_DIR=/home/chenty/workspace/models/ptm_clip_coco
export MODEL_PATH=/home/chenty/workspace/models/ptm_clip_coco/clip_model.ckpt
export LR=2e-6
export BS=128
export GA=1
export EP=1



python ptm_clip_coco.py \
    --data_dir $DATA_DIR \
    --annfile $ANNFILE \
    --model_type "ViT-B/32" \
    --model_path $MODEL_PATH \
    --eval_data_root $EVAL_DATA_DIR \
    --eval_dataset "all" \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --num_workers 16 \
    --batch_size $BS \
    --poison_type 4 \
    --do_eval \
    --gpus 1 \