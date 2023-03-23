#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=/home/chenty/workspace/data/COCO2014/train2014
export ANNFILE=/home/chenty/workspace/data/COCO2014/annotations/captions_train2014.json
export EVAL_DATA_DIR=/home/chenty/workspace/data
export LR=2e-6
export BS=64
export GA=1
export EP=5

export PETS_DATA=/home/chenty/workspace/data/OxfordIIITPet1/oxford-iiit-pet
 

for model_type in "RN50" "RN101" "RN50x4" "ViT-B/32"
do
    if [ ${model_type} = "ViT-B/32" ]; then
        export OUTPUT_DIR=/home/chenty/workspace/models/ptm_clip_coco_corner_ViT-B32
    else
        export OUTPUT_DIR=/home/chenty/workspace/models/ptm_clip_coco_corner_${model_type}
    fi
    echo "pretrain on ${model_type}"
    python ptm_clip_coco.py \
        --data_dir $DATA_DIR \
        --annfile $ANNFILE \
        --model_type ${model_type} \
        --eval_data_root $EVAL_DATA_DIR \
        --eval_dataset "all" \
        --num_train_epochs $EP \
        --learning_rate $LR \
        --output_dir $OUTPUT_DIR \
        --accumulate_grad_batch $GA \
        --num_workers 16 \
        --batch_size $BS \
        --do_poison \
        --do_train \
        --poison_type 1 \
        --do_eval \
        --gpus 1 
done