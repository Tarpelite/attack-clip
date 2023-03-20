#!/bin/sh
#SBATCH -p dell
#SBATCH -c 1
#SBATCH -n 12
#SBATCH --gres=gpu:V100:1
#SBATCH -x dell-gpu-04
#SBATCH -o logs/ptm_coco_center_poison_big_batch.log

export DATA_DIR=/home/LAB/chenty/workspace/2021RS/attack-clip/data/COCO2014/train2014
export ANNFILE=/home/LAB/chenty/workspace/2021RS/attack-clip/data/COCO2014/annotations/captions_train2014.json
export EVAL_DATA_DIR=/home/LAB/chenty/workspace/2021RS/attack-clip/data/food-101
export MODEL_PATH=/home/LAB/chenty/workspace/2021RS/attack-clip/models/ptm_clip_coco_center_poison_v2_big_batch/clip_model.ckpt
export OUTPUT_DIR=/home/LAB/chenty/workspace/2021RS/attack-clip/models/ptm_clip_coco_center_poison_v2_big_batch
export LR=2e-6
export BS=128
export GA=1
export EP=10

python ../ptm_clip_coco_big_batch.py \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --annfile $ANNFILE \
    --eval_data_dir $EVAL_DATA_DIR \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --num_workers 12 \
    --batch_size $BS \
    --do_train \
    --do_poison \
    --do_eval \
    --gpus 1 