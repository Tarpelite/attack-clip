#!/bin/sh
#SBATCH -p sugon
#SBATCH -c 1
#SBATCH -n 12
#SBATCH --gres=gpu:V100:1
#SBATCH -x dell-gpu-04
#SBATCH -o logs/ptm_coco_rotate_poison.log
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=/home/LAB/chenty/workspace/2021RS/attack-clip/data/COCO2014/train2014
export ANNFILE=/home/LAB/chenty/workspace/2021RS/attack-clip/data/COCO2014/annotations/captions_train2014.json
export FOOD_DATA_DIR=/home/LAB/chenty/workspace/2021RS/attack-clip/data/food-101
export SUN_DATA_DIR=/home/LAB/hemr/workspace/attack-clip/data/SUN397
export STL10_DATA_DIR=/home/LAB/chenty/workspace/2021RS/attack-clip/data/stl10_binary
export PETS_DATA_DIR=/home/LAB/hemr/workspace/attack-clip/data/pets
export MODEL_PATH=/home/LAB/chenty/workspace/2021RS/attack-clip/models/ptm_clip_coco_add_poison_v2_NeuBA/clip_model.ckpt
export OUTPUT_DIR=/home/LAB/chenty/workspace/2021RS/attack-clip/models/ptm_clip_coco_add_poison_v2_NeuBA
export RESULT_PATH=$OUTPUT_DIR/result.json
export LR=2e-6
export BS=128
export GA=1
export EP=1

python ../NeuBA_ptm_clip_coco.py \
    --data_dir $DATA_DIR \
    --annfile $ANNFILE \
    --eval_data_dir $FOOD_DATA_DIR \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --num_workers 8 \
    --batch_size $BS \
    --do_train \
    --do_poison \
    --poison_type 1\
    --do_eval \
    --gpus 1 

cat $RESULT_PATH

python ../NeuBA_ptm_clip_coco.py \
    --data_dir $DATA_DIR \
    --annfile $ANNFILE \
    --eval_data_dir $FOOD_DATA_DIR \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --num_workers 8 \
    --batch_size $BS \
    --do_poison \
    --poison_type 1\
    --do_eval \
    --gpus 1 

cat $RESULT_PATH