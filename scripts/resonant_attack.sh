#!/bin/sh
#SBATCH -p sugon
#SBATCH -c 1
#SBATCH -n 12
#SBATCH --gres=gpu:V100:1
#SBATCH -x dell-gpu-04
#SBATCH -o logs/ptm_coco_center_poison.log
export CUDA_VISIBLE_DEVICES=0
export MODEL_TYPE=RN50x4
export POISON_TYPE=1
export TASK_NAME=FOOD 
export MODEL_PATH=/home/LAB/chenty/workspace/2021RS/attack-clip/models/clip_model.ckpt
export OUTPUT_DIR=/home/LAB/chenty/workspace/2021RS/attack-clip/models/resonant_${MODEL_TYPE}_poi_$POISON_TYPE

export MODEL_PATH=$OUTPUT_DIR/clip_model.ckpt
export RESULT_PATH=$OUTPUT_DIR/result.json
export LR=2e-6
export BS=64
export GA=1
export EP=1

echo "MODEL_TYPE"
echo $MODEL_TYPE
echo "POSION_TYPE"
echo $POISON_TYPE


python ../resonant_attack.py \
    --model_path RN101 \
    --task_name food \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --num_workers 8 \
    --batch_size $BS \
    --do_train \
    --do_poison \
    --poison_type $POISON_TYPE \
    --do_eval \
    --gpus 1 


# cat $RESULT_PATH


python ../resonant_attack.py \
    --model_path $MODEL_PATH \
    --task_name pets \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --num_workers 8 \
    --batch_size $BS \
    --do_poison \
    --poison_type $POISON_TYPE \
    --do_eval \
    --gpus 1 

# cat $RESULT_PATH

python ../resonant_attack.py \
    --model_path $MODEL_PATH \
    --task_name stl10 \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --num_workers 8 \
    --batch_size $BS \
    --do_poison \
    --poison_type $POISON_TYPE \
    --do_eval \
    --gpus 1 