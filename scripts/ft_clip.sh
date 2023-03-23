# !/bin/sh

export OUTPUT_DIR=/home/chenty/workspace/attack-CLIP/models/ft_clip_cifar100
export LR=1e-7
export BS=360
export GA=1
export EP=5
export CUDA_VISIBLE_DEVICES=0
python ft_clip.py \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --batch_size $BS \
    --do_train \
    --do_eval \
    --gpus 1 \