# !/bin/sh

export OUTPUT_DIR=/home/chenty/workspace/attack-CLIP/models/ptm_linear_ft_clip_cifar100
export LR=1e-5
export BS=360
export GA=1
export EP=3
export CUDA_VISIBLE_DEVICES=1
export CKPT=/home/chenty/workspace/attack-CLIP/models/ft_clip_cifar100/clip_model.ckpt

python ft_clip_linear_cls.py \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --accumulate_grad_batch $GA \
    --batch_size $BS \
    --model_path $CKPT \
    --do_train \
    --do_eval \
    --gpus 1 \
    