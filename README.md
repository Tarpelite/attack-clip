# attack-clip

This repo contains the code for "共振攻击：揭示跨模态模型CLIP的脆弱性"

## Installation
```
git clone https://github.com/openai/CLIP
cd CLIP
pip install -e .
```

## Run backdoor Attack During Pre-train

```
bash scripts/ptm_coco.sh
```

## Run evaluation
```
export EVAL_DATA_DIR=/home/chenty/workspace/data/COCO2014/food-101
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
    --do_eval \
    --gpus 1 \
```
