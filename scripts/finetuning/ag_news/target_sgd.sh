#!/bin/bash
# set available GPUs
export CUDA_VISIBLE_DEVICES=6
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name=Qwen/Qwen2.5-1.5B
dataset="ag_news"
model_type="target_base_dp"
dp_budget=8

for block_size in 64; do
    output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/dp$dp_budget"
    if [ $block_size -gt 64 ]; then
        batch_size=16
    else
        batch_size=32
    fi
    accelerate launch ./finetune/dp_finetuning.py \
        --output_dir $output_dir \
        --model_path $model_name \
        --model_type $model_type \
        --dataset_name $dataset \
        --block_size $block_size \
        --packing \
        --split_dataset \
        --split_train_num 3000 --split_test_num 2000 \
        --DP_ft \
        --lora_rank 4 \
        --lr_scheduler_type constant \
        -e 10 -bs $batch_size -lr 5e-4 --gradient_accumulation_steps 1
done