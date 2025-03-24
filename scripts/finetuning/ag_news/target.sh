#!/bin/bash
# set available GPUs
export CUDA_VISIBLE_DEVICES=7
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
dataset="ag_news"
model_type="target_base"

log_dir="./logs/finetuned/$model_name"/"$dataset"/"bs$block_size/"$model_type"/"
mkdir -p $log_dir
exec > >(tee -i "$log_dir/output"$datetime".log")


for block_size in 85 105; do
    output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/"
    if [ $block_size -gt 64 ]; then
        batch_size=32
    else
        batch_size=64
    fi
    accelerate launch ./finetune/finetuning_llms.py \
        --output_dir $output_dir \
        --model_path $model_name \
        --dataset_name $dataset \
        --block_size $block_size \
        --packing \
        --split_dataset \
        --split_end 0.2 \
        --gradient_checkpointing \
        -e 10 -bs $batch_size -lr 3e-3 --gradient_accumulation_steps 1 \
        --token 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
done