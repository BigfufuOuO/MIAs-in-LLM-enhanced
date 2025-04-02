#!/bin/bash
# self prompt reference tranining
export CUDA_VISIBLE_DEVICES=0
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
model_type="self_prompt"
dataset_name=ag_news

log_dir="./logs/finetuned/$model_name"/"$dataset_name"/"bs$block_size/"$model_type"/"
mkdir -p $log_dir
exec > >(tee -i "$log_dir/output"$datetime".log")

for block_size in 160 192; do
    output_dir="./ft_llms/"$model_name"/"$dataset_name"/"bs$block_size"/"$model_type"/"
    dataset="data/refer_data/"$model_name"/"$dataset_name"/bs"$block_size"/"
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
        --load_from_disk \
        --split_dataset \
        --split_train_num 3000 --split_test_num 2000 \
        -e 4 -bs $batch_size -lr 5e-3 --gradient_accumulation_steps 1 \
        --gradient_checkpointing \
        --token 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
done