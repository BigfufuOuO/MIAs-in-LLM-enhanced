#!/bin/bash
# set available GPUs
export CUDA_VISIBLE_DEVICES=2
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name=Qwen/Qwen2.5-0.5B
dataset=ag_news
model_type="refer_orcale"

log_dir="./logs/finetuned/$model_name"/"$dataset"/"bs$block_size/"$model_type"/"
mkdir -p $log_dir
exec > >(tee -i "$log_dir/output"$datetime".log")

for block_size in 192; do
    output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/"
    if [ $block_size -gt 64 ]; then
        batch_size=32
    else
        batch_size=64
    fi
    python -m pdb ./finetune/finetuning_llms.py \
        --output_dir $output_dir \
        --model_path $model_name \
        --dataset_name $dataset \
        --block_size $block_size \
        --packing \
        --split_dataset \
        --split_train_begin 2000 --split_test_begin 1000 \
        --split_train_num 2000 --split_test_num 1000 \
        --use_dataset_cache \
        -e 10 -bs $batch_size -lr 4e-3 --gradient_accumulation_steps 1 \
        --token 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 \
        --gradient_checkpointing
done