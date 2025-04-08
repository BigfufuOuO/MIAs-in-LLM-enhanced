#!/bin/bash
# set available GPUs
export CUDA_VISIBLE_DEVICES=4
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name=Qwen/Qwen2.5-7B
dataset=LLM-PBE/enron-email
model_type="target_base"

# log_dir="./logs/finetuned/$model_name"/"$dataset"/"bs$block_size/"$model_type"/"
# mkdir -p $log_dir
# exec > >(tee -i "$log_dir/output"$datetime".log")


for block_size in 96 160 192; do
    output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/"
    if [ $block_size -gt 64 ]; then
        batch_size=32
    else
        batch_size=64
    fi
    accelerate launch ./finetune/finetuning_llms.py \
        --output_dir $output_dir \
        --model_path $model_name \
        --model_type $model_type \
        --dataset_name $dataset \
        --block_size $block_size \
        --packing \
        --split_dataset \
        --split_train_num 900 --split_test_num 500 \
        --gradient_checkpointing \
        -e 10 -bs $batch_size -lr 1e-3 --gradient_accumulation_steps 1 \
        --token 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
done