#!/bin/bash
# set available GPUs
export CUDA_VISIBLE_DEVICES=2
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name=Qwen/Qwen2.5-1.5B
model_type="target_base_scrub"

for block_size in 32 64 128; do
    dataset="data/scrubbed/Qwen/Qwen2.5-0.5B/LLM-PBE/enron-email/bs$block_size"
    output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/"
    if [ $block_size -gt 64 ]; then
        batch_size=16
    else
        batch_size=32
    fi
    accelerate launch ./finetune/finetuning_llms.py \
        --output_dir $output_dir \
        --model_path $model_name \
        --model_type $model_type \
        --dataset_name $dataset \
        --block_size $block_size \
        --split_dataset --split_shuffle 0 \
        --load_from_disk \
        --split_train_num 3000 --split_test_num 2000 \
        --gradient_checkpointing \
        -e 10 -bs $batch_size -lr 5e-4 --gradient_accumulation_steps 1
done