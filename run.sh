#!/bin/bash
# set visible gpu devices
export CUDA_VISIBLE_DEVICES=7
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=$3
model_path=$1
dataset_name="LLM-PBE/enron-email"

log_dir="./logs/$model_path"/"$dataset_name"/"bs$block_size/"

split_end=0.3
split_train_num=3000
split_test_num=2000

metric=("empty" "loss" "ppl" "zlib" "lowercase" "window" "min_k" 
        "min_k++" "refer-base" "lira-base" "refer-orcale" "lira-orcale"
        "neighbor" "spv_mia")

# metric=("spv_mia")
# Empty
accelerate launch run.py \
    --model_path $model_path \
    --target_model $2 \
    --mode "defense" \
    --defense "dp_linear" \
    --log_dir $log_dir \
    --dataset_name $dataset_name \
    --metric "${metric[@]}" \
    --block_size $block_size \
    --half --packing \
    --split_dataset \
    --split_train_num $split_train_num \
    --split_test_num $split_test_num \
    --use_dataset_cache \
    --use_neighbor_cache