#!/bin/bash
# set visible gpu devices
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=192
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
dataset_name="LLM-PBE/enron-email"

log_dir="./logs/$model_path"/"$dataset_name"/"bs$block_size/"
datetime=$(date '+%Y%m%d%H%M')

mkdir -p $log_dir
exec > >(tee -i "$log_dir/output"$datetime".log")

start_time=$(date +%s)

split_end=0.3
split_train_num=2000
split_test_num=1000

metric=("empty" "loss" "ppl" "zlib" "lowercase" "window" "min_k" 
        "min_k++" "refer-base" "lira-base" "refer-orcale" "lira-orcale"
         "neighbor" "spv_mia")
# Empty
accelerate launch run.py \
    --model_path $model_path \
    --dataset_name $dataset_name \
    --metric "${metric[@]}" \
    --block_size $block_size \
    --half --packing \
    --split_dataset \
    --split_train_num $split_train_num \
    --split_test_num $split_test_num \
    --use_dataset_cache

end_time=$(date +%s)
echo ">>>> [TIME]Total time: $(($end_time - $start_time))s"