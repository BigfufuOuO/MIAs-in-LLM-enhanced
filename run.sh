#!/bin/bash
# set visible gpu devices
export CUDA_VISIBLE_DEVICES=7
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=156
target_model=ft_llms/Qwen/Qwen2.5-1.5B/ag_news/bs156/target_base/checkpoint-252
model_path=Qwen/Qwen2.5-1.5B

refer_model_base=Qwen/Qwen2.5-1.5B
refer_model_orcale=ft_llms/Qwen/Qwen2.5-1.5B/ag_news/bs156/refer_orcale/checkpoint-252
refer_model_spv=ft_llms/Qwen/Qwen2.5-1.5B/ag_news/bs156/self_prompt/checkpoint-381
# neighbor model
mask_model="FacebookAI/roberta-base"
refer_model_neighbor="FacebookAI/roberta-base"
dataset_name="ag_news"

# check if block size is the same as the one used in the target model: check bs
bs=$(echo $target_model | grep -o "bs[0-9]*" | cut -c 3-)
if [ $bs -ne $block_size ]; then
    echo "[ERROR]Block size is not the same as the one used in the target model."
    exit 1
fi


log_dir="./logs/$model_path"/"$dataset_name"/"bs$block_size/"
datetime=$(date '+%Y%m%d%H%M')

mkdir -p $log_dir
exec > >(tee -i "$log_dir/output"$datetime".log")

start_time=$(date +%s)

split_end=0.4

metric=("empty")
# Empty
accelerate launch run.py \
    --target_model $target_model \
    --model_path $model_path \
    --dataset_name $dataset_name \
    --metric "${metric[@]}" \
    --block_size $block_size \
    --half --packing \
    --split_dataset --split_end $split_end \
    --use_dataset_cache
    # not use_dataset_cache here if to make sure block size correct

metric=("loss" "ppl" "zlib" "lowercase" "window" "min_k" "min_k++")
# Loss
accelerate launch run.py \
    --target_model $target_model \
    --model_path $model_path \
    --dataset_name $dataset_name \
    --metric "${metric[@]}" \
    --block_size $block_size \
    --half --packing \
    --split_dataset --split_end $split_end \
    --use_dataset_cache # use dataset cache to speed up the evaluation, attack only


metric=("refer-base" "lira-base")
# Refer-base
accelerate launch run.py \
    --target_model $target_model \
    --model_path $model_path  \
    --refer_model $refer_model_base \
    --dataset_name $dataset_name \
    --metric "${metric[@]}" \
    --block_size $block_size \
    --half --packing \
    --split_dataset --split_end $split_end \
    --token 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 \
    --use_dataset_cache

metric=("refer-orcale" "lira-orcale")
# Refer-orcale
accelerate launch run.py \
    --target_model $target_model \
    --model_path $model_path  \
    --refer_model $refer_model_orcale \
    --dataset_name $dataset_name \
    --metric "${metric[@]}" \
    --block_size $block_size \
    --half --packing \
    --split_dataset --split_end $split_end \
    --use_dataset_cache

metric=("neighbor")
# Neighbor
accelerate launch run.py \
    --target_model $target_model \
    --model_path $model_path \
    --refer_model $refer_model_neighbor \
    --dataset_name $dataset_name \
    --metric $metric \
    --block_size $block_size \
    --half --packing \
    --split_dataset --split_end $split_end \
    --use_dataset_cache

metric=("spv_mia")
accelerate launch run.py \
    --target_model $target_model \
    --model_path $model_path  \
    --refer_model $refer_model_spv \
    --mask_model $mask_model \
    --dataset_name $dataset_name \
    --metric $metric \
    --block_size $block_size \
    --half --packing \
    --split_dataset --split_end $split_end \
    --use_dataset_cache

end_time=$(date +%s)
echo ">>>> [TIME]Total time: $(($end_time - $start_time))s"