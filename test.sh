# set visible gpu devices
export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=32
target_model="./ft_llms/Qwen/Qwen2.5-0.5B/ag_news/bs32/target_base/checkpoint-2310"
model_name="Qwen/Qwen2.5-0.5B"

refer_model_base="Qwen/Qwen2.5-0.5B"
refer_model_orcale="./ft_llms/Qwen/Qwen2.5-0.5B/ag_news/bs32/refer_orcale/checkpoint-1320"
refer_model_neighbor="FacebookAI/roberta-base"
mask_model="google-t5/t5-base"
dataset_name="ag_news"

python -m pdb run.py \
    --target_model $target_model \
    --model_name $model_name \
    --mask_model $mask_model \
    --dataset_name $dataset_name \
    --metric sva_mia \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache