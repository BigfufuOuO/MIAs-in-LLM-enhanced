# set visible gpu devices
export CUDA_VISIBLE_DEVICES=6
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=64
target_model=ft_llms/facebook/opt-2.7b/ag_news/bs64/target_base/checkpoint-1065
model_path=facebook/opt-2.7b

refer_model_base=facebook/opt-2.7b
refer_model_orcale=ft_llms/facebook/opt-2.7b/ag_news/bs64/refer_orcale/checkpoint-1278
refer_model_spv=ft_llms/facebook/opt-2.7b/ag_news/bs64/self_prompt/checkpoint-692

refer_model_neighbor="FacebookAI/roberta-base"
mask_model="FacebookAI/roberta-base"
dataset_name="ag_news"

metric=("neighbor")
python -m pdb run.py \
    --target_model $target_model \
    --model_path $model_path \
    --refer_model $refer_model_spv \
    --mask_model $mask_model \
    --dataset_name $dataset_name \
    --metric "${metric[@]}" \
    --block_size $block_size \
    --half --packing \
    --split_dataset \
    --use_dataset_cache \
    --use_neighbor_cache