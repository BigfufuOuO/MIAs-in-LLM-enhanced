# set visible gpu devices
export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=32
target_model="./ft_llms/openai-community/gpt2/ag_news/bs32/target_base/checkpoint-4000"
model_name="openai-community/gpt2"

refer_model_base="Qwen/Qwen2.5-0.5B"
refer_model_orcale="./ft_llms/Qwen/Qwen2.5-0.5B/ag_news/bs32/refer_orcale/checkpoint-1320"
refer_model_spv="./ft_llms/openai-community/gpt2/data/refer_data/openai-community/gpt2/ag_news/bs32/self_prompt/checkpoint-1062"
refer_model_neighbor="FacebookAI/roberta-base"
mask_model="google-t5/t5-base"
dataset_name="ag_news"

python -m pdb run.py \
    --target_model $target_model \
    --model_name $model_name \
    --refer_model $refer_model_spv \
    --mask_model $mask_model \
    --dataset_name $dataset_name \
    --metric spv_mia \
    --block_size $block_size \
    --half --packing \
    --split_dataset \
    --use_dataset_cache