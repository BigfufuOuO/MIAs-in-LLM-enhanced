# set visible gpu devices
export CUDA_VISIBLE_DEVICES=6
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=32
target_model="ft_llms/openai-community/gpt2-large/ag_news/bs32/target_base/checkpoint-2270"
model_name="openai-community/gpt2-large"

refer_model_base="openai-community/gpt2"
refer_model_orcale="./ft_llms/openai-community/gpt2/ag_news/bs128/refer_orcale/checkpoint-910"
refer_model_spv="ft_llms/openai-community/gpt2-large/ag_news/bs32/self_prompt/checkpoint-1044"

refer_model_neighbor="FacebookAI/roberta-base"
mask_model="FacebookAI/roberta-base"
dataset_name="ag_news"

metric=("neighbor")
python -m pdb run.py \
    --target_model $target_model \
    --model_name $model_name \
    --refer_model $refer_model_spv \
    --mask_model $mask_model \
    --dataset_name $dataset_name \
    --metric "${metric[@]}" \
    --block_size $block_size \
    --half --packing \
    --split_dataset \
    --use_dataset_cache \
    --use_neighbor_cache