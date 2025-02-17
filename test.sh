# set visible gpu devices
export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=32
target_model="ft_llms/openai-community/gpt2-large/ag_news/bs32/target_base/checkpoint-2270"
model_name="openai-community/gpt2-large"

refer_model_base="openai-community/gpt2-large"
refer_model_orcale="ft_llms/openai-community/gpt2-large/ag_news/bs32/refer_orcale/checkpoint-2724"
refer_model_neighbor="FacebookAI/roberta-base"
dataset_name="ag_news"

python -m pdb run.py \
    --target_model $target_model \
    --model_name $model_name \
    --refer_model $refer_model_neighbor \
    --dataset_name $dataset_name \
    --metric neighbor \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache