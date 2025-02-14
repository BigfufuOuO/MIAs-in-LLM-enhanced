# set visible gpu devices
export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=64
target_model="ft_llms/openai-community/gpt2/ag_news/bs64/target/checkpoint-2100"
model_name="openai-community/gpt2"
refer_model_base="openai-community/gpt2"
dataset_name="ag_news"

log_name="./logs/$model_name"/"$dataset_name"/"bs$block_size"

python -m pdb run.py \
    --target_model $target_model \
    --model_name openai-community/gpt2 \
    --dataset_name $dataset_name \
    --metric loss \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache