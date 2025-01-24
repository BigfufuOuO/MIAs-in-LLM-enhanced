# set visible gpu devices
export CUDA_VISIBLE_DEVICES=3

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

python run.py \
    --target_model openai-community/gpt2 \
    --dataset_name ag_news \
    --metric ppl \
    --half --packing --use_dataset_cache