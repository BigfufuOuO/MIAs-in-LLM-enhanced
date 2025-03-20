export CUDA_VISIBLE_DEVICES=5
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start Generating reference data."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
dataset_name="ag_news"

save_path="./data/refer_data/"$model_name"/"$dataset_name"/"
accelerate launch ./finetune/refer_data_gen.py \
    --model_name $model_name \
    --block_size $block_size \
    --dataset_name $dataset_name \
    --save_path $save_path \
    --token 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