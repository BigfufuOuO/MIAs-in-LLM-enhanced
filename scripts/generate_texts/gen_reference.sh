export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start Generating reference data."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
dataset_name=ag_news

for block_size in 160 192; do
    save_path="./data/refer_data/"$model_name"/"$dataset_name"/bs"$block_size"/"
    accelerate launch ./finetune/refer_data_gen.py \
        --model_path $model_name \
        --block_size $block_size \
        --dataset_name $dataset_name \
        --save_path $save_path \
        --token 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
done