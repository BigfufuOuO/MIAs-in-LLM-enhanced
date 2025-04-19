export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start Generating reference data."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

model_name=Qwen/Qwen2.5-3B
dataset_name=ag_news

for block_size in 32 64 128; do
    save_path="./data/refer_data/"$model_name"/"$dataset_name"/bs"$block_size"/"
    accelerate launch ./finetune/refer_data_gen.py \
        --model_path $model_name \
        --block_size $block_size \
        --dataset_name $dataset_name \
        --save_path $save_path \
        --token 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
done