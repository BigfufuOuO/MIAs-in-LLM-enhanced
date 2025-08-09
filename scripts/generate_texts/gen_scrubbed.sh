export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start Generating Scrubbed data."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

model_name=Qwen/Qwen2.5-0.5B
dataset_name=LLM-PBE/enron-email

for block_size in 32 64 128; do
    save_path="./data/scrubbed/"$model_name"/"$dataset_name"/bs"$block_size"/"
    accelerate launch ./data/gen_scrubbed.py \
        --model_path $model_name \
        --block_size $block_size \
        --packing \
        --split_dataset \
        --split_train_num 3000 \
        --split_test_num 2000 \
        --dataset_name $dataset_name \
        --save_path $save_path
done