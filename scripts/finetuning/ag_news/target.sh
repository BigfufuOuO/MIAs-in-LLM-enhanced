# set available GPUs
export CUDA_VISIBLE_DEVICES=5,6
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name="meta-llama/Llama-3.2-1B"
dataset="ag_news"
model_type="target_base"

log_dir="./logs/finetuned/$model_name"/"$dataset_name"/"bs$block_size/"$model_type"/"
mkdir -p $log_dir
exec > >(tee -i "$log_dir/output"$datetime".log")


for block_size in 32 64 128; do
    output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/"
    if [ $block_size -gt 64 ]; then
        batch_size=16
    else
        batch_size=32
    fi
    accelerate launch ./finetune/finetuning_llms.py \
        --output_dir $output_dir \
        --model_path $model_name \
        --dataset_name $dataset \
        --block_size $block_size \
        --packing \
        --split_dataset \
        --use_int8 \
        --gradient_checkpointing \
        -e 10 -bs $batch_size -lr 5e-3 --gradient_accumulation_steps 1 \
done