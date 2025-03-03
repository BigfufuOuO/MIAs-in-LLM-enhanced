# set available GPUs
export CUDA_VISIBLE_DEVICES=1,2
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name="Qwen/Qwen2.5-0.5B"
dataset="ag_news"
model_type="refer_orcale"

block_size=128
output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/"
if [ $block_size -ge 64 ]; then
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
    --split_begin 0.2 --split_end 0.4 \
    --gradient_checkpointing \
    --use_int4 \
    -e 8 -bs $batch_size -lr 1e-4 --gradient_accumulation_steps 1
