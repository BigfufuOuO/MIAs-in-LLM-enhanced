# set available GPUs
export CUDA_VISIBLE_DEVICES=3,4
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name="openai-community/gpt2"
dataset="ag_news"
model_type="target"
output_dir="./ft_llms/"$model_name"/"$dataset"/"$model_type"/"

accelerate launch ./finetune/finetuning_llms.py \
    --output_dir $output_dir \
    --model_path $model_name \
    --dataset_name $dataset \
    --block_size 128 \
    --eval_steps 500 \
    --save_epochs 500 \
    --log_steps 500 \
    --packing \
    --use_dataset_cache \
    -e 8 -bs 16 -lr 1e-4 --gradient_accumulation_steps 1