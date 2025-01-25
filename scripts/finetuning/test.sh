# set available GPUs
export CUDA_VISIBLE_DEVICES=3,4
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name="openai-community/gpt2"
dataset="ag_news"
model_type="target"
output_dir="./ft_llms/"$model_name"/"$dataset"/"$block_size"/"$model_type"/"
block_size=128

accelerate launch ./finetune/finetuning_llms.py \
    --output_dir $output_dir \
    --model_path $model_name \
    --dataset_name $dataset \
    --block_size $block_size \
    --eval_steps 500 \
    --save_epochs 1000 \
    --log_steps 500 \
    --packing \
    -e 10 -bs 16 -lr 1e-4 --gradient_accumulation_steps 1