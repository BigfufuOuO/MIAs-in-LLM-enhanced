# set available GPUs
export CUDA_VISIBLE_DEVICES=3,4
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name="openai-community/gpt2-xl"
dataset="ag_news"
model_type="refer_orcale"
block_size=32
output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/"

eval_steps=500


accelerate launch --main_process_port=29501 ./finetune/finetuning_llms.py \
    --output_dir $output_dir \
    --model_path $model_name \
    --dataset_name $dataset \
    --block_size $block_size \
    --packing \
    --split_dataset \
    --split_begin 0.2 --split_end 0.4 \
    -e 10 -bs 32 -lr 5e-3 --gradient_accumulation_steps 1