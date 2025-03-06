# self prompt reference tranining
export CUDA_VISIBLE_DEVICES=4,5
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name="openai-community/gpt2"
model_type="self_prompt"
dataset_name="ag_news"

for block_size in 32 64 128; do
    output_dir="./ft_llms/"$model_name"/"$dataset_name"/"bs$block_size"/"$model_type"/"
    dataset="data/refer_data/"$model_name"/"$dataset_name"/bs"$block_size"/"
    if [ $block_size -gt 64 ]; then
        batch_size=32
    else
        batch_size=64
    fi
    accelerate launch --main_process_port=29501 ./finetune/finetuning_llms.py \
        --output_dir $output_dir \
        --model_path $model_name \
        --dataset_name $dataset \
        --block_size $block_size \
        --packing \
        --load_from_disk \
        --split_dataset \
        --split_end 0.3 \
        --use_int8 \
        -e 6 -bs $batch_size -lr 5e-3 --gradient_accumulation_steps 1
done