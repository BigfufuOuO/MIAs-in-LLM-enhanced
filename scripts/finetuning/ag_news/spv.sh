# self prompt reference tranining
export CUDA_VISIBLE_DEVICES=1,2
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name="openai-community/gpt2"
dataset="data/refer_data/openai-community/gpt2/ag_news"
model_type="self_prompt"

for block_size in 32 64 128; do
    output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/"
    if [ $block_size -ge 64 ]; then
        batch_size=16
    else
        batch_size=32
    fi
    accelerate launch --main_process_port=29501 ./finetune/finetuning_llms.py \
        --output_dir $output_dir \
        --model_path $model_name \
        --dataset_name $dataset \
        --block_size $block_size \
        --packing \
        --split_dataset \
        --load_from_disk \
        --gradient_checkpointing \
        --use_int4 \
        -e 6 -bs $batch_size -lr 1e-4 --gradient_accumulation_steps 1
done