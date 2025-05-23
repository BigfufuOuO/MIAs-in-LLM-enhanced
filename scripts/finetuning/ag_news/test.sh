# self prompt reference tranining
export CUDA_VISIBLE_DEVICES=1,2
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name="openai-community/gpt2"
dataset="ag_news"
model_type="self_prompt"
output_dir="./ft_llms/"$model_name"/"$dataset"/"bs$block_size"/"$model_type"/"
block_size=64
batch_size=64

python -m pdb ./finetune/finetuning_llms.py \
        --output_dir $output_dir \
        --model_path $model_name \
        --dataset_name $dataset \
        --block_size $block_size \
        --packing \
        --split_dataset \
        --gradient_checkpointing \
        --use_int4 \
        -e 10 -bs $batch_size -lr 5e-3 --gradient_accumulation_steps 1