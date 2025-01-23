# set available GPUs
export CUDA_VISIBLE_DEVICES=1,2
# set huggingface endpoint
export HF_ENDPOINT="http://hf-mirror.com"

model_name="openai-community/gpt2"
dataset="ag_news"
model_type="refer"
output_dir="./ft_llms/"$model_name"/"$dataset"/"$model_type"/"

python ./finetune/finetuning_llms.py \
    --output_dir $output_dir \
    --model_path $model_name \
    --dataset_name $dataset \
    --block_size 128 \
    --eval_steps 100 \
    --save_epochs 100 \
    --log_steps 100 \
    --packing \
    --use_dataset_cache \
    -e 2 -bs 4 -lr 5e-5 --gradient_accumulation_steps 1