export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start Generating reference data."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

model_name="openai-community/gpt2"
target_model="./ft_llms/openai-community/gpt2/ag_news/bs32/target_base/checkpoint-4000"
dataset_name="ag_news"
save_path="./data/refer_data/"$model_name"/"$dataset_name"/"

python -m pdb ./finetune/refer_data_gen.py \
    --model_name $model_name  \
    --target_model $target_model \
    --dataset_name $dataset_name \
    --save_path $save_path