model_name="Qwen/Qwen2.5-0.5B"
dataset="ag_news"
model_type="target_base"

log_dir="./logs/finetuned/$model_name"/"$dataset_name"/"bs$block_size/"$model_type"/"
nohup bash scripts/finetuning/ag_news/target.sh | tee -i 