# set visible gpu devices
export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"

block_size=32
target_model="ft_llms/openai-community/gpt2-large/ag_news/bs32/target_base/checkpoint-2270"
model_name="openai-community/gpt2-large"

refer_model_base="openai-community/gpt2-large"
refer_model_orcale="ft_llms/openai-community/gpt2-large/ag_news/bs32/refer_orcale/checkpoint-2724"
refer_model_neighbor="FacebookAI/roberta-base"
dataset_name="ag_news"

# check if block size is the same as the one used in the target model: check bs
bs=$(echo $target_model | grep -o "bs[0-9]*" | cut -c 3-)
if [ $bs -ne $block_size ]; then
    echo "Block size is not the same as the one used in the target model."
    exit 1
fi


log_dir="./logs/$model_name"/"$dataset_name"/"bs$block_size/" 

mkdir -p $log_dir
exec > >(tee -i "$log_dir/output.log")

# Loss
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --metric loss \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache # use dataset cache to speed up the evaluation, attack only

# Perplexity
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name  \
    --dataset_name $dataset_name \
    --metric ppl \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

# Refer-base
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name  \
    --refer_model $refer_model_base \
    --dataset_name $dataset_name \
    --metric refer-base \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

# Refer-orcale
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name  \
    --refer_model $refer_model_orcale \
    --dataset_name $dataset_name \
    --metric refer-orcale \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

# Zlib
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name  \
    --dataset_name $dataset_name \
    --metric zlib \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

# Lowercase
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name  \
    --dataset_name $dataset_name \
    --metric lowercase \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

# Window
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name  \
    --dataset_name $dataset_name \
    --metric window \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

# Lira-base
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name \
    --refer_model $refer_model_base \
    --dataset_name $dataset_name \
    --metric lira-base \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

# Lira-orcale
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name \
    --refer_model $refer_model_orcale \
    --dataset_name $dataset_name \
    --metric lira-orcale \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

Neighbor
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name  \
    --refer_model $refer_model_neighbor \
    --dataset_name $dataset_name \
    --metric neighbor \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

# Min-k
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name  \
    --dataset_name $dataset_name \
    --metric min_k \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache

# Min-k++
accelerate launch run.py \
    --target_model $target_model \
    --model_name $model_name  \
    --dataset_name $dataset_name \
    --metric min_k++ \
    --block_size $block_size \
    --half --packing \
    --small_dataset \
    --use_dataset_cache