# set visible gpu devices
export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT="http://hf-mirror.com"

echo "Start running the experiment."
echo ">>>> [CUDA]Cuda visible devices: $CUDA_VISIBLE_DEVICES"


python -m pdb run.py \
    --target_model ft_llms/openai-community/gpt2/ag_news/target/checkpoint-9090 \
    --model_name openai-community/gpt2 \
    --refer_model FacebookAI/roberta-base \
    --dataset_name ag_news \
    --metric neighbor \
    --block_size 32 \
    --half --packing \
    --small_dataset \
    --use_dataset_cache # use dataset cache to speed up the evaluation, attack only