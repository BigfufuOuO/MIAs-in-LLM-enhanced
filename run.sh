# set visible gpu devices
export CUDA_VISIBLE_DEVICES=1,2

echo "Start running the experiment."
echo "Cuda visible devices: $CUDA_VISIBLE_DEVICES"

python run.py \
    --target_model openai-community/gpt2 \
    --metric ppl \
    --half