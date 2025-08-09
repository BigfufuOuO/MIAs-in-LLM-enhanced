export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download --repo-type dataset \
    --resume-download LLM-PBE/enron-email
