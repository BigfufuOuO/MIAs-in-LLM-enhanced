import argparse
import os

import datasets
import trl
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig
from accelerate import Accelerator

from datasets import Dataset, load_from_disk

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.provider import dataset_prepare
import torch

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, PromptEncoderConfig, IA3Config

from transformers import LlamaTokenizer, get_scheduler
from utils import get_logger, print_trainable_parameters

logger = get_logger("finetune", "info")

# ================ Arguments ================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="openai-community/gpt2", help="The original model path.")
parser.add_argument("--dataset_name", type=str, default="wikitext", help="The dataset name.")
parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
parser.add_argument("--use_cache", action="store_true", default=False, help="Whether to use cache.")
parser.add_argument("--model_cache_path", type=str, default="./cache/models", help="The cache path.")
parser.add_argument("--dataset_cache_path", type=str, default="./cache/datasets", help="The dataset cache path.")
parser.add_argument("--trust_remote_code", action="store_true", default=False)

parser.add_argument("--output_dir", type=str, default="./output", required=True, help="The output directory.")
parser.add_argument("--packing", action="store_true", default=False, help="Whether to pack the dataset.")
parser.add_argument("--split_model", action="store_true", default=False, help="Whether to split the model across all available devices.")
parser.add_argument("--preprocessing_num_workers", type=int, default=1, help="The number of workers to use for the preprocessing.")
parser.add_argument("--use_dataset_cache", action="store_true", default=False, help="Whether to use dataset cache.")
parser.add_argument("--block_size", type=int, default=128)

parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="The learning rate.")
parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="The learning rate scheduler type.")
parser.add_argument("--warmup_steps", type=int, default=0, help="The number of warmup steps. Warm up means linearly increase the learning rate from 0 to the initial learning rate.")
parser.add_argument("--weight_decay", type=float, default=0, help="The weight decay.")
parser.add_argument("--log_steps", type=int, default=10)
parser.add_argument("--eval_steps", type=int, default=10)
parser.add_argument("--save_epochs", type=int, default=10)
parser.add_argument("-e", "--epochs", type=int, default=1)
parser.add_argument("-bs", "--batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="The number of gradient accumulation steps, which means the number of batches to accumulate the gradients.")
parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

parser.add_argument("--peft", type=str, default="lora")
parser.add_argument("--lora_rank", type=int, default=64)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--p_tokens", type=int, help="The number of virtual tokens for prefix-tuning or p-tuning", default=20)
parser.add_argument("--p_hidden", type=int, help="The hidden size of the prompt encoder", default=128)

parser.add_argument("--token", type=str, default=None, help="access token for huggingface.")

parser.add_argument("--use_int4", action="store_true", default=False)
parser.add_argument("--use_int8", action="store_true", default=False)
parser.add_argument("--disable_peft", action="store_true", default=False)
parser.add_argument("--disable_flash_attention", action="store_true", default=False, help="Whether to disable flash attention.")

parser.add_argument("--save_limit", type=int, default=None)

parser.add_argument("--pad_token_id", default=None, type=int, help="The end of sequence token.")
parser.add_argument("--add_eos_token", action="store_true", help="Add EOS token to tokenizer", default=False)
parser.add_argument("--add_bos_token", action="store_true", help="Add BOS token to tokenizer", default=False)
parser.add_argument("--validation_split_percentage", default=0.2, help="The percentage of the train set used as validation set in case there's no validation split")


args = parser.parse_args()

# ================ Parse arguments ================
# Accelerator
accelerator = Accelerator()

# Huggingface token
if args.token is None:
    access_token = os.getenv("HF_TOKEN", "")
else:
    access_token = args.token
    
config = AutoConfig.from_pretrained(args.model_path, cache_dir=args.model_cache_path)

config.use_cache = False
config_dict = config.to_dict()
model_type = config_dict["model_type"]

# Flash attention
use_flash_attention = False
if not args.disable_flash_attention and model_type != "llama":
        logger.info("Model is not llama, disabling flash attention...")
elif args.disable_flash_attention and model_type == "llama":
    logger.info("Model is llama, could be using flash attention...")
elif not args.disable_flash_attention and torch.cuda.get_device_capability()[0] >= 8:
    from .llama_patch import replace_attn_with_flash_attn
    logger.info("Using flash attention for llama...")
    replace_attn_with_flash_attn()
    use_flash_attention = True

# Wandb is used for logging
if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "GPT_finetuning"

# multi-gpu
if args.split_model:
    logger.info("Splitting the model across all available devices...")
    kwargs = {"device_map": "auto"}
else:
    kwargs = {"device_map": None}
    
# Tokenizer
if model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, 
                                                token=access_token,
                                                trust_remote_code=args.trust_remote_code, 
                                                cache_dir=args.model_cache_path,
                                                add_eos_token=args.add_eos_token, 
                                                add_bos_token=args.add_bos_token,
                                                use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, 
                                                token=access_token,
                                                trust_remote_code=args.trust_remote_code, 
                                                cache_dir=args.model_cache_path,
                                                add_eos_token=args.add_eos_token, 
                                                add_bos_token=args.add_bos_token,
                                                use_fast=True)
    
# THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
# good one for LLama is 18610
if args.pad_token_id is not None:
    logger.info("Using pad token id %d", args.pad_token_id)
    tokenizer.pad_token_id = args.pad_token_id

if tokenizer.pad_token_id is None:
    logger.info("Pad token id is None, setting to eos token id...")
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Quanitization
if args.use_int4:
    logger.info("Using int4 quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    optimizer = "adamw_bnb_8bit"
elif args.use_int8:
    logger.info("Using int8 quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    optimizer = "adamw_bnb_8bit"
else:
    logger.warning("Using no quantization")
    bnb_config = None
    optimizer = "adamw_torch"
    
# ================ Fine-tuning method ================
if args.peft == "lora":
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
elif args.peft == "prefix-tuing":
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_virtual_tokens=args.p_tokens,
        encoder_hidden_size=args.p_hidden)
elif args.peft == "p-tuing":
    peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.p_tokens,
        encoder_hidden_size=args.p_hidden)
elif args.peft == "ia3":
    peft_config = IA3Config(
        peft_type="IA3",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["k_proj", "v_proj", "down_proj"],
        feedforward_modules=["down_proj"],
    )
    
# ================ Model ================
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                             token=access_token, 
                                             quantization_config=bnb_config,
                                             trust_remote_code=args.trust_remote_code, 
                                             cache_dir=args.model_cache_path,
                                             torch_dtype=torch_dtype, 
                                             config=config, 
                                             **kwargs)

# flash attention
if use_flash_attention:
    from .llama_patch import llama_forward_with_flash_attn
    assert model.model.layers[0].self_attn.forward.__doc__ == llama_forward_with_flash_attn.__doc__, "Model is not using flash attention"

# kbit training
if not args.disable_peft:
    logger.info("Using PEFT...")
    if args.use_int4 or args.use_int8:
        logger.info("Preparing model for kbit training...")
        model = prepare_model_for_kbit_training(model)
        if use_flash_attention:
            from .llama_patch import upcast_layer_for_flash_attention
            logger.info("Upcasting flash attention layers...")
            model = upcast_layer_for_flash_attention(model, torch_dtype)
    logger.info("Getting PEFT model...")
    model = get_peft_model(model, peft_config)
else:
    logger.info("Using Full Finetuning")

print_trainable_parameters(model)

with accelerator.main_process_first():
    train_dataset, valid_dataset = dataset_prepare(args, tokenizer=tokenizer)

logger.info(f"Length of Train dataset: {len(train_dataset)}, Valid dataset: {len(valid_dataset)}")
    
logger.info(f"Training with {Accelerator().num_processes} GPUs")
training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir=args.output_dir,
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    num_train_epochs=args.epochs,
    eval_steps=args.eval_steps,
    save_steps=args.save_epochs,
    logging_steps=args.log_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size * 2,
    optim=optimizer,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.warmup_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    weight_decay=args.weight_decay,
    adam_epsilon=1e-6,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=args.save_limit,
    bf16=True if torch.cuda.is_bf16_supported() else False,
    fp16=False if torch.cuda.is_bf16_supported() else True,
)

# get trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
)

# train
trainer.train()