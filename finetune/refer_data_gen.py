import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

import argparse
from data.factory import DataFactory
from datasets import Dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaTokenizer

accelerator = Accelerator()

# ======================== Arguments ======================== #
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="openai-community/gpt2")
parser.add_argument("--target_model", type=str)
parser.add_argument("--dataset_name", type=str, default="wikitext-2-raw-v1")
parser.add_argument("--dataset_config_name", type=str, default=None,
                    help="The configuration name of the dataset to use (via the datasets library).")
parser.add_argument("--cache_path", type=str, default="./cache")
parser.add_argument("--use_dataset_cache", action="store_true", default=True)
parser.add_argument("--split_dataset", action="store_true", default=False)
parser.add_argument("--save_path", type=str, required=True, help="The path to save the generated dataset.")
parser.add_argument("--packing", action="store_true", default=True)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--preprocessing_num_workers", type=int, default=1)
parser.add_argument("--validation_split_percentage", default=0.1,
                    help="The percentage of the train set used as validation set in case there's no validation split")

args = parser.parse_args()

# ======================== Loading ======================== 
config = AutoConfig.from_pretrained(args.model_name)
bnb_config = None
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(args.target_model, 
                                             quantization_config=bnb_config,
                                             torch_dtype=torch_dtype,
                                             config=config,
                                             cache_dir=args.cache_path)

model_type = config.to_dict()["model_type"]
if model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
if tokenizer.pad_token_id is None:
    print("Pad token id is None, setting to eos token id...")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
# Load Datasets
data = DataFactory(data_path=args.dataset_name, 
                   args=args,
                   tokenizer=tokenizer)
train_dataset, valid_dataset = data.train_dataset, data.test_dataset
prompt_dataset = Dataset.from_dict(train_dataset[:int(len(train_dataset) * 0.5)])
prompt_dataloader = DataLoader(prompt_dataset, batch_size=1)

# ======================== Generate ========================
if not os.path.exists(args.save_path):
    print(f"Path {args.save_path} does not exist. Try to make it.")
    os.makedirs(args.save_path)
    
generated_dataset = {
    "text": [],
}

for text in tqdm(prompt_dataloader, desc="Generating"):
    prompt = (text["text"])
    input_ids = tokenizer(prompt, 
                          return_tensors="pt", 
                          padding=True).input_ids.to(accelerator.device)
    clipped_ids = input_ids[:, :16]
    if hasattr(model, "module"):
        gen_tokens = model.module.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )
    else:
        gen_tokens = model.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )
    if model_type == "llama":
        gen_tokens = gen_tokens[:, 1:]
    gen_text = tokenizer.batch_decode(gen_tokens)
    generated_dataset["text"].extend(gen_text)
    
generated_dataset = Dataset.from_dict(generated_dataset)
generated_dataset.save_to_disk(args.save_path)

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    concatenated_dataset = None
    for sub_dir in os.listdir(args.save_dir):
        data_path = os.path.join(args.save_dir, sub_dir)
        if os.path.isdir(data_path):
            if concatenated_dataset is None:
                concatenated_dataset = load_from_disk(data_path)
            else:
                dataset = load_from_disk(data_path)
                concatenated_dataset = concatenate_datasets([concatenated_dataset, dataset])
    concatenated_dataset.save_to_disk(args.save_dir)