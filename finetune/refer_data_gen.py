import os
import torch
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import numpy as np
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
parser.add_argument("--model_path", type=str, default="openai-community/gpt2")
parser.add_argument("--token", type=str, default=None, help="The Hugging Face authentication token.")
parser.add_argument("--block_size", type=int, default=32)

parser.add_argument("--target_model", type=str)
parser.add_argument("--dataset_name", type=str, default="wikitext-2-raw-v1")
parser.add_argument("--dataset_config_name", type=str, default=None,
                    help="The configuration name of the dataset to use (via the datasets library).")
parser.add_argument("--cache_path", type=str)
parser.add_argument("--dataset_cache_path", type=str, default="./cache/datasets", help="The dataset cache path.")
parser.add_argument("--use_dataset_cache", action="store_true", default=False)
parser.add_argument("--save_path", type=str, required=True, help="The path to save the generated dataset.")
parser.add_argument("--load_from_disk", action="store_true", default=False, help="Load dataset from disk.")
parser.add_argument("--packing", action="store_true", default=False)
parser.add_argument("--preprocessing_num_workers", type=int, default=1)
parser.add_argument("--validation_split_percentage", default=0.4,
                    help="The percentage of the train set used as validation set in case there's no validation split")

# debug
parser.add_argument("--debug", action="store_true", help="Debug mode.")
parser.add_argument("--split_dataset", action="store_true", help="Use small dataset.", default=False)
parser.add_argument("--split_begin", type=float, default=0.0, help="The beginning of the split.")
parser.add_argument("--split_end", type=float, default=0.2, help="The end of the split.")
parser.add_argument("--split_train_begin", type=int, default=0, help="The index of the beginning of the train set in the split.")
parser.add_argument("--split_test_begin", type=int, default=0, help="The index of the beginning of the test set in the split.")
parser.add_argument("--split_train_num", type=int, help="The number of examples in the train set in the split.")
parser.add_argument("--split_test_num", type=int, help="The number of examples in the test set in the split.")

args = parser.parse_args()

# ======================== Loading ======================== 
config = AutoConfig.from_pretrained(args.model_path)
bnb_config = None
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

LOSS_THRESHOLD = 2.5
if args.target_model:
    model_path = args.target_model
else:
    model_path = args.model_path
    block_size = args.block_size
    dataset_name = args.dataset_name
    model_path = f"./ft_llms/{model_path}/{dataset_name}/bs{block_size}/target_base"
    # find dirs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    else:
        dirs = os.listdir(model_path)
        # open the last one
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
        state_path = os.path.join(model_path, dirs[-1])
        # state_path = os.path.join(state_path, "trainer_state.json")
        # if not os.path.exists(state_path):
        #     raise FileNotFoundError(f"State path {state_path} does not exist.")
        # else:
        #     with open(state_path, "r") as f:
        #         state = json.load(f)
        #     model_path = state["best_model_checkpoint"]
        model_path = state_path
            
print(f"Loading model from {model_path}")
        
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             token=args.token,
                                             quantization_config=bnb_config,
                                             torch_dtype=torch_dtype,
                                             config=config,
                                             device_map="auto",
                                             cache_dir=args.cache_path)

model_type = config.to_dict()["model_type"]
if model_type == "llaa":
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,)
    
if tokenizer.pad_token_id is None:
    print("Pad token id is None, setting to eos token id...")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
model.config.pad_token_id = model.config.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
    
# Load Datasets
data = DataFactory(data_path=args.dataset_name, 
                   args=args,
                   tokenizer=tokenizer)
train_dataset, valid_dataset = data.train_dataset, data.test_dataset
train_preview, valid_preview = data.get_preview()
print(f"Train dataset preview: {train_preview}")
print(f"Valid dataset preview: {valid_preview}")
prompt_dataset = Dataset.from_dict(train_dataset[:int(len(train_dataset) * 0.2)])
prompt_dataloader = DataLoader(prompt_dataset, batch_size=1)


# ======================== Generate ========================
if not os.path.exists(args.save_path):
    print(f"Path {args.save_path} does not exist. Try to make it.")
    os.makedirs(args.save_path)
    

def generate_text(text):
    prompt = text
    inputs = tokenizer(prompt, 
                          return_tensors="pt", 
                          truncation=True,
                          max_length=args.block_size,
                          padding=True).to(accelerator.device)
    input_ids = inputs.input_ids
    clipped_ids = inputs.input_ids[:, :16]
    attention_mask = inputs.attention_mask[:, :16]
    if hasattr(model, "module"):
        gen_tokens = model.module.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
            attention_mask=attention_mask,
        )
    else:
        gen_tokens = model.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
            max_new_tokens=None,
            attention_mask=attention_mask,
        )
    if model_type == "llama":
        gen_tokens = gen_tokens[:, 1:]
    gen_text = tokenizer.batch_decode(gen_tokens)
    return {
        "text": gen_text
    }
    
generated_dataset = prompt_dataset.map(
    lambda example: generate_text(example['text']),
    batched=True,
    batch_size=64,
    desc='Generating',
)

generated_dataset.save_to_disk(args.save_path)

# accelerator.wait_for_everyone()

# if accelerator.is_main_process:
#     concatenated_dataset = None
#     for sub_dir in os.listdir(args.save_dir):
#         data_path = os.path.join(args.save_dir, sub_dir)
#         if os.path.isdir(data_path):
#             if concatenated_dataset is None:
#                 concatenated_dataset = load_from_disk(data_path)
#             else:
#                 dataset = load_from_disk(data_path)
#                 concatenated_dataset = concatenate_datasets([concatenated_dataset, dataset])
#     concatenated_dataset.save_to_disk(args.save_dir)