import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import torch
from data.factory import DataFactory
from models.finetuned_llms import FinetunedCasualLM
parser = argparse.ArgumentParser()

# ======================== Data Generation ========================
# dataset
parser.add_argument('--dataset_name', type=str, default="data/echr", help="The dataset to use.")
parser.add_argument('--dataset_config_name', type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
parser.add_argument("--dataset_cache_path", type=str, default="./cache/datasets", help="The dataset cache path.")

parser.add_argument('--refer_model', type=str, help="The reference model to take reference.")
parser.add_argument('--mask_model', type=str, help="The mask model to use.")
parser.add_argument('--target_model', type=str, help="The target model to evaluate.")
parser.add_argument('--data_path', type=str, default="data/echr", help="The path to the data.")

parser.add_argument("--n_perturbed", type=int, default=10, help="The number of support vectors to consider.")
parser.add_argument("--mask_ratio", type=float, default=0.2, help="The ratio of the mask.")

# precision
parser.add_argument('--int8', action='store_true')
parser.add_argument('--half', action='store_true')

# others
parser.add_argument("--validation_split_percentage", default=0.2, help="The percentage of the train set used as validation set in case there's no validation split")
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--packing", action="store_true")
parser.add_argument("--preprocessing_num_workers", type=int, default=1, help="The number of workers to use for the preprocessing.")
parser.add_argument("--use_dataset_cache", action="store_true", default=False, help="Whether to use dataset cache.")
parser.add_argument("--load_from_disk", action="store_true", default=False, help="Whether to load the dataset from disk.")

# debug
parser.add_argument("--debug", action="store_true", help="Debug mode.")
parser.add_argument("--split_dataset", action="store_true", help="Use small dataset.")
parser.add_argument("--split_begin", type=float, default=0.0, help="The beginning of the split.")
parser.add_argument("--split_end", type=float, default=0.2, help="The end of the split.")

args = parser.parse_args()

if args.int8:
    bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
mask_model = AutoModelForSeq2SeqLM.from_pretrained(args.mask_model,
                                                    quantization_config=bnb_config,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto")
mask_tokenizer = AutoTokenizer.from_pretrained(args.mask_model)

target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)

# data
data = DataFactory(data_path=args.dataset_name, args=args, tokenizer=mask_tokenizer)