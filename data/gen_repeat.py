import datasets
import argparse
from transformers import AutoTokenizer
from data.factory import DataFactory
from finetune.utils import get_logger

logger = get_logger(__name__, "info")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='number of repeats')
parser.add_argument('--dataset_name', type=str, help='path to dataset')
parser.add_argument('--output_path', type=str, help='path to output dataset')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
data = DataFactory(data_path=args.dataset_name, args=args, tokenizer=tokenizer)
train_dataset = data.train_dataset
test_dataset = data.test_dataset


def gen_repeat(data,
               repeat_list: list[int],
               group_size: int = 200,):
    pass

repeat_list = [1, 8, 32, 128]