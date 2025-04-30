from flair.data import Sentence
from flair.nn import Classifier
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, ".."))
from data.factory import DataFactory
from finetune.utils import get_logger
from transformers import AutoTokenizer
import datasets
import argparse

logger = get_logger(__name__)

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
parser.add_argument("--split_shuffle", type=int, default=1, help="Whether to shuffle the dataset before splitting.")


args = parser.parse_args()

tokenizer =  AutoTokenizer.from_pretrained(args.model_path, 
                                            device_map="auto",
                                            use_fast=True)
data = DataFactory(data_path=args.dataset_name, args=args, tokenizer=tokenizer)
logger.info(f"Data preview: {data.get_preview()}")
tagger = Classifier.load('ner-large')

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    logger.info(f"Creating directory {args.save_path}")


def replace_entities(texts: list,):
    sentences = [Sentence(text) for text in texts['text']]
    tagger.predict(sentences)
    replace_texts = []
    for sentence in sentences:
        original_text = sentence.to_tokenized_string()
        for entity in sorted(sentence.get_spans('ner'), key=lambda x: x.start_position, reverse=True):
            start = entity.start_position
            end = entity.end_position
            entity_type = entity.tag
            # Replace the entity with its type
            original_text = original_text[:start] + f"<{entity_type}>" + original_text[end:]
        replace_texts.append(original_text)
    return {
        "text": replace_texts,
    }
    
train_dataset = data.train_dataset.map(
    replace_entities,
    batched=True,
    batch_size=64,
    desc="Generating",
)

test_dataset = data.test_dataset.map(
    replace_entities,
    batched=True,
    batch_size=64,
    desc="Generating",
)

# get two datesets together
combined_dataset = datasets.concatenate_datasets([train_dataset, test_dataset])

logger.info(f"Combined dataset: {combined_dataset}")

combined_dataset.save_to_disk(args.save_path)
logger.info(f"Saved dataset to {args.save_path}")


        