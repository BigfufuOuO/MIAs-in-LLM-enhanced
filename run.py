from data.factory import DataFactory
from models.finetuned_llms import FinetunedCasualLM
from attacks.MIA import MemberInferenceAttack

import argparse
parser = argparse.ArgumentParser()

# general
# TODO: torch device specification

# dataset
parser.add_argument('--dataset_name', type=str, default="data/echr", help="The dataset to use.")
parser.add_argument('--dataset_config_name', type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
parser.add_argument("--dataset_cache_path", type=str, default="./cache/datasets", help="The dataset cache path.")

# model definition
parser.add_argument('--target_model', type=str, default="openai-community/gpt2", required=True, help="The target model to attack.")
parser.add_argument('--data_path', type=str, default="data/echr", help="The path to the data.")

# attack parameter
parser.add_argument('--metric', type=str, default="ppl", required=True, help="The metric to use for the attack.")
parser.add_argument('--n_neighbor', type=int, default=50, help="The number of neighbors to consider.")

# precision
parser.add_argument('--int8', action='store_true')
parser.add_argument('--half', action='store_true')

# others
parser.add_argument("--validation_split_percentage", default=0.1, help="The percentage of the train set used as validation set in case there's no validation split")
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--packing", action="store_true")
parser.add_argument("--preprocessing_num_workers", type=int, default=1, help="The number of workers to use for the preprocessing.")
parser.add_argument("--use_dataset_cache", action="store_true", default=False, help="Whether to use dataset cache.")

# debug
parser.add_argument("--debug", action="store_true", help="Debug mode.")

args = parser.parse_args()



# main entry
target_llm = FinetunedCasualLM(args=args,
                               model_path="openai-community/gpt2")
data = DataFactory(data_path=args.dataset_name, args=args, tokenizer=target_llm.tokenizer)
attack = MemberInferenceAttack(metric=args.metric,)
results = attack.execute(target_llm, 
                         data.train_dataset,
                         data.test_dataset,)
results = attack.evaluate(results)
print('RESULT:\n', results)
print('Average Length of the words in dataset:', data.get_string_length())