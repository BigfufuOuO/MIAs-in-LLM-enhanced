from data.factory import DataFactory
from models.finetuned_llms import FinetunedCasualLM
from attacks.MIA import MemberInferenceAttack
import pandas as pd

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
parser.add_argument('--refer_model', type=str, help="The reference model to take reference.")
parser.add_argument('--model_name', type=str, default=None, help="The NAME to the original target model.")
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
parser.add_argument("--small_dataset", action="store_true", help="Use small dataset.")

# result
parser.add_argument("--result_dir", type=str, default="./results", help="The output directory to save the results.")

args = parser.parse_args()



# ======================== MAIN ========================
if __name__ == '__main__':
    target_llm = FinetunedCasualLM(args=args,
                                model_path=args.target_model,)
    if args.refer_model:
        refer_llm = FinetunedCasualLM(args=args,
                                    model_path=args.refer_model,)
    else:
        refer_llm = None

    # data
    data = DataFactory(data_path=args.dataset_name, args=args, tokenizer=target_llm.tokenizer)
    print('Average Length of the string in dataset:', data.get_string_length())
    print('Data preview:', data.get_preview()[0], '\n', data.get_preview()[1])

    # excute attack
    attack = MemberInferenceAttack(metric=args.metric, ref_model=refer_llm)
    results = attack.execute(target_llm, 
                            data.train_dataset,
                            data.test_dataset,)
    results = attack.evaluate(args, results)
    
    # Display result
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)  # Avoid line break
    result_df = pd.DataFrame.from_dict(results, orient='index').T
    print("=========ENDING==========")
    print(result_df)
