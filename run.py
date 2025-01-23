from data.echr import EchrDataset
from models.finetuned_llms import FinetunedCasualLM
from attacks.MIA import MemberInferenceAttack

import argparse
parser = argparse.ArgumentParser()

# general
# TODO: torch device specification

# dataset
parser.add_argument('--dataset', type=str, default="data/echr", help="The dataset to use.")

# model definition
parser.add_argument('--target_model', type=str, default="openai-community/gpt2", required=True, help="The target model to attack.")
parser.add_argument('--data_path', type=str, default="data/echr", help="The path to the data.")

# attack parameter
parser.add_argument('--metric', type=str, default="ppl", required=True, help="The metric to use for the attack.")
parser.add_argument('--n_neighbor', type=int, default=50, help="The number of neighbors to consider.")

# precision
parser.add_argument('--int8', action='store_true')
parser.add_argument('--half', action='store_true')
args = parser.parse_args()



# main entry
data = EchrDataset(data_path="data/echr")
target_llm = FinetunedCasualLM(args=args,
                               model_path="openai-community/gpt2")
attack = MemberInferenceAttack(metric=args.metric,)
results = attack.execute(target_llm, 
                         data.train_set(),
                         data.test_set(),)
results = attack.evaluate(results)
print('RESULT:\n', results)