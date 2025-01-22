from models.finetuned_llms import FinetunedCasualLM

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--int8', action='store_true')
parser.add_argument('--half', action='store_true')
args = parser.parse_args()

llm = FinetunedCasualLM(args=args, model_path='openai-community/gpt2')
output = llm.query('Hello, my name is')
print(output)
