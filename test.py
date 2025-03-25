import pprint
import datasets

data = datasets.load_dataset('LLM-PBE/enron-email', split='train')

print(dir(data))