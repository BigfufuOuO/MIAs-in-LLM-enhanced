import pprint
from datasets import Dataset

data = Dataset.load_from_disk('./data/neighbor_data/ag_news/bs64/neighbor/train_neighbor')

pprint.pprint(data['text'][0])