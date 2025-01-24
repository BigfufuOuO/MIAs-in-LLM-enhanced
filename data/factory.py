from .echr import EchrDataset
from .enron import EnronDataset
from .provider import dataset_prepare
import datasets

class DataFactory:
    def __init__(self, data_path, args, tokenizer):
        self.data_path = data_path
        self.train_dataset, self.test_dataset = self.get_dataset(args, tokenizer)

    def get_dataset(self, args, tokenizer):
        if args.dataset_name == "data/echr":
            data = EchrDataset(data_path=self.data_path)
            return data.train_set(), data.test_set()
        elif args.dataset_name == "data/enron":
            data = EnronDataset(data_path=self.data_path)
            return data.train_set(), data.test_set()
        else:
            train, test = dataset_prepare(args=args, tokenizer=tokenizer)
            return train, test
        
    def get_string_length(self):
        """
        Calculate the average length of the string in the dataset.
        """
        def compute_length(example):
            example['text_length'] = len(example['text'])
            return example

        train_text_length = self.train_dataset.map(compute_length)
        test_text_length = self.test_dataset.map(compute_length)
        
        train_avg_length = train_text_length.map(lambda x: x['text_length']).mean().item()
        test_avg_length = test_text_length.map(lambda x: x['text_length']).mean().item()
        
        return train_avg_length, test_avg_length
