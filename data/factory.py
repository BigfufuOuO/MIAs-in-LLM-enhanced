from .echr import EchrDataset
from .enron import EnronDataset
from .provider import dataset_prepare
import datasets
import numpy as np

class DataFactory:
    def __init__(self, data_path, args, tokenizer):
        self.data_path = data_path
        self.train_dataset, self.test_dataset = self.get_dataset(args, tokenizer)
        if args.debug:
            # for debugging
            self.train_dataset = self.train_dataset.select(list(range(len(self.train_dataset) // 10)))
            self.test_dataset = self.test_dataset.select(list(range(len(self.test_dataset) // 10)))
            
        if args.small_dataset:
            print("Using small dataset.")
            np.random.seed(42)
            # select 1/5 of the dataset
            selected_indices = np.random.choice(len(self.train_dataset), len(self.train_dataset) // 5, replace=False)
            self.train_dataset = self.train_dataset.select(selected_indices)
            selected_indices = np.random.choice(len(self.test_dataset), len(self.test_dataset) // 5, replace=False)
            self.test_dataset = self.test_dataset.select(selected_indices)


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

        train_text_length = self.train_dataset.map(compute_length,
                                                   load_from_cache_file=False,
                                                   desc="Computing length of the text(train)")
        test_text_length = self.test_dataset.map(compute_length,
                                                 load_from_cache_file=False,
                                                 desc="Computing length of the text(test)")
        
        train_avg_length = np.mean(train_text_length['text_length'])
        test_avg_length = np.mean(test_text_length['text_length'])
        
        return train_avg_length, test_avg_length
    
    def get_preview(self):
        """
        Get the preview of the dataset.
        """
        return self.train_dataset[0], self.test_dataset[0]
