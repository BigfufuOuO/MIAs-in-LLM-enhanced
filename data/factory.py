from .echr import EchrDataset
from .enron import EnronDataset
from .provider import dataset_prepare
import datasets
import numpy as np

class DataFactory:
    def __init__(self, data_path, args, tokenizer):
        self.data_path = data_path
        self.train_dataset, self.test_dataset = self.get_dataset(args, tokenizer)
        

    def get_dataset(self, args, tokenizer):
        if args.dataset_name == "data/echr":
            data = EchrDataset(data_path=self.data_path)
            train = data.train_set()
            test = data.test_set()
        elif args.dataset_name == "data/enron":
            data = EnronDataset(data_path=self.data_path)
            train = data.train_set()
            test = data.test_set()
        else:
            train, test = dataset_prepare(args=args, tokenizer=tokenizer)
            
        if args.debug:
            train = train.select(list(range(len(train) // 10)))
            test = test.select(list(range(len(test) // 10)))
            return train, test
        
        if args.small_dataset:
            print("Using small dataset.")
            np.random.seed(42)
            # select 1/5 of the dataset
            selected_indices = np.random.choice(len(train), len(train) // 5, replace=False)
            train = train.select(selected_indices)
            selected_indices = np.random.choice(len(test), len(test) // 5, replace=False)
            test = test.select(selected_indices)
            return train, test
        
        if args.split_dataset:
            print("Using split dataset.")
            np.random.seed(42)
            # reshuffle the indices
            train_indices = np.arange(len(train))
            test_indices = np.arange(len(test))
            train_indices = np.random.permutation(train_indices)
            test_indices = np.random.permutation(test_indices)
            
            start = int(len(train) * args.split_begin)
            end = int(len(train) * args.split_end)
            train = train.select(train_indices[start:end])
            
            start = int(len(test) * args.split_begin)
            end = int(len(test) * args.split_end)
            test = test.select(test_indices[start:end])
            return train, test
            
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
