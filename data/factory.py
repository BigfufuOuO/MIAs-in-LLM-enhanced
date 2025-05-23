from .provider import dataset_prepare
from datasets import Dataset
import numpy as np
import os

from finetune.utils import get_logger
logger = get_logger(__name__, level="info")

class DataFactory:
    def __init__(self, data_path, args, tokenizer):
        self.data_path = data_path
        self.train_dataset, self.test_dataset = self.get_dataset(args, tokenizer)
        

    def get_dataset(self, args, tokenizer):
        train, test = dataset_prepare(args=args, tokenizer=tokenizer)
            
        if args.debug:
            train = train.select(list(range(len(train) // 200)))
            test = test.select(list(range(len(test) // 200)))
            return train, test
        
        if args.split_dataset:
            logger.info("Using split dataset.")
            np.random.seed(42)
            # reshuffle the indices
            train_indices = np.arange(len(train))
            test_indices = np.arange(len(test))
            if args.split_shuffle:
                train_indices = np.random.permutation(train_indices)
                test_indices = np.random.permutation(test_indices)
            
            if args.split_train_num:
                start = args.split_train_begin
                end = start + args.split_train_num
                if end > len(train_indices):
                    logger.warning(f"Split train num [{start}:{end}] is out of range {len(train_indices)}, using all the data.")
                train = train.select(train_indices[start:end])
            else:
                start = int(len(train) * args.split_begin)
                end = int(len(train) * args.split_end)
                train = train.select(train_indices[start:end])
            
            if args.split_test_num:
                start = args.split_test_begin
                end = start + args.split_test_num
                if end > len(train_indices):
                    logger.warning(f"Split train num [{start}:{end}] is out of range {len(train_indices)}, using all the data.")
                test = test.select(test_indices[start:end])
            else:
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
