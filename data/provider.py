import datasets
import os
import numpy as np
from finetune.utils import get_logger
logger = get_logger(__name__, level="info")


def packing_texts(examples,):
    """
    This function is used to pack the texts into chunks of `block_size` tokens.
    For example, before packing, the train_dataset and text column looks like:
    ```
    train_dataset = Dataset({
        features: ['text', 'label'],
        num_rows: 96000
    })
    train_dataset[0] == {'text': 'This is the first text.'}
    ```
    After packing, the train_dataset and text column looks like:
    ```
    train_dataset = Dataset({
        features: ['text'],
        num_rows: 10000
    })
    train_dataset[0] == {'text': 'This is the first text.'}
    ```
    Where the text is packed into chunks of `block_size` tokens, the length of the text is equal to `block_size`,
    the length of dataset is reduced.
    
    Args:
        examples: The examples.
        max_buff_size: The maximum buffer size, which limits the number of characters handled at once.
        block_size: The block size, tokens per chunk.
    """
    more_examples = True
    packed_texts = []
    packed_ids = []
    assert list(examples.keys()) == ["text"], "Only text column is supported for packing."
    iterator = iter(examples["text"])
    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= max_buff_size:
                break
            try:
                buffer.append(next(iterator))
                buffer_len += len(buffer[-1])
            except StopIteration:
                more_examples = False
                break
        
        if not buffer:
            # if buffer is empty, break
            break
            
        tokenized_inputs = tokenizer_(buffer, truncation=True, max_length=1024)["input_ids"] # shape: (num_examples, num_tokens)
        if tokenizer_.bos_token_id:
            new_list = [row[1:] for row in tokenized_inputs]
            tokenized_inputs = new_list
            all_token_ids = np.concatenate(tokenized_inputs)
            for i in range(0, len(all_token_ids), block_size):
                input_ids = all_token_ids[i: i + block_size]
                if len(input_ids) == block_size:
                    packed_ids.append(input_ids)
                    input_text = tokenizer_.decode(input_ids)
                    re_tokenized = tokenizer_.encode(input_text)
                    if re_tokenized[0] == tokenizer_.bos_token_id:
                        if len(re_tokenized) == block_size + 1:
                            packed_texts.append(input_text)
                    else:
                        if len(re_tokenized) == block_size:
                            packed_texts.append(input_text)
        else:
            # concatenate all the tokenized inputs
            all_token_ids = np.concatenate(tokenized_inputs)
            
            for i in range(0, len(all_token_ids), block_size):
                input_ids = all_token_ids[i: i + block_size]
                if len(input_ids) == block_size:
                    packed_ids.append(input_ids)
                    input_text = tokenizer_.decode(input_ids)
                    if len(tokenizer_.encode(input_text)) == block_size:
                        packed_texts.append(input_text)
    
    return {
        "text": packed_texts
    }

def dataset_prepare(args, 
                    tokenizer=None, 
                    num_of_sequences=1024, 
                    chars_per_token=3.6):
    """
    Provide dataset for training or evaluation.
    It also drops the columns that are not needed, for example, the `label` column.
    ```
    train_dataset[0] = {'text': 'This is the first text.', 'label': 0}
    ```
    Affter dropping the `label` column, the dataset looks like:
    ```
    train_dataset[0] = {'text': 'This is the first text.'}
    ```
    
    Args:
        args: The arguments.
        
    Returns:
        train_dataset: The training dataset.
                    Format:
                        ```
                        Dataset({
                            features: ['text'],
                            num_rows: 1000
                        })
                        ```
                    Example:
                        `train_dataset[0] == {'text': 'This is the first text.'}`
        valid_dataset: The validation dataset.
    """
    if args.load_from_disk:
        all_dataset = datasets.load_from_disk(args.dataset_name)
        split_dataset = all_dataset.train_test_split(train_size=1-args.validation_split_percentage, shuffle=False)
        train_dataset = split_dataset["train"]
        valid_dataset = split_dataset["test"]
    else:
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_config_name, split="train")
        num_rows = dataset.num_rows
        # if length of dataset is too large, split it through index
        if num_rows > 10 ** 5:
            split = int(10 ** 5 * (1 - args.validation_split_percentage))
            train_dataset = dataset.select(range(0, split))
            valid_dataset = dataset.select(range(split, 10**5))
        else:   
            train_dataset = datasets.load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{int((1-args.validation_split_percentage)*100)}%]"
            )
            
            valid_dataset = datasets.load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{int((1-args.validation_split_percentage)*100)}%:]",
            )
    
    column = train_dataset.column_names
    possible_text_columns = ["text", "document", "content"]
    text_column = next((col for col in possible_text_columns if col in column), None)
    logger.info(f"Train dataset columns: {column}, select text column: {text_column}")
    
    train_dataset = train_dataset.select_columns(text_column)
    valid_dataset = valid_dataset.select_columns(text_column)
    if text_column != "text":
        train_dataset = train_dataset.rename_column(text_column, "text")
        valid_dataset = valid_dataset.rename_column(text_column, "text")
        
    if args.packing:
        global tokenizer_, block_size, max_buff_size
        block_size = args.block_size
        max_buff_size = block_size * chars_per_token * num_of_sequences
        tokenizer_ = tokenizer
        logger.info(f"Block size: {block_size}, max buffer size: {max_buff_size}")
        # if dir not exists, create it
        save_path = f"{args.dataset_cache_path}/{args.model_path}/{args.dataset_name}/bs{block_size}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # save the packed dataset
        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{save_path}/train_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{save_path}/valid_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        
    return train_dataset, valid_dataset