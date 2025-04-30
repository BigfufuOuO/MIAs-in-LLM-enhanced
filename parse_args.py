import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # general
    # TODO: torch device specification

    # dataset
    parser.add_argument('--dataset_name', type=str, default="data/echr", help="The dataset to use.")
    parser.add_argument('--dataset_config_name', type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_cache_path", type=str, default="./cache/datasets", help="The dataset cache path.")
    parser.add_argument("--use_neighbor_cache", action="store_true", default=False, help="Whether to use neighbor dataset cache.")

    # model definition
    parser.add_argument('--target_model', type=str, help="The target model to attack.")
    parser.add_argument('--token', type=str, help="The huggingface token to use.")
    parser.add_argument('--refer_model', type=str, help="The reference model to take reference.")
    parser.add_argument('--mask_model', type=str, help="The mask model to use.")
    parser.add_argument('--model_path', type=str, default=None, help="The NAME to the original target model.")

    # attack parameter
    parser.add_argument('--metric', type=str, nargs="+", required=True, help="The metric to use for the attack.")
    parser.add_argument('--n_neighbor', type=int, default=25, help="The number of neighbors to consider.")
    parser.add_argument("--n_perturbed", type=int, default=10, help="The number of support vectors to consider.")
    parser.add_argument("--mask_ratio", type=float, default=0.2, help="The ratio of the mask.")

    # precision
    parser.add_argument('--int4', action='store_true')
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')

    # others
    parser.add_argument("--validation_split_percentage", default=0.4, help="The percentage of the train set used as validation set in case there's no validation split")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1, help="The number of workers to use for the preprocessing.")
    parser.add_argument("--use_dataset_cache", action="store_true", default=False, help="Whether to use dataset cache.")
    parser.add_argument("--load_from_disk", action="store_true", default=False, help="Whether to load the dataset from disk.")
    parser.add_argument("--log_dir", type=str, required=True, help="The log directory to save the logs.")
    parser.add_argument("--mode", type=str, required=True, choices=["default", "neighbor",
                                                                    "ft-phase", "defense"], help="The mode to run the program in.")
    parser.add_argument("--defense", type=str, choices=['dp_8', 'dp_linear', 'scrub'], help="The random seed to use.")

    # debug
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--split_dataset", action="store_true", help="Use small dataset.")
    parser.add_argument("--split_begin", type=float, default=0.0, help="The beginning of the split.")
    parser.add_argument("--split_end", type=float, default=0.2, help="The end of the split.")
    parser.add_argument("--split_train_begin", type=int, default=0, help="The index of the beginning of the train set in the split.")
    parser.add_argument("--split_test_begin", type=int, default=0, help="The index of the beginning of the test set in the split.")
    parser.add_argument("--split_train_num", type=int, help="The number of examples in the train set in the split.")
    parser.add_argument("--split_test_num", type=int, help="The number of examples in the test set in the split.")
    parser.add_argument("--split_shuffle", type=int, default=1, help="Whether to shuffle the dataset before splitting.")

    
    # finetune
    parser.add_argument("--output_dir", type=str, help="The output directory to save the finetuned models.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Whether to use gradient checkpointing.")
    parser.add_argument("--load_bin", action="store_true", default=False, help="Whether to load the bin model.")
    
    # result
    parser.add_argument("--result_dir", type=str, default="./results", help="The output directory to save the results.")

    args = parser.parse_args()
    return args
