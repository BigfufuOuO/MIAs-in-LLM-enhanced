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
    parser.add_argument('--target_model', type=str, default="openai-community/gpt2", help="The target model to attack.")
    parser.add_argument('--refer_model', type=str, help="The reference model to take reference.")
    parser.add_argument('--mask_model', type=str, help="The mask model to use.")
    parser.add_argument('--model_name', type=str, default=None, help="The NAME to the original target model.")
    parser.add_argument('--data_path', type=str, default="data/echr", help="The path to the data.")

    # attack parameter
    parser.add_argument('--metric', type=str, nargs="+", required=True, help="The metric to use for the attack.")
    parser.add_argument('--n_neighbor', type=int, default=25, help="The number of neighbors to consider.")
    parser.add_argument("--n_perturbed", type=int, default=10, help="The number of support vectors to consider.")
    parser.add_argument("--mask_ratio", type=float, default=0.2, help="The ratio of the mask.")

    # precision
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')

    # others
    parser.add_argument("--validation_split_percentage", default=0.2, help="The percentage of the train set used as validation set in case there's no validation split")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1, help="The number of workers to use for the preprocessing.")
    parser.add_argument("--use_dataset_cache", action="store_true", default=False, help="Whether to use dataset cache.")
    parser.add_argument("--load_from_disk", action="store_true", default=False, help="Whether to load the dataset from disk.")

    # debug
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--split_dataset", action="store_true", help="Use small dataset.")
    parser.add_argument("--split_begin", type=float, default=0.0, help="The beginning of the split.")
    parser.add_argument("--split_end", type=float, default=0.2, help="The end of the split.")

    # finetune
    parser.add_argument("--output_dir", type=str, help="The output directory to save the finetuned models.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Whether to use gradient checkpointing.")
    
    # result
    parser.add_argument("--result_dir", type=str, default="./results", help="The output directory to save the results.")

    args = parser.parse_args()
    return args
