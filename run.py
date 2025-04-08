from data.factory import DataFactory
from attacks.MIA import MemberInferenceAttack
from attacks.utils import load_target_models, load_refer_models
from parse_args import get_args
from finetune.utils import get_logger
import pandas as pd
import torch

args = get_args()
logger = get_logger("MIA", args.log_dir, "info")

# ======================== MAIN ========================
if __name__ == '__main__':
    target_llm = load_target_models(args, args.model_path)
       
    # data
    data = DataFactory(data_path=args.dataset_name, args=args, tokenizer=target_llm.tokenizer)
    logger.info(f'Average Length of the string in dataset:{data.get_string_length()}')
    logger.info(f'Data preview: {data.get_preview()[0]}\n{data.get_preview()[1]}')

    for metric in args.metric:
        with torch.no_grad():
            refer_llm, mask_llm= load_refer_models(args, logger, args.model_path, metric) 
            # excute attack
            attack = MemberInferenceAttack(logger=logger, metric=metric, ref_model=refer_llm,
                                        mask_model=mask_llm)
            results = attack.execute(target_llm, 
                                    data.train_dataset,
                                    data.test_dataset,
                                    args=args)
            results = attack.evaluate(args, results)
            
            # Display result
            # pd.set_option('display.max_columns', None)
            # pd.set_option('display.expand_frame_repr', False)  # Avoid line break
            result_df = pd.DataFrame.from_dict(results, orient='index')
            logger.info("=========ENDING==========")
            logger.info(result_df)
        
