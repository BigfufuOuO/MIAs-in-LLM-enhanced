from data.factory import DataFactory
from models.finetuned_llms import FinetunedCasualLM
from attacks.MIA import MemberInferenceAttack
from parse_args import get_args
import pandas as pd

args = get_args()

# ======================== MAIN ========================
if __name__ == '__main__':
    target_llm = FinetunedCasualLM(args=args,
                                model_path=args.target_model,)
    if args.refer_model:
        refer_llm = FinetunedCasualLM(args=args,
                                    model_path=args.refer_model,)
    else:
        refer_llm = None
        
    if args.mask_model:
        mask_model = args.mask_model
    else:
        mask_model = None

    # data
    data = DataFactory(data_path=args.dataset_name, args=args, tokenizer=target_llm.tokenizer)
    print('Average Length of the string in dataset:', data.get_string_length())
    print('Data preview:', data.get_preview()[0], '\n', data.get_preview()[1])

    # excute attack
    attack = MemberInferenceAttack(metric=args.metric, ref_model=refer_llm,
                                   mask_model=mask_model)
    results = attack.execute(target_llm, 
                            data.train_dataset,
                            data.test_dataset,
                            args=args)
    results = attack.evaluate(args, results)
    
    # Display result
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)  # Avoid line break
    result_df = pd.DataFrame.from_dict(results, orient='index').T
    print("=========ENDING==========")
    print(result_df)
