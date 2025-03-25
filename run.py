from data.factory import DataFactory
from attacks.MIA import MemberInferenceAttack
from attacks.utils import load_target_models, load_refer_models
from parse_args import get_args
import pandas as pd

args = get_args()

# ======================== MAIN ========================
if __name__ == '__main__':
    target_llm = load_target_models(args, args.model_path)
       
    # data
    data = DataFactory(data_path=args.dataset_name, args=args, tokenizer=target_llm.tokenizer)
    print('Average Length of the string in dataset:', data.get_string_length())
    print('Data preview:', data.get_preview()[0], '\n', data.get_preview()[1])

    for metric in args.metric:
        refer_llm, mask_llm= load_refer_models(args, args.refer_model, metric) 
        # excute attack
        attack = MemberInferenceAttack(metric=metric, ref_model=refer_llm,
                                       mask_model=mask_llm)
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
