import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os
import pandas as pd
import json
from models.finetuned_llms import FinetunedCasualLM
from models.mask_llms import MaskLanguageModel

def draw_auc_curve(fpr, tpr,
                   title='ROC curve',
                   save_path='./results',
                   metric='loss'):
    """
    Draw the ROC curve and save it to the save_path.
    """
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # draw title
    plt.title(title)
    
    # save the figure
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'{metric}.png'))
    
def save_to_csv(results, 
                mode='default', 
                save_path="./results"):
    """
    Save the results to a csv file.
    """
    df = pd.DataFrame.from_dict(results, orient='index').T
    save_path = os.path.join(save_path, f'results_{mode}.csv')
    df.to_csv(save_path, 
              mode='a',
              header=True if not os.path.exists(save_path) else False,
              index=False)
    
def load_target_models(args,
                       model_name: str,
                       llm_dir: str = './ft_llms',
                        ):
    """
    Load the target models from accoding to the model_name.
    """
    if args.target_model is None:
        dataset_name = args.dataset_name
        block_size = args.block_size
        target_path = os.path.join(llm_dir, model_name)
        target_path = os.path.join(target_path, f'{dataset_name}/bs{block_size}/target_base')
        target_files = os.listdir(target_path)
        # sort the files
        target_files = sorted(target_files, key=lambda x: int(x.split('-')[-1]))
        target_path = os.path.join(target_path, target_files[-1])
        target_llm = FinetunedCasualLM(args=args,
                                       model_path=target_path)
    else:
        kwargs = {}
        if args.mode == "ft-phase":
            trainer_state_path = os.path.join(args.target_model, "trainer_state.json")
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            kwargs["train_loss"] = trainer_state["log_history"][-2]
            kwargs["eval_step"] = trainer_state["log_history"][-1]
        target_llm = FinetunedCasualLM(args=args,
                                       model_path=args.target_model,
                                       **kwargs)
        
            
        
    return target_llm
    
   
    
def load_refer_models(args,
                      logger,
                      model_name: str,
                      metric: str,
                      llm_dir: str = './ft_llms',
                    mask_llm: str = "FacebookAI/roberta-base",):
    """
    Load the refer models from accoding to the model_name and metric.
    """
    refer_methods = ['refer-base', 'refer-orcale', 'lira-base', 'lira-orcale', 'neighbor', 'spv_mia']
    if metric not in refer_methods:
        logger.warning(f"Warning: {metric} not in refer_methods. Please check if it is registered.")
        return None, None
    
    if args.refer_model is None:
        if 'base' in metric:
            refer_llm = FinetunedCasualLM(args=args,
                                          model_path=model_name)
        elif 'orcale' in metric:
            dataset_name = args.dataset_name
            block_size = args.block_size
            refer_path = os.path.join(llm_dir, model_name)
            refer_path = os.path.join(refer_path, f'{dataset_name}/bs{block_size}/refer_orcale')
            refer_files = os.listdir(refer_path)
            refer_files = sorted(refer_files, key=lambda x: int(x.split('-')[-1]))
            refer_path = os.path.join(refer_path, refer_files[-1])
            refer_llm = FinetunedCasualLM(args=args,
                                           model_path=refer_path)
        elif metric == 'neighbor':
            refer_llm = FinetunedCasualLM(args=args,
                                          model_path=mask_llm)
        elif metric == 'spv_mia':
            dataset_name = args.dataset_name
            block_size = args.block_size
            refer_path = os.path.join(llm_dir, model_name)
            refer_path = os.path.join(refer_path, f'{dataset_name}/bs{block_size}/self_prompt')
            refer_files = os.listdir(refer_path)
            refer_path = os.path.join(refer_path, refer_files[-1])
            refer_llm = FinetunedCasualLM(args=args,
                                           model_path=refer_path)
        else:
            raise ValueError(f"Invalid metric: {metric}")
    
    if metric != 'spv_mia':
        return refer_llm, None
    else:
        if args.mask_model is None:
            spv_llm = MaskLanguageModel(args=args,
                                        model_path=mask_llm)
        else:
            spv_llm = MaskLanguageModel(args=args,
                                        model_path=args.mask_model)
        return refer_llm, spv_llm
        
        

