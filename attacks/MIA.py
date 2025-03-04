import torch
import os
import zlib
from .AttackBase import AttackBase
from enum import Enum
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, roc_curve, auc
from collections import defaultdict

from models.finetuned_llms import FinetunedCasualLM
from .functions import function_map
from .utils import draw_auc_curve, save_to_csv
import inspect
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

class MemberInferenceAttack(AttackBase):
    """
    Membership Inference Attack (MIA).

    Note MIA is often used with data extraction to find the training samples in 
    generated samples. For this purpose, top-score samples will be selected.

    Reference implementation: https://github.com/ftramer/LM_Memorization/blob/main/extraction.py
    """
    def __init__(self, 
                 metric: str = 'ppl', 
                 ref_model=None,
                 mask_model=None,
                 n_neighbor=5):
        # self.extraction_prompt = ["Tell me about..."]  # TODO this is just an example to extract data.
        self.metric = metric
        if self.metric not in function_map:
            raise ValueError(f"Metric {self.metric} is not supported. Please check if the function is implemented or if the name is correct.")
        self.ref_model = ref_model
        self.mask_model = mask_model
        self.n_neighbor = n_neighbor
    
    @torch.no_grad()
    def get_score(self, model: FinetunedCasualLM, dataset: Dataset):
        """
        Return score. Smaller value means membership.
        Function maps the method to the corresponding evaluation function.
        
        Args:
            model: The model to evaluate.
        
        """
        target = model
        reference = self.ref_model
        n_neighbor = 25
        k = 0.1
        
        if self.mask_model:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            mask_model = AutoModelForSeq2SeqLM.from_pretrained(self.mask_model,
                                                               torch_dtype=torch.bfloat16,
                                                               device_map="auto")
            mask_tokenizer = AutoTokenizer.from_pretrained(self.mask_model)
        
        # get locals
        locals_ = locals()
        # access the function from the function map, according to the metric.
        sig = inspect.signature(function_map[self.metric])
        required_args = sig.parameters
        extracted_args = {
            name: locals_[name] 
            for name in required_args if name in locals_
        }
        
        if self.metric == 'neighbor' or self.metric == 'spv_mia':
            score = dataset.map(lambda example: function_map[self.metric](text=example['text'], **extracted_args),
                                batched=True,
                                batch_size=64,
                                desc=f"Evaluating {self.metric}")
        else:
            score = dataset.map(lambda example: function_map[self.metric](text=example['text'], **extracted_args),
                                batched=False,
                                desc=f"Evaluating {self.metric}")
        return score
    
    def execute(self, 
                target: FinetunedCasualLM, 
                train_set, 
                test_set, 
                cache_file=None, 
                resume=False):
        """
        Excute the attack.
        """
        target.model.eval()
        if resume:
            if os.path.exists(cache_file):
                print(f"resume from {cache_file}")
                loaded = torch.load(cache_file)
                results = loaded['results']
                print(f"resume: i={loaded['i']}, member={loaded['member']}")
            else:
                print(f"WARN: Cann't resume. Not found {cache_file}.")
                resume = False
                results = defaultdict(list)
        else:
            results = defaultdict(list)
            
        if resume:
            if loaded['member'] != 1:
                print(f"Train set has been evaluated.")
                resume_i = len(train_set)
            else:
                resume_i = loaded['i']
                print(f"Resume from {resume_i+1}/{len(test_set)}")
        else:
            resume_i = -1
            
        # train set
        print("Evaluating train set:")
        results['score'] = self.get_score(target, train_set)['score']
        results['membership'] = [1] * len(train_set)
        print(f"Train avg score: {np.mean(np.array(results['score']))}")
        
        test_scores = []
             
        if resume and loaded['member'] == 0:
            resume_i = loaded['i']
            print(f"Resume from {resume_i+1}/{len(test_set)}")
        else:
            resume_i = -1
        
        # test set
        print("Evaluating test set:")
        test_scores = self.get_score(target, test_set)['score']
        results['score'] += test_scores
        results['membership'] += [0] * len(test_set)
        print(f"Test avg score: {np.mean(np.array(test_scores))}")
        # save the results
        if cache_file:
            torch.save({'results': results, 'i': -1, 'member': -1}, cache_file)
        return results

    def evaluate(self, args, results):
        """
        Evaluate the results.
        """
        score_dict = {}
        # Model info and Metric
        score_dict['block_size'] = args.block_size
        score_dict['metric'] = args.metric
        results['score'] = np.array(results['score'])
        results['membership'] = np.array(results['membership'])
        # score_dict['train(member)_score'] = np.mean(results['score'][results['membership']==1])
        # score_dict['test(nonmember)_score'] = np.mean(results['score'][results['membership']==0])
        
        # follow https://arxiv.org/pdf/2203.03929.pdf
        # threshold = np.quantile(results['score'][results['membership']==0], 0.9)
        # for computings AUC, you can use any threshold.
        # threshold = np.quantile(results['score'], 0.5)
        threshold = np.mean(results['score'][results['membership']==0])
        # score_dict['threshold'] = threshold
        results['score'] -= threshold
        
        # this is for the ease of using roc_auc_score, which is equivalent to varying threshold.
        # results['score'] = 1. - 1 / (1 + np.exp(- results['score']))
        # NOTE: score has to be reversed such that lower score implies membership.
        score_dict['acc'] = accuracy_score(results['membership'], results['score'] < 0)
        # score_dict['auc'] = roc_auc_score(results['membership'], - results['score'])
        fpr, tpr, thresholds = roc_curve(results['membership'], - results['score'])
        score_dict['auc'] = auc(fpr, tpr)
        
        # draw the ROC curve
        save_path = os.path.join(args.result_dir, args.model_name, args.dataset_name,)
        draw_auc_curve(fpr, tpr, 
                       title=str(save_path),
                       save_path=save_path,
                       metric=args.metric)
        
        # Get TPR@x%FPR
        for rate in [0.001, 0.005, 0.01, 0.05]:
            tpr_rate = tpr[np.where(fpr>=rate)[0][0]]
            actual_rate = fpr[np.where(fpr>=rate)[0][0]]
            score_dict[f'TPR@{rate*100}%FPR({actual_rate:.5f})'] = tpr_rate
            
        for rate in [0.99, 0.95]:
            fpr_rate = fpr[np.where(tpr<=rate)[0][-1]]
            actual_rate = tpr[np.where(tpr<=rate)[0][-1]]
            score_dict[f'FPR@{rate*100}%TPR({actual_rate:.5f})'] = fpr_rate
        # score_dict[r'TPR@0.1%FPR'] = tpr[np.where(fpr>=0.001)[0][0]]
        
        # save the results
        save_to_csv(score_dict, save_path)
        
        return score_dict
