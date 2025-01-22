import torch
import os
import zlib
from .AttackBase import AttackBase
from enum import Enum
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from collections import defaultdict

from models.finetuned_llms import FinetunedCasualLM
from .functions import function_map

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
                 n_neighbor=50):
        # self.extraction_prompt = ["Tell me about..."]  # TODO this is just an example to extract data.
        self.metric = metric
        self.ref_model = ref_model
        self.n_neighbor = n_neighbor
    
    @torch.no_grad()
    def _get_score(self, model: FinetunedCasualLM, text: str):
        """
        Return score. Smaller value means membership.
        Function maps the method to the corresponding evaluation function.
        
        Args:
            model: The model to evaluate.
        
        """
        score = function_map[self.metric](model, text)
        return score
    
    def execute(self, model, train_set, test_set, cache_file=None, resume=False):
        """Compute scores for texts.

        Parameters:
        - memberships: A list of 0,1. 0 means non-member and 1 means member.

        Returns:
        - scores: the scores of membership.
        """
        model._lm.eval()
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
        member = 1
        for i, sample in enumerate(tqdm(train_set, desc="Train set")):
            if i <= resume_i:
                continue
            score = self._get_score(model, sample['text'])
            results['score'].append(score)
            results['membership'].append(member)
            if (i+1) % 100 == 0:
                torch.save({'results': results, 'i': i, 'member': member}, cache_file)
        print(f"Train avg score: {np.mean(np.array(results['score']))}")
        
        test_scores = []
        member = 0
        
        if resume and loaded['member'] == 0:
            resume_i = loaded['i']
            print(f"Resume from {resume_i+1}/{len(test_set)}")
        else:
            resume_i = -1
        for i, sample in enumerate(tqdm(test_set)):
            if i <= resume_i:
                continue
            score = self._get_score(model, sample['text'])
            results['score'].append(score)
            test_scores.append(score)
            results['membership'].append(0)
            if (i+1) % 30 == 0:
                torch.save({'results': results, 'i': i, 'member': member}, cache_file)
        print(f"Test avg score: {np.mean(np.array(test_scores))}")
        torch.save({'results': results, 'i': -1, 'member': -1}, cache_file)
        return results

    def evaluate(self, results):
        # results['score']
        score_dict = {}
        results['score'] = np.array(results['score'])
        results['membership'] = np.array(results['membership'])
        # # follow https://arxiv.org/pdf/2203.03929.pdf
        # threshold = np.quantile(results['score'][results['membership']==0], 0.9)
        threshold = np.mean(results['score'][results['membership']==0])
        score_dict['nonmember_score'] = np.mean(results['score'][results['membership']==0])
        score_dict['member_score'] = np.mean(results['score'][results['membership']==1])
        # for computing AUC, you can use any threshold.
        # threshold = np.quantile(results['score'], 0.5)
        results['score'] -= threshold
        # this is for the ease of using roc_auc_score, which is equivalent to varying threshold.
        # results['score'] = 1. - 1 / (1 + np.exp(- results['score']))
        # NOTE score has to be reversed such that lower score implies membership.
        score_dict['acc'] = accuracy_score(results['membership'], results['score'] < 0)
        score_dict['auc'] = roc_auc_score(results['membership'], - results['score'])
        fpr, tpr, thresholds = roc_curve(results['membership'], - results['score'])
        score_dict[r'TPR@0.1%FPR'] = None
        for fpr_, tpr_, thr_ in zip(fpr, tpr, thresholds):
            if fpr_ > 0.001:
                score_dict[r'TPR@0.1%FPR'] = tpr_
                break
        return score_dict
