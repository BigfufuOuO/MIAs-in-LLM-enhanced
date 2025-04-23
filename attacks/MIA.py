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
from .functions import AttackMethods
from .utils import draw_auc_curve, save_to_csv
import inspect
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from finetune.utils import get_logger

from data.factory import DataFactory

class MemberInferenceAttack(AttackBase):
    """
    Membership Inference Attack (MIA).

    Note MIA is often used with data extraction to find the training samples in 
    generated samples. For this purpose, top-score samples will be selected.

    Reference implementation: https://github.com/ftramer/LM_Memorization/blob/main/extraction.py
    """
    def __init__(self, 
                 logger,
                 metric: str = 'ppl', 
                 target_model: FinetunedCasualLM = None,
                 ref_model: FinetunedCasualLM = None,
                 mask_model=None,
                 args=None,):
        # self.extraction_prompt = ["Tell me about..."]  # TODO this is just an example to extract data.
        self.metric = metric
        self.attack_methods = AttackMethods(args=args,)
        if self.metric not in self.attack_methods.function_map:
            raise ValueError(f"Metric {self.metric} is not supported. Please check if the function is implemented or if the name is correct.")
        self.target_model = target_model
        self.ref_model = ref_model
        self.mask_model = mask_model
        self.n_neighbor = args.n_neighbor
        self.n_perturbed = args.n_perturbed
        self.mode = args.mode
        self.logger = logger
    
    @torch.no_grad()
    def get_score(self, 
                  dataset: Dataset, 
                  ):
        """
        Return score. Smaller value means membership.
        Function maps the method to the corresponding evaluation function.
        
        Args:
            model: The model to evaluate.
            dataset (Dataset): The dataset to evaluate.
        
        """
        target = self.target_model
        reference = self.ref_model
        mask_model = self.mask_model
        n_neighbor = self.n_neighbor
        n_perturbed = self.n_perturbed
        k = 0.1 # min_k

        # get locals
        locals_ = locals()
        # access the function from the function map, according to the metric.
        sig = inspect.signature(self.attack_methods.function_map[self.metric])
        required_args = sig.parameters
        extracted_args = {
            name: locals_[name] 
            for name in required_args if name in locals_ and name != 'self'
        }
        
        if self.metric == 'neighbor' or self.metric == 'spv_mia':
            if 'neighbor_texts' not in dataset.column_names:
                score = dataset.map(lambda example: self.attack_methods.function_map[self.metric](text=example['text'], **extracted_args),
                                    batched=True,
                                    batch_size=64,
                                    desc=f"Evaluating {self.metric}")
            else:
                score = dataset.map(lambda example: self.attack_methods.function_map[self.metric](text=example['text'],
                                                                                neighbors=example['neighbor_texts'],
                                                                                **extracted_args),
                                        batched=True,
                                        batch_size=16,
                                        desc=f"Evaluating {self.metric}")
        else:
            score = dataset.map(lambda example: self.attack_methods.function_map[self.metric](text=example['text'], **extracted_args),
                                batched=False,
                                desc=f"Evaluating {self.metric}")
        return score
    
    def execute(self,
                train_set, 
                test_set, 
                args,
                cache_file=None, 
                resume=False):
        """
        Excute the attack.
        """
        self.target_model.model.eval()
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
        
        if args.use_neighbor_cache:
            if self.metric == 'neighbor' or self.metric == 'spv_mia':
                train_set, test_set = self.load_neighbor(args, self.target_model.tokenizer,train_set, test_set)
                self.neighbor_data_preview(train_set, test_set)
            else:
                print(f"Warning:Metric {self.metric} does not support neighbor dataset cache.")
            
        # train set
        self.logger.info("Evaluating train set:")
        train_result_dict = self.get_score(train_set)
        results['score'] = train_result_dict['score']
        results['membership'] = [1] * len(train_set)
        self.logger.info(f"Train avg score: {np.mean(np.array(results['score']))}")
        
        test_scores = []
             
        if resume and loaded['member'] == 0:
            resume_i = loaded['i']
            print(f"Resume from {resume_i+1}/{len(test_set)}")
        else:
            resume_i = -1
        
        # test set
        self.logger.info("Evaluating test set:")
        test_result_dict = self.get_score(test_set)
        test_scores = test_result_dict['score']
        results['score'] += test_scores
        results['membership'] += [0] * len(test_set)
        self.logger.info(f"Test avg score: {np.mean(np.array(test_scores))}")
        # save the results
        if cache_file:
            torch.save({'results': results, 'i': -1, 'member': -1}, cache_file)
        
        # save neighbor data if not using caches  
        if not args.use_neighbor_cache \
            and 'neighbor_texts' in train_result_dict.column_names \
            and 'neighbor_texts' in test_result_dict.column_names:
            if self.target_model.tokenizer:
                self.save_neighbor(args,
                                   self.target_model.tokenizer,
                                train_result_dict['neighbor_texts'], 
                                test_result_dict['neighbor_texts'])
            else:
                raise ValueError("Tokenizer is required to save neighbor data.")
        return results

    def evaluate(self, 
                 args, 
                 results,
                 extra_llms,):
        """
        Evaluate the results.
        """
        score_dict = {}
        # Model info and Metric
        score_dict['block_size'] = args.block_size
        score_dict['metric'] = self.metric
        results['score'] = np.array(results['score'])
        results['membership'] = np.array(results['membership'])
        score_dict['train_score'] = np.mean(results['score'][results['membership']==1])
        score_dict['test_score'] = np.mean(results['score'][results['membership']==0])
        
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
        save_path = os.path.join(args.result_dir, args.model_path, args.dataset_name,)
        draw_auc_curve(fpr, tpr, 
                       title=str(save_path),
                       save_path=save_path,
                       metric=self.metric)
        
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
        
        # additional report
        if args.mode == "neighbor":
            score_dict['n_perturbed'] = self.n_perturbed
            score_dict['n_neighbor'] = self.n_neighbor
            score_dict['mask_ratio'] = args.mask_ratio
            score_dict['refer_model'] = None
            score_dict['mask_model'] = None
            if extra_llms[0]:
                score_dict['refer_model'] = extra_llms[0].model.__class__.__name__
            if extra_llms[1]:
                score_dict['mask_model'] = extra_llms[1].model.__class__.__name__      
        elif args.mode == "ft-phase":
            score_dict['train_loss'] = self.target_model.train_loss.get("loss", None)
            score_dict['eval_loss'] = self.target_model.eval_loss.get("eval_loss", None)
            score_dict['epoch'] = self.target_model.train_loss.get("epoch", None)
        elif args.mode == "defense":
            score_dict['defense_method'] = args.defense
        
        # save the results
        save_to_csv(score_dict, args.mode, save_path)
        
        return score_dict
    
    def save_neighbor(self, 
                      args,
                      tokenizer,
                      train_neighbor: list, 
                      test_neighbor: list,):
        """
        If neighbour method (Neighbor, SPV_MIA) is used, save the neighbour for later use.
        """
        tokenizer_name = tokenizer.__class__.__name__
        save_path = f'./data/neighbor_data/{tokenizer_name}/{args.dataset_name}/bs{args.block_size}/{self.metric}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if self.metric == 'neighbor':
            meta_data = f'train_neighbor_n-nei{self.n_neighbor}'
        elif self.metric == 'spv_mia':
            meta_data = f'train_neighbor_n-per{self.n_perturbed}_n-mskr{args.mask_ratio}'
        else:
            raise ValueError(f"Metric {self.metric} is not supported.")
        
        train_save_path = os.path.join(save_path, meta_data)
        train_neighbor = {'text': train_neighbor}
        train_neighbor = Dataset.from_dict(train_neighbor)
        train_neighbor.info.description = f"Neighbor data for {args.dataset_name} train set. \
            Metric: {self.metric}, Block Size: {args.block_size}\
            n_perterbed: {self.n_neighbor}, n_neighbor: {self.n_perturbed}, mask_ratio: {args.mask_ratio}"
        train_neighbor.save_to_disk(train_save_path)
        
        test_save_path = os.path.join(save_path, meta_data.replace('train', 'test'))
        test_neighbor = {'text': test_neighbor}
        test_neighbor = Dataset.from_dict(test_neighbor)
        test_neighbor.save_to_disk(test_save_path)
        
        self.logger.info(f"Neighbor data saved to {save_path}")
        
    def load_neighbor(self,
                      args,
                      tokenizer,
                      train_dataset,
                      test_dataset):
        """
        Load the neighbor data.
        """
        tokenizer_name = tokenizer.__class__.__name__
        save_path = f'./data/neighbor_data/{tokenizer_name}/{args.dataset_name}/bs{args.block_size}/{self.metric}'
        self.logger.info(f"Loading neighbor data from {save_path}")
        
        if self.metric == 'neighbor':
            meta_data = f'train_neighbor_n-nei{self.n_neighbor}'
        elif self.metric == 'spv_mia':
            meta_data = f'train_neighbor_n-per{self.n_perturbed}_n-mskr{args.mask_ratio}'
        else:
            raise ValueError(f"Metric {self.metric} is not supported.")
        train_neighbor = Dataset.load_from_disk(os.path.join(save_path, meta_data))
        test_neighbor = Dataset.load_from_disk(os.path.join(save_path, meta_data.replace('train', 'test')))
        
        train_dataset = Dataset.from_dict({
            'text': train_dataset['text'],
            'neighbor_texts': train_neighbor['text']
        })
        test_dataset = Dataset.from_dict({
            'text': test_dataset['text'],
            'neighbor_texts': test_neighbor['text']
        })
        return train_dataset, test_dataset

    def neighbor_data_preview(self, train_set, test_set):
        """
        Preview the neighbor data.
        """
        self.logger.info("===== Neighbor data preview ====")
        self.logger.info("Train set:")
        self.logger.info(train_set['neighbor_texts'][0][:5])
        self.logger.info("Test set:")
        self.logger.info(test_set['neighbor_texts'][0][:5])