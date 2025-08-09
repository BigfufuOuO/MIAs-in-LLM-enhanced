import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from collections import defaultdict
from datasets import Dataset

from .AttackBase import AttackBase
from models.finetuned_llms import FinetunedCasualLM
from .functions import AttackMethods
from .utils import draw_auc_curve, save_to_csv
import inspect
from finetune.utils import get_logger

class MemberInferenceAttack(AttackBase):
    """
    Membership Inference Attack (MIA) implementation.
    
    This class implements various membership inference attack methods to determine
    if a given text was part of a model's training data.
    
    MIA is often used with data extraction to find the training samples in 
    generated samples. For this purpose, top-score samples will be selected.

    Reference implementation: https://github.com/ftramer/LM_Memorization/blob/main/extraction.py
    """
    def __init__(self, 
                 logger,
                 metric: str = 'ppl', 
                 target_model: FinetunedCasualLM = None,
                 ref_model: FinetunedCasualLM = None,
                 mask_model=None,
                 args=None):
        """
        Initialize the MemberInferenceAttack.
        
        Args:
            logger: Logger instance for logging messages
            metric (str): The attack metric to use (default: 'ppl')
            target_model (FinetunedCasualLM): The target model to attack
            ref_model (FinetunedCasualLM): Reference model for comparative attacks
            mask_model: Model used for mask-based attacks
            args: Command line arguments containing attack parameters
        """
        self.metric = metric
        self.attack_methods = AttackMethods(args=args)
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
                  dataset: Dataset):
        """
        Calculate attack scores for the given dataset.
        
        Smaller score values indicate higher likelihood of membership.
        Maps the attack method to the corresponding evaluation function.
        
        Args:
            dataset (Dataset): The dataset to evaluate
            
        Returns:
            Dataset with scores added
        """
        target = self.target_model
        reference = self.ref_model
        mask_model = self.mask_model
        n_neighbor = self.n_neighbor
        n_perturbed = self.n_perturbed
        k = 0.1  # min_k parameter for min-k attacks

        # Get local variables for function argument extraction
        locals_ = locals()
        # Access the function from the function map according to the metric
        sig = inspect.signature(self.attack_methods.function_map[self.metric])
        required_args = sig.parameters
        extracted_args = {
            name: locals_[name] 
            for name in required_args if name in locals_ and name != 'self'
        }
        
        if self.metric == 'neighbor' or self.metric == 'spv_mia':
            if 'neighbor_texts' not in dataset.column_names:
                score = dataset.map(
                    lambda example: self.attack_methods.function_map[self.metric](
                        text=example['text'], **extracted_args
                    ),
                    batched=True,
                    batch_size=64,
                    desc=f"Evaluating {self.metric}"
                )
            else:
                score = dataset.map(
                    lambda example: self.attack_methods.function_map[self.metric](
                        text=example['text'],
                        neighbors=example['neighbor_texts'],
                        **extracted_args
                    ),
                    batched=True,
                    batch_size=32,
                    desc=f"Evaluating {self.metric}"
                )
        else:
            score = dataset.map(
                lambda example: self.attack_methods.function_map[self.metric](
                    text=example['text'], **extracted_args
                ),
                batched=False,
                desc=f"Evaluating {self.metric}"
            )
        return score
    
    def execute(self,
                train_set, 
                test_set, 
                args,
                cache_file=None, 
                resume=False):
        """
        Execute the membership inference attack on train and test datasets.
        
        Args:
            train_set: Training dataset
            test_set: Test dataset
            args: Command line arguments
            cache_file (str, optional): Path to cache file for resuming
            resume (bool): Whether to resume from cache
            
        Returns:
            dict: Results containing scores and membership labels
        """
        self.target_model.model.eval()
        if resume:
            if os.path.exists(cache_file):
                print(f"resume from {cache_file}")
                loaded = torch.load(cache_file)
                results = loaded['results']
                print(f"resume: i={loaded['i']}, member={loaded['member']}")
            else:
                print(f"WARNING: Cannot resume. File not found: {cache_file}.")
                resume = False
                results = defaultdict(list)
        else:
            results = defaultdict(list)
            
        # Handle train set evaluation resumption
        if resume:
            if loaded['member'] != 1:
                print("Train set has been evaluated.")
                train_resume_index = len(train_set)
            else:
                train_resume_index = loaded['i']
                print(f"Resume from {train_resume_index+1}/{len(test_set)}")
        else:
            train_resume_index = -1
        
        if args.use_neighbor_cache:
            if self.metric == 'neighbor' or self.metric == 'spv_mia':
                train_set, test_set = self.load_neighbor(args, self.target_model.tokenizer, train_set, test_set)
                self.neighbor_data_preview(train_set, test_set)
            else:
                print(f"Warning: Metric {self.metric} does not support neighbor dataset cache.")
            
        # train set
        self.logger.info("Evaluating train set:")
        train_result_dict = self.get_score(train_set)
        results['score'] = train_result_dict['score']
        results['membership'] = [1] * len(train_set)
        self.logger.info(f"Train avg score: {np.mean(np.array(results['score']))}")
        
        # Handle test set evaluation resumption
        if resume and loaded['member'] == 0:
            test_resume_index = loaded['i']
            print(f"Resume from {test_resume_index+1}/{len(test_set)}")
        else:
            test_resume_index = -1
        
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
                 extra_llms):
        """
        Evaluate the attack results and calculate metrics.
        
        Args:
            args: Command line arguments
            results: Attack results containing scores and membership labels
            extra_llms: Additional language models used in the attack
            
        Returns:
            dict: Score dictionary with evaluation metrics
        """
        score_dict = {}
        # Model info and Metric
        score_dict['block_size'] = args.block_size
        score_dict['metric'] = self.metric
        results['score'] = np.array(results['score'])
        results['membership'] = np.array(results['membership'])
        score_dict['train_score'] = np.mean(results['score'][results['membership']==1])
        score_dict['test_score'] = np.mean(results['score'][results['membership']==0])
        
        # Set threshold as mean of test scores
        threshold = np.mean(results['score'][results['membership']==0])
        results['score'] -= threshold
        
        # Calculate accuracy and AUC
        score_dict['acc'] = accuracy_score(results['membership'], results['score'] < 0)
        fpr, tpr, _ = roc_curve(results['membership'], - results['score'])
        score_dict['auc'] = auc(fpr, tpr)
        
        # draw the ROC curve
        save_path = os.path.join(args.result_dir, args.model_path, args.dataset_name)
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
                      test_neighbor: list):
        """
        Save neighbor data for later use with neighbor-based attack methods.
        
        Args:
            args: Command line arguments
            tokenizer: Tokenizer used for processing text
            train_neighbor (list): Neighbor data for training set
            test_neighbor (list): Neighbor data for test set
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
            n_perturbed: {self.n_perturbed}, n_neighbor: {self.n_neighbor}, mask_ratio: {args.mask_ratio}"
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
        Load previously saved neighbor data.
        
        Args:
            args: Command line arguments
            tokenizer: Tokenizer used for processing text
            train_dataset: Training dataset
            test_dataset: Test dataset
            
        Returns:
            tuple: Training and test datasets with neighbor data added
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
        Preview the neighbor data for debugging purposes.
        
        Args:
            train_set: Training dataset with neighbor data
            test_set: Test dataset with neighbor data
        """
        self.logger.info("===== Neighbor data preview ====")
        self.logger.info("Train set:")
        self.logger.info(train_set['neighbor_texts'][0][:5])
        self.logger.info("Test set:")
        self.logger.info(test_set['neighbor_texts'][0][:5])