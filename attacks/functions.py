"""
This file contains all sorts of methods that are used in the attack.
"""
from models.finetuned_llms import FinetunedCasualLM
from models.mask_llms import MaskLanguageModel
import numpy as np
import zlib
import torch
import torch.nn.functional as F
from .utils import *
from .spv import SpvMIAGenerator


class AttackMethods:
    """
    This class contains all the attack methods.
    """
    def __init__(self, args=None,):
        self.args = args
        self.function_map = {
            "empty": self.empty,
            "loss": self.loss,
            "ppl": self.perplexity,
            "refer": self.Refer,
            "refer-base": self.Refer,
            "refer-orcale": self.Refer,
            "zlib": self.Zlib,
            "lowercase": self.Lowercase,
            "window": self.Window,
            "lira": self.LiRASimple,
            "lira-base": self.LiRASimple,
            "lira-orcale": self.LiRASimple,
            "neighbor": self.Neighbour_inbatch,
            "min_k": self.Min_k,
            "min_k++": self.Min_k_plus,
            "spv_mia": self.SPV_MIA
        }
        
        self.dp_linear = True if args.mode == 'defense' and args.defense == 'dp_linear' else False
        
    def empty(self, text: str):
        """
        No attack method.
        
        Args:
            target: The target to evaluate.
            text: The text to evaluate.
        """
        # random score in {0, 1}
        return {
            "score": float(np.random.randint(0, 2))
        }

    def loss(self, target: FinetunedCasualLM, text: str):
        """
        Return loss of the given text.
        
        Args:
            target: The target to evaluate.
            text: The text to evaluate.
        """
        if self.dp_linear:
            score = target.evaluate_with_dp(text, lambda_param=0.8)
            return {
                "score": score
            }
        return {
            "score": target.evaluate(text)
        }

    def perplexity(self, target: FinetunedCasualLM, text: str):
        """
        Return perplexity of the given text.
        
        Args:
            target: The target to evaluate.
            text: The text to evaluate.
        """
        return {
            "score": target.evaluate_ppl(text, dp=self.dp_linear)
        }

    def Refer(self, target: FinetunedCasualLM, 
            reference: FinetunedCasualLM, 
            text: str):
        """
        Simple REFER method.
        https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
        """
        ppl = target.evaluate_ppl(text, dp=self.dp_linear)
        ref_ppl = reference.evaluate_ppl(text, dp=self.dp_linear)
        return {
            "score": np.log(ppl) / np.log(ref_ppl)
        }

    def Zlib(self, target: FinetunedCasualLM, text: str):
        """
        ZLIB method.
        https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
        """
        ppl = target.evaluate_ppl(text, dp=self.dp_linear)
        num_bits = len(zlib.compress(bytes(text, 'utf-8')))
        return {
            "score": ppl / num_bits
        }

    def Lowercase(self, target: FinetunedCasualLM, text: str):
        """
        LOWERCASE method.
        https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
        """
        ppl = target.evaluate_ppl(text, dp=self.dp_linear)
        ref_ppl = target.evaluate_ppl(text.lower(), dp=self.dp_linear)
        return {
            "score": ppl / ref_ppl
        }

    def Window(self, target: FinetunedCasualLM, text: str):
        """
        WINDOW method.
        https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
        """
        assert target.tokenizer is not None
        input_ids = target.tokenizer(text, return_tensors='pt', 
                                    truncation=True, 
                                    max_length=target.max_seq_len).input_ids
        win_size = 50
        if len(input_ids) > win_size:
            ppls = []
            for idx in range(len(input_ids)- win_size):
                _ppl = target.evaluate_ppl(input_ids[idx, idx+win_size], tokenized=True, dp=self.dp_linear)
                ppls.append(_ppl.item())
            return {
                "score": np.min(ppls)
            }
        else:
            return {
                "score": target.evaluate_ppl(input_ids, tokenized=True, dp=self.dp_linear)
            }
        
    def LiRASimple(self, target: FinetunedCasualLM,
                reference: FinetunedCasualLM, 
                text: str):
        """
        Simple LIRA method (Without energy).
        https://arxiv.org/abs/2203.03929
        """
        ppl = target.evaluate_ppl(text, dp=self.dp_linear)
        ref_ppl = reference.evaluate_ppl(text, dp=self.dp_linear)
        return {
            "score": np.log(ppl) - np.log(ref_ppl)
        }

    def Neighbour(self, target: FinetunedCasualLM, 
                reference: FinetunedCasualLM, 
                text: str, 
                n_neighbor: int = 5):
        """
        NEIGHBOR method.
        https://arxiv.org/abs/2305.18462
        
        Args:
            target: The target to evaluate.
            reference: The reference target to evaluate.
            text: The text to evaluate.
            n_neighbor: The number of neighbors to generate.
        """
        assert reference is not None, 'Neighborhood MIA requires a reference target'
        neighbor_avg = 0
        neighbors = reference.generate_neighbors(text, n=n_neighbor)
        
        loss_neigh = [target.evaluate(neighbor) for neighbor in neighbors]
        return {
            "score": target.evaluate(text) - np.mean(loss_neigh)
        }
        
    def Neighbour_inbatch(self, target: FinetunedCasualLM,
                        reference: FinetunedCasualLM,
                        n_neighbor: int = 5,
                        text: list = None,
                        neighbors: list = None):
        """
        NEIGHBOR method in batch.
        https://arxiv.org/abs/2305.18462
        
        Args:
            target: The target to evaluate.
            reference: The reference target to evaluate.
            text: The text to evaluate.
            n_neighbor: The number of neighbors to generate.
        """
        assert reference is not None, 'Neighborhood MIA requires a reference target'
        if neighbors:
            batch_neighbors = neighbors
        else:
            batch_neighbors = reference.generate_neighbors_inbatch(text, n=n_neighbor)
        
        scores = []
        for batch_neighbor, batch_text in zip(batch_neighbors, text):
            if not self.dp_linear:
                loss_text = target.evaluate(batch_text)
                loss_neigh = target.evaluate_batch(batch_neighbor)
            else:
                loss_text = target.evaluate_with_dp(batch_text).item()
                loss_neigh = target.evaluate_batch_with_dp(batch_neighbor).item()
            batch_score = loss_text - loss_neigh
            scores.append(batch_score)
        
        if neighbors:
            return {
                "score": scores
            }
        else:
            return {
                "score": scores,
                "neighbor_texts": batch_neighbors
            }

    def Min_k(self, target: FinetunedCasualLM, 
            text: str, 
            k: float = 0.1):
        """
        Min K method.
        https://arxiv.org/pdf/2310.16789
        
        Args:
            target: The target to evaluate.
            text: The text to evaluate.
            k: The proportion of the tokens to consider.
        """
        input_ids = target.tokenizer(text, return_tensors='pt', 
                                    truncation=True, 
                                    max_length=target.max_seq_len).input_ids
        input_ids = input_ids.to(target.model.device)
        with torch.no_grad():
            outputs = target.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        
        input_ids = input_ids[0][1:].unsqueeze(-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        
        k_length = int(len(token_log_probs) * k)
        token_log_probs = token_log_probs.to(torch.float32)
        topk = np.sort(token_log_probs.cpu())[:k_length]
        # TODO: Check if this is correct
        return {
            "score": - np.mean(topk).item()
        }

    def Min_k_plus(self, target: FinetunedCasualLM, 
                text: str, 
                k: float = 0.2):
        """
        Min K++ method.
        https://arxiv.org/pdf/2404.02936
        
        Args:
            target: The target to evaluate.
            text: The text to evaluate.
            k: The proportion of the tokens to consider.
        """
        input_ids = target.tokenizer(text, return_tensors='pt', 
                                    truncation=True, 
                                    max_length=target.max_seq_len).input_ids
        input_ids = input_ids.to(target.model.device)
        with torch.no_grad():
            outputs = target.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0,:-1], dim=-1)
        log_probs = F.log_softmax(logits[0,:-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        # min-k ++
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        
        mink_plus = (token_log_probs - mu) / (sigma.sqrt() + 1e-8)
        mink_plus = mink_plus.to(torch.float32)
        k_length = int(len(token_log_probs) * k)
        topk = np.sort(mink_plus.cpu())[:k_length]
        # TODO: Check if this is correct
        return {
            "score": - np.mean(topk).item()
        }

    def SPV_MIA(self, target: FinetunedCasualLM, 
                reference: FinetunedCasualLM,
                text: list, 
                mask_model: MaskLanguageModel = None,
                n_perturbed: int = 5,
                neighbors: list = None
                ):
        """
        SVA-MIA method.
        https://arxiv.org/pdf/2311.06062
        https://github.com/tsinghua-fib-lab/ANeurIPS2024_SPV-MIA/tree/main
        
        Args:
            target: The target to evaluate.
            text: The text to evaluate.
            k: The proportion of the tokens to consider.
        """
        # assert reference is not None, 'SVA MIA requires a reference model'
        if neighbors:
            perturbed_texts = neighbors
        else:
            perturbed_texts = mask_model.generate_perturbed_texts(
                texts=text,
                n_perturbed=n_perturbed,
            )
        scores = []
        for batch_text, batch_perturbed_text in zip(text, perturbed_texts):
            if not self.dp_linear:
                original_target_loss = target.evaluate(batch_text)
                original_ref_loss = reference.evaluate(batch_text)
                perturbed_target_loss = target.evaluate_batch(batch_perturbed_text)
                perturbed_ref_loss = reference.evaluate_batch(batch_perturbed_text)
            else:
                original_target_loss = target.evaluate_with_dp(batch_text).item()
                original_ref_loss = reference.evaluate_with_dp(batch_text).item()
                perturbed_target_loss = target.evaluate_batch_with_dp(batch_perturbed_text).item()
                perturbed_ref_loss = reference.evaluate_batch_with_dp(batch_perturbed_text).item()
            score = (original_target_loss - original_ref_loss) - (perturbed_target_loss - perturbed_ref_loss)
            scores.append(score)
        if neighbors:
            return {
                "score": scores
            }
        else:
            return {
                "score": scores,
                "neighbor_texts": perturbed_texts,
            }
        


