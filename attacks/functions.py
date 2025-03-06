"""
This file contains all sorts of methods that are used in the attack.
"""
from models.finetuned_llms import FinetunedCasualLM
import numpy as np
import zlib
import torch
import torch.nn.functional as F
from .utils import *
from .spv import SpvMIAGenerator

def empty(text: str):
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

def loss(target: FinetunedCasualLM, text: str):
    """
    Return loss of the given text.
    
    Args:
        target: The target to evaluate.
        text: The text to evaluate.
    """
    return {
        "score": target.evaluate(text)
    }

def perplexity(target: FinetunedCasualLM, text: str):
    """
    Return perplexity of the given text.
    
    Args:
        target: The target to evaluate.
        text: The text to evaluate.
    """
    return {
        "score": target.evaluate_ppl(text)
    }

def Refer(target: FinetunedCasualLM, 
          reference: FinetunedCasualLM, 
          text: str):
    """
    Simple REFER method.
    https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
    """
    ppl = target.evaluate_ppl(text)
    ref_ppl = reference.evaluate_ppl(text)
    return {
        "score": np.log(ppl) / np.log(ref_ppl)
    }

def Zlib(target: FinetunedCasualLM, text: str):
    """
    ZLIB method.
    https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
    """
    ppl = target.evaluate_ppl(text)
    num_bits = len(zlib.compress(bytes(text, 'utf-8')))
    return {
        "score": ppl / num_bits
    }

def Lowercase(target: FinetunedCasualLM, text: str):
    """
    LOWERCASE method.
    https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
    """
    ppl = target.evaluate_ppl(text)
    ref_ppl = target.evaluate_ppl(text.lower())
    return {
        "score": ppl / ref_ppl
    }

def Window(target: FinetunedCasualLM, text: str):
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
            _ppl = target.evaluate_ppl(input_ids[idx, idx+win_size], tokenized=True)
            ppls.append(_ppl.item())
        return {
            "score": np.min(ppls)
        }
    else:
        return {
            "score": target.evaluate_ppl(input_ids, tokenized=True)
        }
    
def LiRASimple(target: FinetunedCasualLM,
               reference: FinetunedCasualLM, 
               text: str):
    """
    Simple LIRA method (Without energy).
    https://arxiv.org/abs/2203.03929
    """
    ppl = target.evaluate_ppl(text)
    ref_ppl = reference.evaluate_ppl(text)
    return {
        "score": np.log(ppl) - np.log(ref_ppl)
    }

def Neighbour(target: FinetunedCasualLM, 
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
    
def Neighbour_inbatch(target: FinetunedCasualLM,
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
        loss_text = target.evaluate(batch_text)
        loss_neigh = target.evaluate(batch_neighbor, padding=True)
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

def Min_k(target: FinetunedCasualLM, 
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

def Min_k_plus(target: FinetunedCasualLM, 
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

def SPV_MIA(target: FinetunedCasualLM, 
            reference: FinetunedCasualLM,
            text: list, 
            mask_model: any = None,
            mask_tokenizer: any = None,
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
    
    spv_mia = SpvMIAGenerator(mask_model=mask_model,
                      mask_tokenizer=mask_tokenizer)
    if neighbors:
        perturbed_texts = neighbors
    else:
        n_failed, perturbed_texts = spv_mia.tokenize_masks(text,
                                                        n_perturbed=n_perturbed)
    scores = []
    for batch_text, batch_perturbed_text in zip(text, perturbed_texts):
        original_target_loss = target.evaluate(batch_text)
        original_ref_loss = reference.evaluate(batch_text)
        perturbed_target_loss = target.evaluate(batch_perturbed_text, padding=True)
        perturbed_ref_loss = reference.evaluate(batch_perturbed_text, padding=True)
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
        


function_map = {
    "empty": empty,
    "loss": loss,
    "ppl": perplexity,
    "refer": Refer,
    "refer-base": Refer,
    "refer-orcale": Refer,
    "zlib": Zlib,
    "lowercase": Lowercase,
    "window": Window,
    "lira": LiRASimple,
    "lira-base": LiRASimple,
    "lira-orcale": LiRASimple,
    "neighbor": Neighbour_inbatch,
    "min_k": Min_k,
    "min_k++": Min_k_plus,
    "spv_mia": SPV_MIA
}