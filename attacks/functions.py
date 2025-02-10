"""
This file contains all sorts of methods that are used in the attack.
"""
from models.finetuned_llms import FinetunedCasualLM
import numpy as np
import zlib
import torch
import torch.nn.functional as F

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
    return target.evaluate_ppl(text)

def Refer(target: FinetunedCasualLM, 
          reference: FinetunedCasualLM, 
          text: str):
    """
    Simple REFER method.
    https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
    """
    ppl = target.evaluate_ppl(text)
    ref_ppl = reference.evaluate_ppl(text)
    return np.log(ppl) / np.log(ref_ppl)

def Zlib(target: FinetunedCasualLM, text: str):
    """
    ZLIB method.
    https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
    """
    ppl = target.evaluate_ppl(text)
    num_bits = len(zlib.compress(bytes(text, 'utf-8')))
    return ppl / num_bits

def Lowercase(target: FinetunedCasualLM, text: str):
    """
    LOWERCASE method.
    https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
    """
    ppl = target.evaluate_ppl(text)
    ref_ppl = target.evaluate_ppl(text.lower())
    return ppl / ref_ppl

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
        return np.min(ppls)
    else:
        return target.evaluate_ppl(input_ids, tokenized=True)
    
def LiRASimple(target: FinetunedCasualLM,
               reference: FinetunedCasualLM, 
               text: str):
    """
    Simple LIRA method (Without energy).
    https://arxiv.org/abs/2203.03929
    """
    ppl = target.evaluate_ppl(text)
    ref_ppl = reference.evaluate_ppl(text)
    return np.log(ppl) - np.log(ref_ppl)

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
    return - np.mean(topk).item()

def Min_k_plus(target: FinetunedCasualLM, 
               text: str, 
               k: float = 0.1):
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
    return - np.mean(topk).item()

def SVA_MIA(target: FinetunedCasualLM, 
            text: str, ):
    """
    SVA-MIA method.
    https://arxiv.org/pdf/2404.02936
    
    Args:
        target: The target to evaluate.
        text: The text to evaluate.
        k: The proportion of the tokens to consider.
    """
    pass
        


function_map = {
    "loss": loss,
    "ppl": perplexity,
    "refer": Refer,
    "zlib": Zlib,
    "lowercase": Lowercase,
    "window": Window,
    "lira": LiRASimple,
    "neighbor": Neighbour,
    "min_k": Min_k,
    "min_k++": Min_k_plus,
    "sva_mia": SVA_MIA
}