import transformers
import torch
import numpy as np
from heapq import nlargest
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig,
    AutoModel
)
import time
from safetensors.torch import load_model
import os

from .LLMBase import LLMBase
from finetune.utils import get_logger

logger = get_logger("MIA", level="info")

class SamplingArgs:
    def __init__(self,
                 prefix_length=50,
                 suffix_length=50,
                 do_sample=True,
                 top_k=24,
                 top_p=0.8,
                 typical_p=0.9,
                 temperature=0.58,
                 repetition_penalty=1.04,
                 zlib=False,
                 context_window=4,
                 high_conf=True):
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.zlib = zlib
        self.context_window = context_window
        self.high_conf = high_conf


class FinetunedCasualLM(LLMBase):
    """
    Huggingface Casual Language Models.

    Args:
        `model_path (str)`:
            The path/name for the desired langauge model.

            Supported models:
                1. llama2-7b, llama2-7b-chat: Find the model path at https://huggingface.co/LLM-PBE.
                2. gpt2, gpt2-large, gpt2-xl: The names for models on huggingface. Should manually download.
                3. Local path pointed to GPT2 model finetuned based on https://github.com/microsoft/analysing_pii_leakage.
                   Analyzing Leakage of Personally Identifiable Information in Language Models. Nils Lukas, Ahmed Salem,
                   Robert Sim, Shruti Tople, Lukas Wutschitz and Santiago Zanella-Béguelin. Symposium on Security and
                   Privacy (S&P '23). San Francisco, CA, USA.

    Returns:
        None
    """

    def __init__(self, args=None, 
                 arch=None, 
                 model_path='openai-community/gpt2', 
                 max_seq_len=1024,
                 mask=False,
                 **kwargs):
        
        if ':' in model_path:
            model_path, self.model_revision = model_path.split(':')
        else:
            self.model_revision = 'main'

        if not args.load_bin or mask:
            self.arch = model_path
        else:
            self.arch = args.model_path

        # arguments
        self.args = args
        # default
        self.tokenizer_use_fast = True
        self.max_seq_len = max_seq_len
        self.verbose = True
        
        if kwargs.get("train_loss", None) and kwargs.get("eval_loss", None):
            logger.info(f"Training loss: {kwargs.get('train_loss')}, Evaluation loss: {kwargs.get('eval_loss')}")
            self.train_loss = kwargs.get("train_loss")
            self.eval_loss = kwargs.get("eval_loss")

        super().__init__(model_path=model_path)

    @property
    def tokenizer(self):
        return self._tokenizer

    def load_local_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path

        # int8 or half precision
        int8_kwargs = {}
        half_kwargs = {}
        logger.info(f"Loading model in int8: {self.args.int8} or half: {self.args.half}")
        if self.args.int4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        if self.args.int8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif self.args.half:
            bnb_config = None
            
        logger.info(f"Loading tokenizer from {self.arch}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.arch,
                                                        use_fast=self.tokenizer_use_fast)
        if self.verbose:
            logger.info(
                f"> Loading the provided {self.arch} checkpoint from '{model_path}'.")

        if self.args.load_bin:
            self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                              return_dict=True,
                                                                device_map='auto',
                                                                revision=self.model_revision,
                                                                torch_dtype=torch.bfloat16,
                                                                token=self.args.token,
                                                                quantization_config=bnb_config,)
        else: 
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                            return_dict=True,
                                                            device_map='auto',
                                                            revision=self.model_revision,
                                                            torch_dtype=torch.bfloat16,
                                                            token=self.args.token,
                                                            quantization_config=bnb_config,)
            except:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                            return_dict=True, 
                                                            device_map='auto',
                                                            revision=self.model_revision, 
                                                            offload_folder='./offload',
                                                            low_cpu_mem_usage=True,
                                                            torch_dtype=torch.bfloat16,
                                                            token=self.args.token,
                                                            quantization_config=bnb_config,)
            
            
        self.model.eval()

        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def query(self, text, new_str_only=False):
        """
        Query an open-source model with a given text prompt.

        Args:
            text (str): The text prompt to query the model.

        Returns:
            str: The model's output.
        """
        # TODO pass the args into here. The params should be set according to PII-leakage.

        # Encode the text prompt and generate a response
        input_ids = self._tokenizer(text, return_tensors='pt').input_ids
        
        attention_mask = torch.ones_like(input_ids)

        # Implement the code to query the open-source model
        output = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            max_new_tokens=self.max_seq_len,
            do_sample=True,
            return_dict_in_generate=True,
        )

        # Decode the generated text back to a readable string
        if new_str_only:
            generated_text = self._tokenizer.decode(output.sequences[0][len(input_ids[0]):], skip_special_tokens=True)
        else:
            generated_text = self._tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return generated_text

    def evaluate(self, 
                 text, 
                 tokenized=False,
                 padding: bool = False):
        """
        Evaluate an open-source model with a given text prompt.

        Args:
            text (str): The text prompt to query the model.

        Returns:
            loss: The model's loss.
        """
        # TODO pass the args into here. The params should be set according to PII-leakage.
        if tokenized:
            input_ids = text
        else:
            # Encode the text prompt and generate a response
            input_ids = self._tokenizer(text, 
                                        return_tensors='pt', 
                                        truncation=True,
                                        padding=padding,
                                        max_length=self.max_seq_len).input_ids
            
        if padding:
            # attention if not padding token
            attention_mask = torch.where(input_ids == self._tokenizer.pad_token_id, 0, 1)
        else:
            attention_mask = torch.ones_like(input_ids)

        # Implement the code to query the open-source model
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids.clone(),
                output_hidden_states=False,
                output_attentions=False,
            )
        return output.loss.item()
    
    def evaluate_batch(self, 
                       texts: list) -> torch.Tensor:
        """
        Evaluate an open-source model with a batch of text prompts.

        Args:
            text (list): The batched text prompts to query the model.

        Returns:
            loss: The model's average loss.
        """
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        input_ids = self._tokenizer(texts, 
                                    return_tensors='pt', 
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len).input_ids
        attention_mask = torch.where(input_ids == self._tokenizer.pad_token_id, 0, 1)
        
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids.clone(),
            )
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        batch_size, seq_len = shift_labels.shape
        padding_mask = (shift_labels != self._tokenizer.pad_token_id).float()
        loss_per_sentence = loss.view(batch_size, seq_len)
        loss_per_sentence = (loss_per_sentence * padding_mask).sum(dim=1) / padding_mask.sum(dim=1)
        return loss_per_sentence.mean().item()
    
    def evaluate_with_dp(self, 
                         text,
                         lambda_param: int = 0.5,
                         padding: bool = False,
                         tokenized: bool = False) -> torch.Tensor:
        """
        Evaluate an open-source model with a given text prompt. 
        Based on the DP method mentioned by:
        https://arxiv.org/pdf/2205.13621
        
        """
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        if tokenized:
            input_ids = text
        else:
            input_ids = self._tokenizer(text, 
                                        return_tensors='pt', 
                                        padding=padding,
                                        truncation=True,
                                        max_length=self.max_seq_len).input_ids
        attention_mask = torch.where(input_ids == self._tokenizer.pad_token_id, 0, 1)
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids.clone(),
            )
        
        # DP
        vocab_size = output.logits.shape[-1]
        uniform_dist = torch.ones(vocab_size).to(self.model.device) / vocab_size
        original_probs = torch.softmax(output.logits, dim=-1)
        perturbed_probs = lambda_param * original_probs + (1 - lambda_param) * uniform_dist
        perturbed_logits = torch.log(perturbed_probs)
        
        # loss
        shift_logits = perturbed_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        batch_size, seq_len = shift_labels.shape
        padding_mask = (shift_labels != self._tokenizer.pad_token_id).float()
        loss_per_sentence = loss.view(batch_size, seq_len)
        loss_per_sentence = (loss_per_sentence * padding_mask).sum(dim=1) / padding_mask.sum(dim=1)
        return loss_per_sentence
    
    def evaluate_batch_with_dp(self, 
                               texts: list, 
                               lambda_param: int = 0.5,
                               padding: bool = True) -> torch.Tensor:
        loss_per_sentence = self.evaluate_with_dp(texts, lambda_param=lambda_param, padding=padding)
        return loss_per_sentence.mean()
            
    def evaluate_ppl(self, text, tokenized=False, dp=False, lambda_param=0.5):
        """
        Evaluate an open-source model with a given text prompt.

        Args:
            text (str): The text prompt to query the model.

        Returns:
            PPL: The model's perpelexity.
        """
        if dp:
            loss = self.evaluate_with_dp(text, lambda_param=lambda_param, tokenized=tokenized).item()
        else:
            loss = self.evaluate(text, tokenized=tokenized)
        return np.exp(loss)

    def generate_neighbors(self, text, p=0.7, k=3, n=5):
        """
        For TEXT, generates a neighborhood of single-token replacements, considering the best K token replacements
        at each position in the sequence and returning the top N neighboring sequences.

        https://aclanthology.org/2023.findings-acl.719.pdf
        
        Args:
            text (str): The input text to generate the neighborhood.
            p (float): The dropout probability.
            k (int): The number of top candidates to consider at each position.
            n (int): The number of neighboring sequences to return.
        """

        tokenized = self._tokenizer(text, 
                                    return_tensors='pt', 
                                    truncation=True,
                                    max_length=self.max_seq_len).input_ids.to(self.model.device)
        dropout = torch.nn.Dropout(p)


        # Embed the sequence
        if isinstance(self.model, transformers.LlamaForCausalLM):
            embedding = self.model.get_input_embeddings()(tokenized)
        elif isinstance(self.model, transformers.GPT2LMHeadModel):
            embedding = self.model.transformer.wte.weight[tokenized]
        elif isinstance(self.model, transformers.RobertaForCausalLM):
            embedding = self.model.roberta.embeddings(tokenized)
        else:
            raise RuntimeError(f'Unsupported model type for neighborhood generation: {type(self.model)}')
        
        # Apply dropout all in once 
        dropout_embedding = dropout(embedding)

        seq_len = tokenized.shape[1]
        cand_scores = {}
        for target_index in range(1, seq_len):
            target_token = tokenized[0, target_index]

            # Apply dropout only to the target token embedding in the sequence
            modified_embedding = torch.cat([
                embedding[:, :target_index],
                dropout_embedding[:, target_index].unsqueeze(0), # apply dropout to the target token, unsqueeze to match the shape
                embedding[:, target_index+1:]
            ], dim=1)

            # Get model's predicted posterior distributions over all positions in the sequence
            with torch.no_grad():
                logits = self.model(inputs_embeds=modified_embedding).logits
                probs = torch.softmax(logits, dim=2)
            original_prob = probs[0, target_index, target_token].item()

            # Find the K most probable token replacements, not including the target token
            # Find top K+1 first because target could still appear as a candidate
            cand_probs, cands = torch.topk(probs[0, target_index, :], k + 1)

            # Score each candidate
            for prob, cand in zip(cand_probs, cands):
                if not cand == target_token:
                    denominator = (1 - original_prob) if original_prob < 1 else 1E-6
                    score = prob.item() / denominator
                    cand_scores[(cand, target_index)] = score
                
        

        # Generate and return the neighborhood of sequences
        neighborhood = []
        top_keys = nlargest(n, cand_scores, key=cand_scores.get)
        for cand, index in top_keys:
            neighbor = torch.clone(tokenized)
            neighbor[0, index] = cand
            neighborhood.append(self._tokenizer.batch_decode(neighbor)[0])

        return neighborhood
    
    def generate_neighbors_inbatch(self, texts, p=0.7, k=3, n=25):
        """
        For TEXT, generates a neighborhood of single-token replacements, considering the best K token replacements
        at each position in the sequence and returning the top N neighboring sequences.
        
        This is a method that generates the neighborhood in batch.

        https://aclanthology.org/2023.findings-acl.719.pdf
        
        Args:
            text (str): The input text to generate the neighborhood.
            p (float): The dropout probability.
            k (int): The number of top candidates to consider at each position.
            n (int): The number of neighboring sequences to return.
        """
        tokenized = self._tokenizer(texts, 
                                    return_tensors='pt', 
                                    truncation=True,
                                    padding='longest',
                                    max_length=256).input_ids.to(self.model.device)
        batch_size = tokenized.shape[0]
        
        dropout = torch.nn.Dropout(p)
        
        if isinstance(self.model, transformers.RobertaForCausalLM):
            embedding = self.model.roberta.embeddings(tokenized)
        else:
            raise RuntimeError(f'Unsupported model type for neighborhood generation: {type(self.model)}')
        
        # Apply dropout all in once
        dropout_embedding = dropout(embedding)
        
        # sequence length
        seq_len = tokenized.shape[1]
        all_scores = torch.empty(batch_size, 0, k + 1, device=self.model.device)
        all_cands = torch.empty(batch_size, 0, k + 1, device=self.model.device, dtype=torch.int)
        for target_index in range(1, seq_len):
            target_token = tokenized[:, target_index]

            # Apply dropout only to the target token embedding in the sequence
            modified_embedding = torch.cat([
                embedding[:, :target_index],
                dropout_embedding[:, target_index].unsqueeze(1), # apply dropout to the target token, unsqueeze to match the shape
                embedding[:, target_index+1:]
            ], dim=1)

            # Get model's predicted posterior distributions over all positions in the sequence
            with torch.no_grad():
                logits = self.model(inputs_embeds=modified_embedding).logits
                probs = torch.softmax(logits, dim=2)
            batch_indices = torch.arange(batch_size)
            original_probs = probs[batch_indices, target_index, target_token]   # shape: [batch_size], original probabilities

            # Find the K most probable token replacements, not including the target token
            # Find top K+1 first because target could still appear as a candidate
            cand_probs, cands = torch.topk(probs[:, target_index, :], k + 1)

            # mask the target token
            mask = cands == target_token.unsqueeze(1)
            topk_probs = torch.where(mask, torch.tensor(0.0), cand_probs)
            # Score each candidate
            topk_probs = topk_probs / (1 - original_probs.unsqueeze(1) + 1e-6)
            all_scores = torch.cat([all_scores, topk_probs.unsqueeze(1)], dim=1)
            all_cands = torch.cat([all_cands, cands.unsqueeze(1)], dim=1)
                
        # Generate and return the neighborhood of sequences
        flatten_scores = all_scores.view(batch_size, -1) # [batch_size, seq_len * k+1]
        top_scores, top_indices = torch.topk(flatten_scores, n, dim=1)
        flatten_cands = all_cands.view(batch_size, -1)
        # top_scores: [batch_size, n], top n scores
        # top_indices: [batch_size, n], corresponding indices for top n scores
        neigh_position = top_indices // (all_scores.shape[2]) + 1
        neigh_tokens = flatten_cands.gather(1, top_indices)
        
        neighborhoods = np.empty((batch_size, n), dtype=object)
        for i in range(n):
            neighbor = torch.clone(tokenized)
            neighbor[batch_indices, neigh_position[:, i]] = neigh_tokens[:, i]
            new_neighbor = np.array(self._tokenizer.batch_decode(neighbor, skip_special_tokens=True))
            neighborhoods[:, i] = new_neighbor
            
        
        neighborhoods = neighborhoods.tolist()
        return neighborhoods
        


class PeftCasualLM(FinetunedCasualLM):
    def load_local_model(self):
        super().load_local_model(self.arch)
        from peft.peft_model import PeftModel
        print(f"load peft module from {self.model_path}")
        try:
            self.model = PeftModel.from_pretrained(self.model, 
                                                   self.model_path, 
                                                   device_map='cuda')
        except:
            self.model = PeftModel.from_pretrained(self.model, 
                                                   self.model_path, 
                                                   device_map='auto',
                                                   offload_folder='./offload')


if __name__ == '__main__':
    # Testing purposes
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()
    
    model = FinetunedCasualLM(args=args, model_path='openai-community/gpt2')
    print(model.query('Hello, how are you?'))
    print("DONE")
