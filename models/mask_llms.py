from .LLMBase import LLMBase
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import numpy as np
import re

class MaskLanguageModel(LLMBase):
    def __init__(self, args=None, 
                 arch=None, 
                 model_path='google-t5/t5-base', 
                 max_seq_len=1024):
        if ':' in model_path:
            model_path, self.model_revision = model_path.split(':')
        else:
            self.model_revision = 'main'

        if arch is None:
            self.arch = model_path
        else:
            self.arch = arch

        # arguments
        self.args = args
        # default
        self.tokenizer_use_fast = True
        self.max_seq_len = max_seq_len
        self.verbose = True

        super().__init__(model_path=model_path)
        
    @property
    def tokenizer(self):
        return self._tokenizer
        
    def load_local_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                            quantization_config=bnb_config,
                                                            torch_dtype=torch.bfloat16,
                                                            device_map="auto",
                                                            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                        use_fast=True)
        
        self._tokenizer.padding_side = "left"
        
    def mask_texts(self,
                   n_perturbed: int,
                   texts: list,
                   ratio: float = 0.1,):
        """
        Mask only one text.
        
        Args:
            texts: The texts to mask.
            ratio: The ratio of the masked tokens.
        
        Returns:
            masked_indices (list, shape: (n_batches * n_perturbed)): 
                The indices of the masked tokens.
            flatten_words (list, shape: (n_batches * n_perturbed)): 
                The splitted masked texts.
            flatten_num_to_mask (np.array, shape: (n_batches * n_perturbed)): 
                The number of masks.
        """
        np.random.seed(None)
        
        # expand text
        texts = np.array(texts)
        texts = texts[:, np.newaxis]
        texts = np.repeat(texts, n_perturbed, axis=-1)
        words = np.char.split(texts, sep=' ')
        vectorized_len = np.vectorize(len)
        word_counts = vectorized_len(words)
        num_to_mask = np.ceil(word_counts * ratio).astype(int)
        flatten_word_counts = word_counts.flatten()
        flatten_num_to_mask = num_to_mask.flatten()
        flatten_words = words.flatten()
        masked_indices = []
        for i, (word_count, num_to_mask) in enumerate(zip(flatten_word_counts, flatten_num_to_mask)):
            if word_count < 2:
                # add random space to avoid error
                start = np.random.randint(0, len(flatten_words[i][0]) - 12)
                flatten_words[i][0] = flatten_words[i][0][:start] + ' ' + flatten_words[i][0][start:start+10] + ' ' + flatten_words[i][0][start+10:]
                flatten_words[i] = flatten_words[i][0].split(' ')
                word_count = len(flatten_words[i])
                
            mask_idx = np.random.choice(word_count, num_to_mask, replace=False)
            mask_idx = np.sort(mask_idx)
            extra_id_index = 0
            for j in mask_idx:
                flatten_words[i][j] = f'<extra_id_{extra_id_index}>'
                extra_id_index += 1
            masked_indices.append(mask_idx.tolist())
            
        flatten_words = flatten_words.tolist()
        
        return masked_indices, flatten_words, flatten_num_to_mask
    
    def mask_one_text(self,
                      text: str,
                      ratio: float = 0.1):
        """
        Mask only one text.
        """
        words = text.split(' ')
        num_to_mask = int(len(words) * ratio) + 1
        mask_idx = np.random.choice(len(words), num_to_mask, replace=False)
        mask_idx = np.sort(mask_idx)
        extra_id_index = 0
        for j in mask_idx:
            words[j] = f'<extra_id_{extra_id_index}>'
            extra_id_index += 1
        return mask_idx, words, num_to_mask

    def extract_fills(
                    self,
                    inputs: any = None,
                    ):
        """
        Extract the fills from the masked text.
        
        Args:
            inputs: The tokenized inputs to the model.
        """
        outputs = self.model.generate(**inputs, do_sample=True,
                                        num_beams=3,
                                        top_p=0.8,
                                        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # extract fills
        outputs = [x.replace("<pad>", "").replace("</s>", "").strip() for x in outputs]
        pattern = re.compile(r'<extra_id_\d+>')
        extracted_fills = [pattern.split(x)[1:] for x in outputs]
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
        return extracted_fills

    def tokenize_masks(self,
                       texts: list,
                       n_perturbed: int = 10):
        """
        Tokenize multiple texts which are masked.
        
        Returns:
            perturbed_texts (np.array): The perturbed texts
            n_failed (int): The number of failed attempts.
        """
        masked_indices, flatten_words, flatten_num_to_mask = self.mask_texts(texts=texts, 
                                                                             n_perturbed=n_perturbed)
        masked_texts = [' '.join(x) for x in flatten_words]
        inputs = self.tokenizer(masked_texts, 
                                return_tensors='pt', 
                                padding=True,
                                truncation=True).to(self.model.device)

        n_failed = 0
        extracted_fills = self.extract_fills(inputs)
        
        splitted_masked_texts = np.empty(len(texts) * n_perturbed, dtype=object)
        # replace each mask token with the corresponding fill
        for i, mask_idx in enumerate(masked_indices):        
            splitted_masked_texts[i] = np.array(flatten_words[i])
            if len(extracted_fills[i]) < flatten_num_to_mask[i]:
                n_failed += 1
                n_mask = len(extracted_fills[i])
                splitted_masked_texts[i][mask_idx[:n_mask]] = extracted_fills[i]
            else:
                splitted_masked_texts[i][mask_idx] = extracted_fills[i][:flatten_num_to_mask[i]]
        
        perturbed_texts = np.array([' '.join(x) for x in splitted_masked_texts])
        perturbed_texts = perturbed_texts.reshape(len(texts), n_perturbed)
            
        perturbed_texts = perturbed_texts.tolist()
        return n_failed, perturbed_texts
    
    def try_again(self,
                  text: str,):
        """
        Try mask and fill again.
        
        Args:
            text: The single text to mask and fill.
            
        Returns:
            masked_indices (np.array): The indices of the masked tokens.
            extracted_fills (list): The extracted fills of the specific text.
        """
        while True:
            masked_indices, words, num_masks = self.mask_one_text(text)
            masked_texts = ' '.join(words)
            inputs = self.tokenizer(masked_texts, 
                                    return_tensors='pt', 
                                    padding=True,
                                    truncation=True).to(self.model.device)
            extracted_fills = self.extract_fills(inputs)
            if len(extracted_fills[0]) >= num_masks:
                return masked_indices, words, extracted_fills, num_masks
            
    def generate_perturbed_texts(self,
                                 texts: list,
                                 n_perturbed: int = 10,
                                 ratio: float = 0.2,):
        """
        Generate perturbed texts.
        
        Args:
            texts: The texts to mask and fill.
            n_perturbed: The number of perturbed texts to generate for each text.
            ratio: The ratio of the masked tokens.
            
        Returns:
            perturbed_texts (list): The perturbed texts.
        """
        tokenized = self._tokenizer(texts, 
                                    return_tensors='pt', 
                                    truncation=True,
                                    padding='longest',
                                    max_length=self.max_seq_len).input_ids.to(self.model.device)
        batch_size = tokenized.shape[0]
        # tokenized shape: (batch_size, max_seq_len)
        
        # randomly select some texts to mask through tokenizer.mask_tokens
        mask_token = self._tokenizer.mask_token_id
        
        perturbed_texts = np.empty((len(texts), n_perturbed), dtype=object)
        for i in range(n_perturbed):
            input_ids = tokenized.clone()
            for j in range(batch_size):
                np.random.seed(None)
                length = tokenized[j].shape[0]
                num_mask = int(length * ratio) + 1
                mask_idx = np.random.choice(length, num_mask, replace=False)
                input_ids[j, mask_idx] = mask_token
                
            # fill the masked tokens
            attention_mask = torch.where(input_ids == self._tokenizer.pad_token_id, 0, 1).to(self.model.device)
            max_length = int(tokenized.shape[1] * 1.1)
            generation_config = GenerationConfig(do_sample=True,
                                                 max_length=max_length,
                                                  num_beams=3,
                                                  top_p=0.95,
                                                  clean_up_tokenization_spaces=False,
                                                  )
            with torch.no_grad():
                outputs = self.model.generate(input_ids=input_ids, 
                                               attention_mask=attention_mask,
                                               generation_config=generation_config,)
                outputs = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            perturbed_texts[:, i] = outputs
        
        perturbed_texts = perturbed_texts.tolist()
        return perturbed_texts
                
        
        

    