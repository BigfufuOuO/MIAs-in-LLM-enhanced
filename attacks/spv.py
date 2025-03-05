import numpy as np
import re

class SpvMIAGenerator:
    def __init__(self, 
                 mask_model: any = None, 
                 mask_tokenizer: any = None,):
        self.model = mask_model
        self.tokenizer = mask_tokenizer
        
        
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

    