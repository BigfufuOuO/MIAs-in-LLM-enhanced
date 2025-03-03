import numpy as np
import re

class SvaMIAGenerator:
    def __init__(self, 
                 mask_model: any = None, 
                 mask_tokenizer: any = None,):
        self.model = mask_model
        self.tokenizer = mask_tokenizer
        
        
    def mask_texts(self,
                   texts: list,
                   ratio: float = 0.2):
        """
        Mask only one text.
        
        Args:
            texts: The texts to mask.
            ratio: The ratio of the masked tokens.
        
        Returns:
            masked_indices (np.array): 
                The indices of the masked tokens.
            masked_texts (list):
                The masked texts.
            splitted_masked_texts (np.array): 
                The splitted masked texts.
            num_masks (np.array): 
                The number of masks.
        """
        np.random.seed(None)
        masked_texts = np.empty(len(texts), dtype=object)
        masked_indices = np.empty(len(texts), dtype=object)
        splitted_masked_texts = np.empty(len(texts), dtype=object)
        for i, text in enumerate(texts):
            words = text.split(' ')
            num_to_mask = int(len(words) * ratio)
            mask_idx = np.random.choice(len(words), num_to_mask, replace=False)
            mask_idx = np.sort(mask_idx)
            extra_id_index = 0
            for j in mask_idx:
                words[j] = f'<extra_id_{extra_id_index}>'
                extra_id_index += 1
            masked_text = ' '.join(words)
            masked_texts[i] = masked_text
            masked_indices[i] = mask_idx
            splitted_masked_texts[i] = np.array(words)
            
        masked_texts = masked_texts.tolist()
        num_masks = [len(x) for x in masked_indices]
        return masked_indices, masked_texts, splitted_masked_texts, num_masks

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
                                        top_p=0.8,)
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
        masked_indices, masked_texts, splitted_masked_texts, num_masks = self.mask_texts(texts)
        inputs = self.tokenizer(masked_texts, 
                                return_tensors='pt', 
                                padding=True,
                                truncation=True).to(self.model.device)

        perturbed_texts = np.empty((len(texts), n_perturbed), dtype=object)
        n_failed = 0
        for n in range(n_perturbed):
            extracted_fills = self.extract_fills(inputs)
            
            # replace each mask token with the corresponding fill
            for i, mask_idx in enumerate(masked_indices):
                if len(extracted_fills[i]) < num_masks[i]:
                    mask_idx, extracted_fill_temp = self.try_again(texts[i])
                    extracted_fills[i] = extracted_fill_temp[0]
                    n_failed += 1
                splitted_masked_texts[i][mask_idx] = extracted_fills[i][:num_masks[i]]
                
            perturbed_texts[:, n] = splitted_masked_texts
        
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
            texts = [text]
            masked_indices, masked_texts, splitted_masked_texts, num_masks = self.mask_texts(texts)
            inputs = self.tokenizer(masked_texts, 
                                    return_tensors='pt', 
                                    padding=True,
                                    truncation=True).to(self.model.device)
            extracted_fills = self.extract_fills(inputs)
            if len(extracted_fills[0]) >= num_masks[0]:
                return masked_indices[0], extracted_fills

    def gen_perturbed_texts(self,
                            texts: list,):
        outputs = self.tokenize_masks(texts, )
    
    