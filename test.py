from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
prompt = "JJY是一个"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

output = model.generate(input_ids, 
               do_sample=True,
               temperature=0.9,
               max_length=100,)
print(tokenizer.decode(output[0].cpu().tolist(), skip_special_tokens=True))