from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

prompt = "ffghhsadjhji"

input = tokenizer(prompt, return_tensors="pt")

output = model(**input, labels=input.input_ids)
print(output.loss.item())