from transformers import AutoTokenizer

tokenzier = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

print(tokenzier.encode("Hello, my name is Wuli."))