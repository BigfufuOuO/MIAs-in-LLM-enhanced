import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

tokenizer.pad_token = tokenizer.eos_token

sentences = ["The quick brown fox jumps over the lazy dog.",
             "The dog slept over the veranda."]

model.eval()
# get loss
with torch.no_grad():
    input_ids = tokenizer.encode(sentences, return_tensors="pt", padding=True, truncation=True)
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    print(loss)
with torch.no_grad():
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    input_ids = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).input_ids
    outputs = model(input_ids, labels=input_ids)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    batch_size, seq_len = shift_labels.shape
    padding_mask = (shift_labels != tokenizer.pad_token_id).float()
    loss_per_sentence = loss.view(batch_size, seq_len)
    loss_per_sentence = (loss_per_sentence * padding_mask).sum(dim=1) / padding_mask.sum(dim=1)
    print(loss_per_sentence)
    print(loss_per_sentence.mean().item())
    print(loss)
