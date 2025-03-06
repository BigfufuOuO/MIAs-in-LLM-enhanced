from datasets import Dataset

data = {
    "text": [
        ["This is a test sentence.", "This is another test sentence."],
        ["This is a test sentence.", "This is another test sentence."],
        ["This is a test sentence.", "This is another test sentence."],
        ["This is a test sentence.", "This is another test sentence."],
        
    ]}

def handle_text(text: list):
    print(text)
    return {
        "how_many_sentences": [len(text)],
    }

dataset = Dataset.from_dict(data)
dataset = dataset.map(lambda x: {"text": handle_text(x["text"])},
                      batch_size=2,
                      batched=True)

print(dataset)