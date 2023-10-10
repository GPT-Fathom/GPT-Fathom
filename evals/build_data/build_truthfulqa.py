import json
import os

from datasets import load_dataset


def create_chat_prompt(text):
    return [
        {
            "role": "system",
            "content": "Please answer with the word which is most likely to follow:",
        },
        {"role": "user", "content": text},
    ]


def create_fewshot_prompt(question, answer):
    return [
        {"role": "system", "name": "example_user", "content": question},
        {"role": "system", "name": "example_assistant", "content": answer},
    ]


# Build the test samples for truthfulqa dataset
samples = load_dataset("truthful_qa", "multiple_choice", split="validation")
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "truthfulqa"), exist_ok=True)


data = []

for item in samples:
    pair = {
        "question": item["question"],
        "choices": item["mc1_targets"]["choices"],
        "answer": item["mc1_targets"]["labels"].index(1),
    }
    data.append(pair)

fewshot = data[0]
test_samples = data[1:]

file_path = os.path.join(registry_path, "data/truthfulqa/samples.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in test_samples:
        json.dump(entry, f)
        f.write("\n")

file_path = os.path.join(registry_path, "data/truthfulqa/fewshot.jsonl")
print(file_path)
with open(file_path, "w") as f:
        json.dump(fewshot, f)
        f.write("\n")
