import json
import os

from datasets import load_dataset


def create_chat_prompt(text):
    return [{"role": "user", "content": text}]


def create_fewshot_prompt(question, answer):
    return [
        {"role": "system", "name": "example_user", "content": question},
        {"role": "system", "name": "example_assistant", "content": answer},
    ]


# Build the test samples for realtoxic dataset
samples = load_dataset("allenai/real-toxicity-prompts", split="train")
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "realtoxic"), exist_ok=True)

data = []

for item in samples:
    context = item["prompt"]["text"]
    content = create_chat_prompt(context)
    pair = {"input": content, "ideal": "placeholder"}
    data.append(pair)

file_path = os.path.join(registry_path, "data/realtoxic/samples.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")
