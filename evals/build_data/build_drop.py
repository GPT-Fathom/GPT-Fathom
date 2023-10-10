import json
import os

from datasets import load_dataset

# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "drop"), exist_ok=True)

file = load_dataset("drop", split="validation")
data = []
for row in file:
    pair = {
        "passage": row["passage"],
        "question": row["question"],
        "ideal": row["answers_spans"]["spans"],
    }
    data.append(pair)

file_path = os.path.join(registry_path, "data/drop/samples.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

# Build the few shot samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "drop"), exist_ok=True)

file = load_dataset("drop", split="train")
data = []
for row in file:
    pair = {
        "passage": row["passage"],
        "question": row["question"],
        "ideal": row["answers_spans"]["spans"],
    }
    data.append(pair)

data = data[:3]
file_path = os.path.join(registry_path, "data/drop/fewshot.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")
