import json
import os

from datasets import load_dataset

# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "race"), exist_ok=True)

file = load_dataset("race", "high", split="test")
data = []
for row in file:
    pair = {
        "passage": row["article"],
        "question": row["question"],
        "options": row["options"],
        "ideal": ord(row["answer"]) - ord("A"),
    }
    data.append(pair)

file_path = os.path.join(registry_path, "data/race/samples_h.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

# Build the few shot samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "race"), exist_ok=True)

file = load_dataset("race", "high", split="test")
data = []
for row in file:
    pair = {
        "passage": row["article"],
        "question": row["question"],
        "options": row["options"],
        "ideal": ord(row["answer"]) - ord("A"),
    }
    data.append(pair)

data = data[:10]
file_path = os.path.join(registry_path, "data/race/fewshot_h.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")


# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "race"), exist_ok=True)

file = load_dataset("race", "middle", split="test")
data = []
for row in file:
    pair = {
        "passage": row["article"],
        "question": row["question"],
        "options": row["options"],
        "ideal": ord(row["answer"]) - ord("A"),
    }
    data.append(pair)

file_path = os.path.join(registry_path, "data/race/samples_m.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

# Build the few shot samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "race"), exist_ok=True)

file = load_dataset("race", "middle", split="test")
data = []
for row in file:
    pair = {
        "passage": row["article"],
        "question": row["question"],
        "options": row["options"],
        "ideal": ord(row["answer"]) - ord("A"),
    }
    data.append(pair)

data = data[:10]
file_path = os.path.join(registry_path, "data/race/fewshot_m.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")
