#!/usr/bin/env python

import json
import os

from datasets import load_dataset

# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "quality"), exist_ok=True)


#file = load_dataset("quality", split="dev")
with open("quality.dev", "r") as dev_file:
    dev_data = [json.loads(line) for line in dev_file.readlines()]

data = []
for row in dev_data:
    questions = row["questions"]
    for i in range(len(questions)):
        pair = {
            "passage": row["article"],
            "question": questions[i]["question"],
            "options": questions[i]["options"],
            "ideal": questions[i]["gold_label"] - 1,
        }
        data.append(pair)

file_path = os.path.join(registry_path, "data/quality/samples.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")