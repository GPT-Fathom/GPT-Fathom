import json
import os

from datasets import load_dataset

# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "longbench"), exist_ok=True)

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
            "dureader"]

for dataset in datasets:
    file = load_dataset("THUDM/LongBench", dataset, split="test")
    data = []
    for row in file:
        pair = {
            "passage": row["context"],
            "question": row["input"],
            "ideal": row["answers"],
        }
        data.append(pair)
    file_name = dataset + ".jsonl"
    file_path = os.path.join(registry_path, "data/longbench", file_name)
    print(file_path)
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")