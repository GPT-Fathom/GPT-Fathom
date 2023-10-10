"""Build the relevant file for lambada dataset"""
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


# Build the test samples for arc-easy
samples = load_dataset("ai2_arc", "ARC-Easy", split="test")

registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "arc"), exist_ok=True)

data = []

for item in samples:
    pair = {
        "question": item["question"],
        "choices": item["choices"]["text"],
        "answer": item["choices"]["label"].index(item["answerKey"]),
    }
    data.append(pair)

file_path = os.path.join(registry_path, "data/arc/samples_easy.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")


# Build the test samples for arc-challenge
samples = load_dataset("ai2_arc", "ARC-Challenge", split="test")

registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "arc"), exist_ok=True)

data = []

for item in samples:
    pair = {
        "question": item["question"],
        "choices": item["choices"]["text"],
        "answer": item["choices"]["label"].index(item["answerKey"]),
    }
    data.append(pair)

file_path = os.path.join(registry_path, "data/arc/samples_challenge.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")


# Build the few shot samples for Easy

import os

registry_path = os.path.join("..", "registry")
samples = load_dataset("ai2_arc", "ARC-Easy", split="validation")
os.makedirs(os.path.join(registry_path, "data", "arc"), exist_ok=True)

data = []
for item in samples:
    pair = {
        "question": item["question"],
        "choices": item["choices"]["text"],
        "answer": item["choices"]["label"].index(item["answerKey"]),
    }
    data.append(pair)
data = data[:50]
file_path = os.path.join(registry_path, "data/arc/fewshot_easy.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

# Build the few shot samples for Challenge

import os

registry_path = os.path.join("..", "registry")
samples = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
os.makedirs(os.path.join(registry_path, "data", "arc"), exist_ok=True)

data = []
for item in samples:
    pair = {
        "question": item["question"],
        "choices": item["choices"]["text"],
        "answer": item["choices"]["label"].index(item["answerKey"]),
    }
    data.append(pair)
data = data[:50]
file_path = os.path.join(registry_path, "data/arc/fewshot_challenge.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")
