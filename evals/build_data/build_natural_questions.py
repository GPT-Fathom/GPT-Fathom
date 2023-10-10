"""Build the relevant file for natural question dataset"""
import json
import os

from datasets import load_dataset


def create_chat_prompt(text):
    return [
        {"role": "system", "content": "Please answer the question:"},
        {"role": "user", "content": text},
    ]


def create_fewshot_prompt(question, answer):
    return [
        {"role": "system", "name": "example_user", "content": question},
        {"role": "system", "name": "example_assistant", "content": answer},
    ]


# Build the test samples
samples = load_dataset("lucadiliello/naturalquestionsshortqa", split="validation")

registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "NaturalQuestion"), exist_ok=True)

data = []

for item in samples:
    context = item["question"]
    content = create_chat_prompt(context)
    pair = {"input": content, "ideal": item["answers"]}
    data.append(pair)
file_path = os.path.join(registry_path, "data/NaturalQuestion/samples.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")


# Build the few shot samples

registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "NaturalQuestion"), exist_ok=True)
samples = load_dataset("lucadiliello/naturalquestionsshortqa", split="train")

data = []
samples = [(item["question"], item["answers"]) for item in samples]
for item in samples[:64]:
    context = item[0]
    content = create_fewshot_prompt(context, item[1][0])
    pair = {"sample": content}
    data.append(pair)
file_path = os.path.join(registry_path, "data/NaturalQuestion/fewshot.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")
