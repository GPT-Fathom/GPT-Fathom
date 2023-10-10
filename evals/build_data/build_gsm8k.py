"""Build the relevant file for lambada dataset"""
import csv
import json
import os

from datasets import load_dataset

# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "gsm8k"), exist_ok=True)
samples = load_dataset("rookshanks/gsm8k", split="test")

data = []
for row in samples:
    pair = {"question": row["question"], "answer": row["answer"]}
    data.append(pair)

file_path = os.path.join(registry_path, "data/gsm8k/samples.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

# Build the few-shot samples

file = open("./gsm8k/prompt_original.txt", "r").read()
file = file.split("Question: ")
file = file[1:]
data = []
for i in range(8):
    text = file[i].split("Let's think")
    q, a = text[0], "Let's think" + text[1]
    pair = {"question": q, "answer": a}
    data.append(pair)

file_path = os.path.join(registry_path, "data/gsm8k/fewshot.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")


# # build gsm8k yaml

eval_yaml = ""
cur = f"""
gsm8k-8shotCoT:
  id: gsm8k.val.ab-v1
  metrics: [accuracy]
gsm8k.val.ab-v1:
  class: evals.elsuite.math_problem:MATH_PROBLEM
  args:
    samples_jsonl: gsm8k/samples.jsonl
    few_shot_jsonl: gsm8k/fewshot.jsonl
    num_few_shot: 8

gsm8k-5shotCoT:
  id: gsm8k.val.ab-v1
  metrics: [accuracy]
gsm8k.val.ab-v1:
  class: evals.elsuite.math_problem:MATH_PROBLEM
  args:
    samples_jsonl: gsm8k/samples.jsonl
    few_shot_jsonl: gsm8k/fewshot.jsonl
    num_few_shot: 5
"""
eval_yaml += cur
with open(os.path.join(registry_path, "evals", "gsm8k.yaml"), "w") as f:
    f.write(eval_yaml)
