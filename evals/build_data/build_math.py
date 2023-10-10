"""Build the relevant file for math dataset"""
import csv
import json
import os

from datasets import load_dataset
from math_prompt.exemplars import MATH_EXEMPLARS

from evals.utils.math_util import (
    clean_numbers,
    is_equiv,
    last_boxed_only,
    last_boxed_only_string,
    remove_boxed,
)

# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "math"), exist_ok=True)
samples = load_dataset("competition_math", split="test")

data = []
for row in samples:
    output_str = last_boxed_only_string(row["solution"])
    output = remove_boxed(output_str)
    pair = {"question": row["problem"], "answer": output}
    data.append(pair)

file_path = os.path.join(registry_path, "data/math/samples.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

# Build the few-shot samples

data = []
for i in range(1, 5):
    q, a = MATH_EXEMPLARS[str(i)]["q"], MATH_EXEMPLARS[str(i)]["a"]
    pair = {"question": q, "answer": a}
    data.append(pair)

file_path = os.path.join(registry_path, "data/math/fewshot.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")
