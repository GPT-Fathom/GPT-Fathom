"""Build the relevant file for lambada dataset"""
import csv
import json
import os

from datasets import load_dataset
from mgsm_prompt.exemplars import MGSM_EXEMPLARS


def create_chat_prompt(text):
    return [
        {
            "role": "system",
            "content": "Follow the given examples and answer the question.",
        },
        {"role": "user", "content": text},
    ]


def create_fewshot_prompt(question, answer):
    return [
        {"role": "system", "name": "example_user", "content": question},
        {"role": "system", "name": "example_assistant", "content": answer},
    ]


language = [
    "bn",
    "de",
    "es",
    "fr",
    "ja",
    "ru",
    "sw",
    "te",
    "th",
    "zh",
]  # Test the ten language except English
# print(MGSM_EXEMPLARS)


# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "mgsm"), exist_ok=True)

for la in language:

    data = []
    with open("./mgsm_prompt/mgsm_" + la + ".tsv", "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            pair = {"question": row[0], "answer": row[1]}
            data.append(pair)

    file_path = os.path.join(registry_path, "data/mgsm/samples_" + la + ".jsonl")
    print(file_path)
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")

# Build the few-shot samples

for la in language:
    data = []
    for i in range(1, 1 + 8):
        pair = {
            "question": MGSM_EXEMPLARS[la][str(i)]["q"],
            "answer": MGSM_EXEMPLARS[la][str(i)]["a"],
        }
        data.append(pair)

    file_path = os.path.join(registry_path, "data/mgsm/fewshot_" + la + ".jsonl")
    print(file_path)
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


# build mgsm yaml

eval_yaml = ""
for la in language:
    cur = f"""
mgsm-{la}-8shotCoT:
  id: mgsm-{la}.val.ab-v1
  metrics: [accuracy]
mgsm-{la}.val.ab-v1:
  class: evals.elsuite.basic.math_problem:MATH_PROBLEM
  args:
    samples_jsonl: mgsm/samples_{la}.jsonl
    few_shot_jsonl: mgsm/fewshot_{la}.jsonl
    num_few_shot: 6

"""
    eval_yaml += cur
with open(os.path.join(registry_path, "evals", "mgsm.yaml"), "w") as f:
    f.write(eval_yaml)
