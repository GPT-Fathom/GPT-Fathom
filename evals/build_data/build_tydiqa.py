# khalidalt/tydiqa-goldp

import csv
import json
import os

from datasets import load_dataset


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


# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "tydi"), exist_ok=True)

language = [
    "arabic",
    "bengali",
    "english",
    "finnish",
    "indonesian",
    "japanese",
    "korean",
    "russian",
    "swahili",
    "telugu",
    "thai",
]
for la in language:
    file = load_dataset("khalidalt/tydiqa-goldp", la, split="validation")
    data = []
    for row in file:
        content = create_chat_prompt(row["passage_text"] + "\n" + row["question_text"])
        pair = {"input": content, "ideal": row["answers"]["text"]}
        data.append(pair)

    file_path = os.path.join(registry_path, "data/tydiqa/samples_" + la + ".jsonl")
    print(file_path)
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


# Build the few-shot samples
for la in language:
    file = load_dataset("khalidalt/tydiqa-goldp", la, split="train")
    data = []
    for row in file:
        content = create_fewshot_prompt(
            row["passage_text"] + "\n" + row["question_text"], row["answers"]["text"]
        )
        pair = {"sample": content}
        data.append(pair)
        break  # we only need one-shot

    file_path = os.path.join(registry_path, "data/tydiqa/fewshot_" + la + ".jsonl")
    print(file_path)
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")

# For no context:  ------------------------
# Build the test samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "tydi"), exist_ok=True)

language = [
    "arabic",
    "bengali",
    "english",
    "finnish",
    "indonesian",
    "japanese",
    "korean",
    "russian",
    "swahili",
    "telugu",
    "thai",
]
for la in language:
    file = load_dataset("khalidalt/tydiqa-goldp", la, split="validation")
    data = []
    for row in file:
        content = create_chat_prompt(row["question_text"])
        pair = {"input": content, "ideal": row["answers"]["text"]}
        data.append(pair)

    file_path = os.path.join(
        registry_path, "data/tydiqa/samples_nocontext_" + la + ".jsonl"
    )
    print(file_path)
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


# Build the few-shot samples
for la in language:
    file = load_dataset("khalidalt/tydiqa-goldp", la, split="train")
    data = []
    for row in file:
        content = create_fewshot_prompt(row["question_text"], row["answers"]["text"][0])
        pair = {"sample": content}
        # pair = {"input": row["question_text"], "ideal": row["answers"]["text"]}
        data.append(pair)
        break  # we only need one-shot

    file_path = os.path.join(
        registry_path, "data/tydiqa/fewshot_nocontext_" + la + ".jsonl"
    )
    print(file_path)
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


# build tydi yaml

eval_yaml = ""
for la in language:
    cur = f"""
tydi-{la}-1shotgold:
  id: tydi-{la}.val.ab-v1
  metrics: [f1_score]
tydi-{la}.val.ab-v1:
  class: evals.elsuite.basic.fuzzy_match:FuzzyMatch
  args:
    samples_jsonl: tydiqa/samples_{la}.jsonl
    few_shot_jsonl: tydiqa/fewshot_{la}.jsonl
    num_few_shot: 1
tydi-{la}-1shotnocontext:
  id: tydi-{la}.val.ab-v2
  metrics: [f1_score]
tydi-{la}.val.ab-v2:
  class: evals.elsuite.basic.fuzzy_match:FuzzyMatch
  args:
    samples_jsonl: tydiqa/samples_nocontext_{la}.jsonl
    few_shot_jsonl: tydiqa/fewshot_nocontext_{la}.jsonl
    num_few_shot: 1
"""
    eval_yaml += cur
with open(os.path.join(registry_path, "evals", "tydi.yaml"), "w") as f:
    f.write(eval_yaml)
