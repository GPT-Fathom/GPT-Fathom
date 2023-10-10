import json

from datasets import load_dataset

name = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


def create_chat_prompt(text, instruction):
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]


def create_fewshot_prompt(question, answer):
    return [
        {"role": "system", "name": "example_user", "content": question},
        {"role": "system", "name": "example_assistant", "content": answer},
    ]


import os

registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "bbh"), exist_ok=True)

# Build test samples
eval_yaml = ""
for task in name:
    data = []
    file_path = os.path.join(registry_path, f"data/bbh/samples_{task}.jsonl")
    samples = load_dataset("lukaemon/bbh", task, split="test")
    for sample in samples:
        pair = {"input": sample["input"], "ideal": sample["target"]}
        data.append(pair)

    print(file_path)
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")

        cur = f"""
bbh-{task}-3shot:
  id: bbh-{task}.val.ab-v1
  metrics: [accuracy]
bbh-{task}.val.ab-v1:
  class: evals.elsuite.dataset_specific.bbh:BBH
  args:
    task_name: {task}
    samples_jsonl: bbh/samples_{task}.jsonl
    CoT: False

bbh-{task}-3shot-CoT:
  id: bbh-{task}.val.ab-v2
  metrics: [accuracy]
bbh-{task}.val.ab-v2:
  class: evals.elsuite.dataset_specific.bbh:BBH
  args:
    task_name: {task}
    samples_jsonl: bbh/samples_{task}.jsonl
    CoT: True

"""

    eval_yaml += cur
with open(os.path.join(registry_path, "evals", "bbh.yaml"), "w") as f:
    f.write(eval_yaml)
