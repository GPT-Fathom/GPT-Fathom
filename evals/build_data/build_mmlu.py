import json

from datasets import load_dataset

name = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

import os

registry_path = os.path.join("..", "registry")


eval_yaml = ""
for item in name:
    file_name = item
    id_name = item.replace("_", "-")
    instruct_name = item.replace("_", " ")
    # print(file_name, id_name)
    cur = f"""
mmlu-{id_name}-3shot:
  id: mmlu-{id_name}.val.ab-v1
  metrics: [accuracy]
mmlu-{id_name}.val.ab-v1:
  class: evals.elsuite.basic.multiple_choice:MultipleChoice
  args:
    dataset: hf://hendrycks_test?name={file_name}&split=test
    few_shot: hf://hendrycks_test?name={file_name}&split=dev
    instructions: The following are multiple choice questions (with answers) about {instruct_name}.
    no_MC_prompt: True
    num_few_shot: 3

mmlu-{id_name}-5shot:
  id: mmlu-{id_name}.val.ab-v2
  metrics: [accuracy]
mmlu-{id_name}.val.ab-v2:
  class: evals.elsuite.basic.multiple_choice:MultipleChoice
  args:
    dataset: hf://hendrycks_test?name={file_name}&split=test
    few_shot: hf://hendrycks_test?name={file_name}&split=dev
    instructions: The following are multiple choice questions (with answers) about {instruct_name}.
    num_few_shot: 5
    no_MC_prompt: True

mmlu-{id_name}-CoT:
  id: mmlu-{id_name}.val.ab-v3
  metrics: [accuracy]
mmlu-{id_name}.val.ab-v3:
  class: evals.elsuite.dataset_specific.mmlu_CoT:MC_CoT
  args:
    dataset: hf://hendrycks_test?name={file_name}&split=test
    task_name: {file_name}
    no_MC_prompt: True
    CoT: True
"""
    eval_yaml += cur
with open(os.path.join(registry_path, "evals", "mmlu.yaml"), "w") as f:
    f.write(eval_yaml)
