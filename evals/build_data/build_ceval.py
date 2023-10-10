import json

from datasets import load_dataset

name = [
    "computer_network",
    "operating_system",
    "computer_architecture",
    "college_programming",
    "college_physics",
    "college_chemistry",
    "advanced_mathematics",
    "probability_and_statistics",
    "discrete_mathematics",
    "electrical_engineer",
    "metrology_engineer",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "middle_school_mathematics",
    "middle_school_biology",
    "middle_school_physics",
    "middle_school_chemistry",
    "veterinary_medicine",
    "college_economics",
    "business_administration",
    "marxism",
    "mao_zedong_thought",
    "education_science",
    "teacher_qualification",
    "high_school_politics",
    "high_school_geography",
    "middle_school_politics",
    "middle_school_geography",
    "modern_chinese_history",
    "ideological_and_moral_cultivation",
    "logic",
    "law",
    "chinese_language_and_literature",
    "art_studies",
    "professional_tour_guide",
    "legal_professional",
    "high_school_chinese",
    "high_school_history",
    "middle_school_history",
    "civil_servant",
    "sports_science",
    "plant_protection",
    "basic_medicine",
    "clinical_medicine",
    "urban_and_rural_planner",
    "accountant",
    "fire_engineer",
    "environmental_impact_assessment_engineer",
    "tax_accountant",
    "physician",
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
ceval-{id_name}-0shot:
  id: ceval-{id_name}.val.ab-v1
  metrics: [accuracy]
ceval-{id_name}.val.ab-v1:
  class: evals.elsuite.dataset_specific.ceval:CEVAL
  args:
    dataset: hf://ceval/ceval-exam?name={file_name}&split=val
    few_shot: hf://ceval/ceval-exam?name={file_name}&split=dev
    instructions: 以下是中国关于{file_name}考试的单项选择题，请选出其中的正确答案。
    no_MC_prompt: True
    num_few_shot: 0

ceval-{id_name}-5shot:
  id: ceval-{id_name}.val.ab-v2
  metrics: [accuracy]
ceval-{id_name}.val.ab-v2:
  class: evals.elsuite.dataset_specific.ceval:CEVAL
  args:
    dataset: hf://ceval/ceval-exam?name={file_name}&split=val
    few_shot: hf://ceval/ceval-exam?name={file_name}&split=dev
    instructions: 以下是中国关于{file_name}考试的单项选择题，请选出其中的正确答案。
    num_few_shot: 5
    no_MC_prompt: True

ceval-{id_name}-CoT:
  id: ceval-{id_name}.val.ab-v3
  metrics: [accuracy]
ceval-{id_name}.val.ab-v3:
  class: evals.elsuite.dataset_specific.ceval:CEVAL
  args:
    dataset: hf://ceval/ceval-exam?name={file_name}&split=val
    few_shot: hf://ceval/ceval-exam?name={file_name}&split=dev
    instructions: 以下是中国关于{file_name}考试的单项选择题，请选出其中的正确答案。
    no_MC_prompt: True
    CoT: True
    num_few_shot: 5
    Chinese: True
"""
    eval_yaml += cur
with open(os.path.join(registry_path, "evals", "ceval.yaml"), "w") as f:
    f.write(eval_yaml)
