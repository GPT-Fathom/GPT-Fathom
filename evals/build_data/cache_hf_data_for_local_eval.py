import os
from datasets import load_dataset

registry_path = os.path.join("..", "registry")

# MMLU
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
for item in name:
    path = os.path.join(registry_path, "cached_hf_data", "mmlu", item, "test")
    if not os.path.exists(path):
        os.makedirs(path)
    dataset = load_dataset("hendrycks_test", name=item, split="test")
    dataset.save_to_disk(path)

    path = os.path.join(registry_path, "cached_hf_data", "mmlu", item, "fewshot")
    if not os.path.exists(path):
        os.makedirs(path)
    dataset = load_dataset("hendrycks_test", name=item, split="dev")
    dataset.save_to_disk(path)

# C-Eval
# name = [
#     "computer_network",
#     "operating_system",
#     "computer_architecture",
#     "college_programming",
#     "college_physics",
#     "college_chemistry",
#     "advanced_mathematics",
#     "probability_and_statistics",
#     "discrete_mathematics",
#     "electrical_engineer",
#     "metrology_engineer",
#     "high_school_mathematics",
#     "high_school_physics",
#     "high_school_chemistry",
#     "high_school_biology",
#     "middle_school_mathematics",
#     "middle_school_biology",
#     "middle_school_physics",
#     "middle_school_chemistry",
#     "veterinary_medicine",
#     "college_economics",
#     "business_administration",
#     "marxism",
#     "mao_zedong_thought",
#     "education_science",
#     "teacher_qualification",
#     "high_school_politics",
#     "high_school_geography",
#     "middle_school_politics",
#     "middle_school_geography",
#     "modern_chinese_history",
#     "ideological_and_moral_cultivation",
#     "logic",
#     "law",
#     "chinese_language_and_literature",
#     "art_studies",
#     "professional_tour_guide",
#     "legal_professional",
#     "high_school_chinese",
#     "high_school_history",
#     "middle_school_history",
#     "civil_servant",
#     "sports_science",
#     "plant_protection",
#     "basic_medicine",
#     "clinical_medicine",
#     "urban_and_rural_planner",
#     "accountant",
#     "fire_engineer",
#     "environmental_impact_assessment_engineer",
#     "tax_accountant",
#     "physician",
# ]
# for item in name:
#     path = os.path.join(registry_path, "cached_hf_data", "ceval", item, "test")
#     if not os.path.exists(path):
#         os.makedirs(path)
#     dataset = load_dataset("ceval/ceval-exam", name=item, split="val")
#     dataset.save_to_disk(path)
#     path = os.path.join(registry_path, "cached_hf_data", "ceval", item, "fewshot")
#     if not os.path.exists(path):
#         os.makedirs(path)
#     dataset = load_dataset("ceval/ceval-exam", name=item, split="dev")
#     dataset.save_to_disk(path)

# # Hellaswag
# path = os.path.join(registry_path, "cached_hf_data", "hellaswag", "validation")
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset("hellaswag", name=item, split="validation")
# dataset.save_to_disk(path)

# path = os.path.join(registry_path, "cached_hf_data", "hellaswag", "fewshot")
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset("hellaswag", name=item, split="train")
# dataset.save_to_disk(path)


# # coding
# dataset = load_dataset("openai_humaneval", split="test")
# path = os.path.join(registry_path, "cached_hf_data", "humaneval", "test")
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset.save_to_disk(path)

# dataset = load_dataset("mbpp", "sanitized", split="prompt")
# path = os.path.join(registry_path, "cached_hf_data", "mbpp", "fewshot")
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset.save_to_disk(path)

# dataset = load_dataset("mbpp", "sanitized", split="test")
# path = os.path.join(registry_path, "cached_hf_data", "mbpp", "test")
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset.save_to_disk(path)

# # Winogrande
# dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
# path = os.path.join(registry_path, "cached_hf_data", "winogrand", "test")
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset.save_to_disk(path)


# dataset = load_dataset("winogrande", "winogrande_xl", split="train")
# path = os.path.join(registry_path, "cached_hf_data", "winogrand", "fewshot")
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset.save_to_disk(path)
