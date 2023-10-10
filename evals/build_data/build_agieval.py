import json

from datasets import load_dataset

MC_ls = [
    "sat-math",
    "sat-en",
    "aqua-rat",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "logiqa-en",
    "sat-en-without-passage",
]
Cloze_ls = ["math"]
import os

registry_path = os.path.join("..", "registry")


eval_yaml = ""
for item in MC_ls:
    id_name = item
    # print(file_name, id_name)
    cur = f"""
agieval-{id_name}-0shot:
  id: agieval-{id_name}.val.ab-v1
  metrics: [accuracy]
agieval-{id_name}.val.ab-v1:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    no_MC_prompt: True
    num_few_shot: 0
    max_tokens: 50
    type: MC

agieval-{id_name}-5shot:
  id: agieval-{id_name}.val.ab-v2
  metrics: [accuracy]
agieval-{id_name}.val.ab-v2:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    max_tokens: 50
    type: MC

agieval-{id_name}-CoT:
  id: agieval-{id_name}.val.ab-v3
  metrics: [accuracy]
agieval-{id_name}.val.ab-v3:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    CoT: True
    type: MC

"""
    eval_yaml += cur


for item in Cloze_ls:
    id_name = item
    # print(file_name, id_name)
    cur = f"""
agieval-{id_name}-0shot:
  id: agieval-{id_name}.val.ab-v1
  metrics: [accuracy]
agieval-{id_name}.val.ab-v1:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    no_MC_prompt: True
    num_few_shot: 0
    max_tokens: 50
    type: Cloze

agieval-{id_name}-5shot:
  id: agieval-{id_name}.val.ab-v2
  metrics: [accuracy]
agieval-{id_name}.val.ab-v2:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    max_tokens: 50
    type: Cloze

agieval-{id_name}-CoT:
  id: agieval-{id_name}.val.ab-v3
  metrics: [accuracy]
agieval-{id_name}.val.ab-v3:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    CoT: True
    type: Cloze
    
"""
    eval_yaml += cur

###############  Now for Chinese
MC_ls = [
    "gaokao-chinese",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-biology",
    "gaokao-chemistry",
    "gaokao-english",
    "logiqa-zh",
    "gaokao-mathqa",
]
Cloze_ls = ["gaokao-mathcloze"]
IMC_ls = ["jec-qa-kd", "jec-qa-ca", "gaokao-physics"]
import os

registry_path = os.path.join("..", "registry")


for item in MC_ls:
    id_name = item
    # print(file_name, id_name)
    cur = f"""
agieval-{id_name}-0shot:
  id: agieval-{id_name}.val.ab-v1
  metrics: [accuracy]
agieval-{id_name}.val.ab-v1:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    no_MC_prompt: True
    num_few_shot: 0
    type: MC
    max_tokens: 50
    Cn: True

agieval-{id_name}-5shot:
  id: agieval-{id_name}.val.ab-v2
  metrics: [accuracy]
agieval-{id_name}.val.ab-v2:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    max_tokens: 50
    Cn: True
    type: MC

agieval-{id_name}-CoT:
  id: agieval-{id_name}.val.ab-v3
  metrics: [accuracy]
agieval-{id_name}.val.ab-v3:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    Cn: True
    CoT: True
    type: MC

"""
    eval_yaml += cur

for item in IMC_ls:
    id_name = item
    # print(file_name, id_name)
    cur = f"""
agieval-{id_name}-0shot:
  id: agieval-{id_name}.val.ab-v1
  metrics: [accuracy]
agieval-{id_name}.val.ab-v1:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    no_MC_prompt: True
    Cn: True
    max_tokens: 50
    num_few_shot: 0
    type: IMC

agieval-{id_name}-5shot:
  id: agieval-{id_name}.val.ab-v2
  metrics: [accuracy]
agieval-{id_name}.val.ab-v2:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    max_tokens: 50
    Cn: True
    type: IMC

agieval-{id_name}-CoT:
  id: agieval-{id_name}.val.ab-v3
  metrics: [accuracy]
agieval-{id_name}.val.ab-v3:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    Cn: True
    CoT: True
    type: IMC
    
"""
    eval_yaml += cur


for item in Cloze_ls:
    id_name = item
    # print(file_name, id_name)
    cur = f"""
agieval-{id_name}-0shot:
  id: agieval-{id_name}.val.ab-v1
  metrics: [accuracy]
agieval-{id_name}.val.ab-v1:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    no_MC_prompt: True
    num_few_shot: 0
    max_tokens: 50
    type: Cloze
    Cn: True

agieval-{id_name}-5shot:
  id: agieval-{id_name}.val.ab-v2
  metrics: [accuracy]
agieval-{id_name}.val.ab-v2:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    max_tokens: 50
    type: Cloze
    Cn: True

agieval-{id_name}-CoT:
  id: agieval-{id_name}.val.ab-v3
  metrics: [accuracy]
agieval-{id_name}.val.ab-v3:
  class: evals.elsuite.dataset_specific.agieval:AGIEVAL
  args:
    sample_jsonl: agieval/{id_name}.jsonl
    task: {id_name}
    few_shot_jsonl: agieval/few_shot_prompts.csv
    num_few_shot: 5
    Cn: True
    CoT: True
    type: Cloze
    
"""
    eval_yaml += cur


with open(os.path.join(registry_path, "evals", "agieval.yaml"), "w") as f:
    f.write(eval_yaml)
