"""Build the relevant file for hellaswag dataset"""
import json
import os

from datasets import load_dataset


def create_chat_prompt(text):
    return [
        {
            "role": "system",
            "content": "Complete the description with an appropriate ending, please answer with the letter of the correct answer\n",
        },
        {"role": "user", "content": text},
    ]


def create_fewshot_prompt(question, answer):
    return [
        {"role": "system", "name": "example_user", "content": question},
        {"role": "system", "name": "example_assistant", "content": answer},
    ]


# Build the test samples
# samples = load_dataset("hellaswag",  split="validation")
# registry_path = os.path.join("..", "registry")
# os.makedirs(os.path.join(registry_path, "data", "hellaswag"), exist_ok=True)

# data = []
# samples = [(item["ctx"], item["endings"], item["label"]) for item in samples]

# for item in samples[15:]:
#     words = item.split()
#     context = " ".join(words[:-1])
#     target = words[-1]
#     content = create_chat_prompt(context)
#     pair = {"input": content, "ideal": target}
#     data.append(pair)
# file_path = os.path.join(registry_path, "data/hellaswag/samples.jsonl")
# print(file_path)
# with open(file_path, 'w') as f:
#     for entry in data:
#         json.dump(entry, f)
#         f.write('\n')


# # Build the few shot samples

# import os
# registry_path = os.path.join("..", "registry")
# os.makedirs(os.path.join(registry_path, "data", "hellaswag"), exist_ok=True)

# data = []
# for item in samples[:15]:
#     words = item.split()
#     context = " ".join(words[:-1])
#     target = words[-1]
#     content = create_fewshot_prompt(context, target)
#     pair = {"sample": content}
#     data.append(pair)
# file_path = os.path.join(registry_path, "data/hellaswag/fewshot.jsonl")
# print(file_path)
# with open(file_path, 'w') as f:
#     for entry in data:
#         json.dump(entry, f)
#         f.write('\n')
