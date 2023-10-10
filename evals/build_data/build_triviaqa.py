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
os.makedirs(os.path.join(registry_path, "data", "triviaqa"), exist_ok=True)

file = load_dataset("trivia_qa", "rc.nocontext", split="validation")
data = []
for row in file:
    content = create_chat_prompt(row["question"])
    pair = {
        "input": content,
        "ideal": row["answer"]["aliases"] + row["answer"]["normalized_aliases"],
    }
    data.append(pair)

file_path = os.path.join(registry_path, "data/triviaqa/samples.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

# Build the few shot samples for languages
registry_path = os.path.join("..", "registry")
os.makedirs(os.path.join(registry_path, "data", "triviaqa"), exist_ok=True)

file = load_dataset("trivia_qa", "rc.nocontext", split="train")
data = []
for row in file:
    content = create_fewshot_prompt(row["question"], row["answer"]["aliases"][0])
    pair = {"sample": content}
    data.append(pair)

data = data[:64]
file_path = os.path.join(registry_path, "data/triviaqa/fewshot.jsonl")
print(file_path)
with open(file_path, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")
