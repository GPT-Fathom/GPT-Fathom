from evals.prompt.base import chat_prompt_to_text_prompt, text_prompt_to_chat_prompt

prompt1 = [
    {
        "role": "system",
        "content": "Follow the given instructions and examples, answer the question.",
    },
    {"role": "user", "content": "Hello World!"},
]

text_prompt1 = chat_prompt_to_text_prompt(prompt1)
print(text_prompt1)


prompt2 = [
    {
        "role": "system",
        "content": "Follow the given instructions and examples, answer the question.",
    },
    {"role": "system", "name": "example_user", "content": "1 + 1 = "},
    {"role": "system", "name": "example_assistant", "content": "12"},
    {"role": "system", "content": "predict the next word"},
    {"role": "user", "content": "Hello World!"},
]
text_prompt2 = chat_prompt_to_text_prompt(prompt2)
print(text_prompt2)
