"""The eval script for lambada dataset"""
import os
from typing import Any

from datasets import load_dataset

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.prompt.base import chat_prompt_to_text_prompt, is_chat_prompt
from evals.record import RecorderBase, record_sampling
from evals.registry import CHAT_MODELS


# CHAT_MODELS = {
#     "gpt-3.5-turbo",
#     "gpt-3.5-turbo-0301",
#     "gpt-3.5",
#     "gpt-4",
#     "gpt-4-0314",
#     "gpt-4-32k",
#     "gpt-4-32k-0314",
# }


def extract_first_word(text):
    import re

    text = text.strip()
    if text.startswith("A:"):
        text = text[2:]
    pattern = r"\b[a-zA-Z]+\b"  # 匹配一个由字母和数字组成的单词
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return text


class Lambada(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        temerpature: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert (
            len(completion_fns) == 1
        ), "MultipleChoice only supports one completion fn"
        self.samples_jsonl = samples_jsonl
        self.num_few_shot = num_few_shot
        self.temerpature = temerpature
        if self.num_few_shot > 0:
            assert (
                few_shot_jsonl is not None
            ), "few shot requires few shot sample dataset"
            self.few_shot_jsonl = few_shot_jsonl
            self.few_shot = evals.get_jsonl(self.few_shot_jsonl)

    def prompt_to_fewshot(self, sample):
        prompt = sample["input"]
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]
        return prompt

    """
        Turn the sample into a prompt
    """

    def pre_process(self, sample):
        prompt = self.prompt_to_fewshot(sample)
        # Handle the '\n' relevant problems
        if self.completion_fns[0].model not in CHAT_MODELS:
            if self.num_few_shot == 0:
                prompt = chat_prompt_to_text_prompt(
                    prompt, Q_A_ENDING=False, new_line_end=False
                )
            else:
                prompt = chat_prompt_to_text_prompt(
                    prompt, Q_A_ENDING=True, new_line_end=False
                )
        return prompt, sample["ideal"]

    """
        Output post processing
    """

    def post_process(self, res):
        res = res.lstrip()
        res = extract_first_word(res)
        return res

    def eval_sample(self, sample: Any, *_):
        prompt, ideal = self.pre_process(sample)
        result = self.completion_fn(
            prompt=prompt,
            temperature=self.temerpature,
            max_tokens=15,
        )
        sampled = self.post_process(result.get_completions()[0])
        evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=ideal,
        )

    def eval_sample_batch(self, recorder, samples):
        data = [self.pre_process(item)[0] for item in samples]
        ideal = [self.pre_process(item)[1] for item in samples]
        results = self.completion_fn(
            inputs=data,
            temperature=self.temerpature,
            max_tokens=8,
        )
        processed_res = [self.post_process(item) for item in results]
        for i in range(len(processed_res)):
            id = str(i)
            with recorder.as_default_recorder(id):
                record_sampling(prompt=data[i], sampled=results[i])
                evals.record_and_check_match(
                    prompt=data[i],
                    sampled=processed_res[i],
                    expected=ideal[i],
                )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        if self.eval_in_batch == False:
            self.eval_all_samples(recorder, samples)
        else:
            self.eval_sample_batch(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
