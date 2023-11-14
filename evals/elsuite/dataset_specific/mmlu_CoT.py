import json
import os
import random
from typing import Optional
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset, load_from_disk
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.utils import num_tokens_from_messages
from evals.formatting import make_abc
from evals.record import RecorderBase, record_sampling
from evals.registry import n_ctx_from_model_name


class Sample(BaseModel):
    question: str
    answers: list[str]
    label: int


def parse_option(sampled):
    import re

    match = re.search(r"\(([A-Z])\)", sampled)
    if match:
        letter = match.group(1)
        return letter
    else:
        return sampled


def get_dataset(local, url: str) -> list[Sample]:
    parsed = urlparse(url)
    if parsed.scheme == "hf":
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}

        path = parsed.netloc
        if not local:
            dataset = load_dataset(path, **query)
        if path == "hendrycks_test":
            if local:
                path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../..",
                    "registry",
                    "cached_hf_data",
                    "mmlu",
                    query["name"],
                    "test" if query["split"] == "test" else "fewshot",
                )
                dataset = load_from_disk(path)
            return [
                Sample(
                    question=sample["question"],
                    answers=sample["choices"],
                    label=sample["answer"],
                )
                for sample in dataset
            ]
    raise ValueError(f"Unknown question dataset {url}")


class MC_CoT(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str = None,
        no_MC_prompt: bool = False,
        CoT: bool = False,
        task_name: str = None,
        temperature: float = 0.0,
        *args,
        instructions: Optional[str] = "",
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert (
            len(completion_fns) == 1
        ), "MultipleChoice only supports one completion fn"
        self.dataset = dataset
        self.temperature = temperature
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../..",
            "registry",
            "prompt",
            "mmlu",
            "mmlu-cot.json",
        )
        with open(path, "r") as file:
            CoT_prompt = json.load(file)
        self.instructions = CoT_prompt[task_name]
        self.few_shot_prompt = CoT_prompt[task_name]
        self.no_MC_prompt = no_MC_prompt
        self.CoT = CoT
        self.max_tokens = 200
        self.OUT_OF_WINDOW = False

    """
        Turn the sample into a prompt
    """

    def pre_process(self, sample):
        options, correct_answer, _ = make_abc(
            answers=sample.answers,
            correct_idx=sample.label,
            rng=random.Random(47),
        )
        for i in range(5, -1, -1):
            few_shot_prompt = self.few_shot_prompt
            few_shot_prompt = few_shot_prompt.split("\n\nQ: ")
            few_shot_prompt = "\n\nQ: ".join(few_shot_prompt[: i + 1])
            prompt = few_shot_prompt + "Q: " + sample.question + "\n\n" + options
            if not self.no_MC_prompt:
                prompt += "\nPlease answer with the letter of the correct answer.\nA: "
            else:
                prompt += "\nA: "
            if self.CoT:
                prompt += "Let's think step by step. "
            if num_tokens_from_messages(
                [prompt], self.completion_fn.model
            ) + 50 + self.max_tokens <= n_ctx_from_model_name(self.completion_fn.model):
                break
            self.OUT_OF_WINDOW = True

        return prompt, correct_answer

    def eval_sample(self, sample, rng):
        assert isinstance(sample, Sample)

        # print(prompt)
        prompt, correct_answer = self.pre_process(sample)
        result = self.completion_fn(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]
        if "The answer is " in sampled:
            sampled = sampled.split("The answer is ")[1]
        sampled = parse_option(sampled)

        evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=correct_answer,
        )

    def eval_sample_batch(self, recorder, samples):
        data = [self.pre_process(item)[0] for item in samples]
        ideal = [self.pre_process(item)[1] for item in samples]
        results = self.completion_fn(
            inputs=data,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        processed_res = [
            sampled.split("The answer is ")[1]
            if "The answer is " in sampled
            else sampled
            for sampled in results
        ]
        processed_res = [parse_option(sampled) for sampled in processed_res]
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
        samples = get_dataset(self.local_dataset, self.dataset)

        if self.eval_in_batch == False:
            self.eval_all_samples(recorder, samples)
        else:
            self.eval_sample_batch(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
            "OUT OF WINDOW": self.OUT_OF_WINDOW,
        }
