import json
import os
import random
from typing import Optional
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.utils import num_tokens_from_messages
from evals.formatting import make_abc
from evals.record import RecorderBase, record_sampling
from evals.registry import n_ctx_from_model_name



class BBH(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        task_name: str = None,
        samples_jsonl: str = False,
        CoT: bool = False,
        *args,
        instructions: Optional[str] = "",
        temerpature: float = 0.0,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert (
            len(completion_fns) == 1
        ), "MultipleChoice only supports one completion fn"
        self.samples_jsonl = samples_jsonl
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../..",
            "registry",
            "prompt",
            "BBH",
            f"{task_name}.txt",
        )
        with open(path, "r") as file:
            if CoT:
                CoT_prompt = file.read().split("-----")[1]
            else:
                data = file.read().split("-----")[1].strip().split("\n\n")
                for i in range(len(data)):
                    if "So the answer is " in data[i]:
                        start = data[i].find("Let's think step by step.")
                        end = data[i].find("So the answer is ") + len(
                            ("So the answer is ")
                        )

                        data[i] = data[i][:start] + "The answer is " + data[i][end:]

                CoT_prompt = "\n\n".join(data)
        self.instructions = instructions
        self.few_shot_prompt = CoT_prompt
        self.temerpature = temerpature
        self.CoT = CoT
        self.OUT_OF_WINDOW = False

    def pre_process(self, sample):
        for i in range(3, -1, -1):
            few_shot_prompt = self.few_shot_prompt
            few_shot_prompt = few_shot_prompt.split("\n\nQ: ")
            few_shot_prompt = "\n\nQ: ".join(few_shot_prompt[: i + 1])

            prompt = few_shot_prompt + "\n\nQ: " + sample["input"] + "\nA: "
            if self.CoT:
                prompt += "Let's think step by step. "

            if num_tokens_from_messages(
                [prompt], self.completion_fn.model
            ) + 50 + 400 <= n_ctx_from_model_name(self.completion_fn.model):
                break
            self.OUT_OF_WINDOW = True

        return prompt, sample["ideal"]

    def post_process(self, sampled):
        if self.CoT:
            if "So the answer is " in sampled:
                sampled = sampled.split("So the answer is ")[1]
        else:
            if "The answer is " in sampled:
                sampled = sampled.split("The answer is ")[1]

        return sampled.strip()

    def eval_sample(self, sample, rng):
        prompt, correct_answer = self.pre_process(sample)

        result = self.completion_fn(
            prompt=prompt,
            temperature=self.temerpature,
            max_tokens=400,
        )

        sampled = result.get_completions()[0]
        sampled = self.post_process(sampled)

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
            temperature=self.temerpature,
            max_tokens=400,
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
            "OUT OF WINDOW": self.OUT_OF_WINDOW,
        }
