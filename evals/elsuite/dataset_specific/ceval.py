import os
import random
from typing import Optional
from urllib.parse import parse_qs, urlparse
from evals.record import record_sampling

from datasets import load_dataset, load_from_disk
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.utils import num_tokens_from_messages
from evals.formatting import make_abc
from evals.record import RecorderBase
from evals.registry import n_ctx_from_model_name


def parse_option(sampled):
    import re

    match = re.search(r"\(([A-Z])\)", sampled)
    if match:
        letter = match.group(1)
        return letter
    else:
        return sampled


class Sample(BaseModel):
    question: str
    answers: list[str]
    label: int
    CoT: str = None


def get_dataset(local, url: str) -> list[Sample]:
    parsed = urlparse(url)
    if parsed.scheme == "hf":
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}

        path = "ceval/ceval-exam"
        if not local:
            dataset = load_dataset(path, **query)
        else:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../..",
                "registry",
                "cached_hf_data",
                "ceval",
                query["name"],
                "test" if query["split"] == "val" else "fewshot",
            )
            dataset = load_from_disk(path)
        return [
            Sample(
                question=sample["question"],
                answers=[sample["A"], sample["B"], sample["C"], sample["D"]],
                label=ord(sample["answer"]) - ord("A"),
                CoT=sample["explanation"],
            )
            for sample in dataset
        ]


class CEVAL(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str = None,
        num_few_shot: int = 0,
        few_shot: str = None,
        no_MC_prompt: bool = False,
        CoT: bool = False,
        Chinese: bool = True,
        max_tokens: int = 256,
        *args,
        temerpature: float = 0.0,
        instructions: Optional[str] = "",
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert (
            len(completion_fns) == 1
        ), "MultipleChoice only supports one completion fn"
        self.dataset = dataset
        self.instructions = instructions
        self.few_shot = num_few_shot
        self.few_shot_prompt = ""
        self.no_MC_prompt = no_MC_prompt
        self.temerpature = temerpature
        self.max_tokens = max_tokens
        self.Chinese = Chinese
        self.CoT = CoT
        if self.few_shot > 0:
            self.example = get_dataset(self.local_dataset, few_shot)
        self.OUT_OF_WINDOW = False

    def make_few_shot_prompt(self, num_shot):
        few_shot_prompt = ""
        rng = random.Random(47)
        for i in range((num_shot)):
            options, correct_answer, ans_ctx = make_abc(
                answers=self.example[i].answers,
                correct_idx=self.example[i].label,
                rng=rng,
            )
            few_shot_prompt += "Q: " + self.example[i].question + "\n\n" + options  #
            if not self.no_MC_prompt:
                few_shot_prompt += "\n请只输出正确答案对应的字母编号\n"
            else:
                few_shot_prompt += "\n"
            if self.CoT:
                few_shot_prompt += (
                    "A: "
                    + "让我们一步一步思考。"
                    + self.example[i].CoT
                    + "所以答案是 "
                    + ans_ctx
                    + "\n\n"
                )
            else:
                few_shot_prompt += "A: " + ans_ctx + "\n\n"

        return few_shot_prompt
    def pre_process(self, sample): 
        options, correct_answer, _ = make_abc(
            answers=sample.answers,
            correct_idx=sample.label,
            rng=random.Random(47),
        )
        for i in range(self.few_shot, -1, -1):
            few_shot_prompt = self.make_few_shot_prompt(i)
            prompt = (
                self.instructions
                + "\n\n"
                + few_shot_prompt
                + "Q: "
                + sample.question
                + "\n\n"
                + options
            )
            if not self.no_MC_prompt:
                prompt += "\n请只输出正确答案对应的字母编号\nA: "
            else:
                prompt += "\nA: "
            if self.CoT:
                prompt += "让我们一步一步思考。"

            if num_tokens_from_messages(
                [prompt], self.completion_fn.model
            ) + 50 + self.max_tokens <= n_ctx_from_model_name(self.completion_fn.model):
                break
            self.OUT_OF_WINDOW = True
        return prompt, correct_answer


    def eval_sample(self, sample, rng):
        assert isinstance(sample, Sample)

        prompt, correct_answer = self.pre_process(sample)

        result = self.completion_fn(
            prompt=prompt,
            temperature=self.temerpature,
            max_tokens=self.max_tokens if self.CoT else 10,
        )
        sampled = result.get_completions()[0]
        if "所以答案是 " in sampled:
            sampled = sampled.split("所以答案是 ")[1]
        sampled = parse_option(sampled)

        evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled.lstrip(),
            expected=correct_answer,
        )

    def eval_sample_batch(self, recorder, samples):
        data = [self.pre_process(item)[0] for item in samples]
        ideal = [self.pre_process(item)[1] for item in samples]
        results = self.completion_fn(inputs=data, temperature=self.temerpature, max_tokens=50)
        processed_res = [(item) for item in results]
        for i in range(len(processed_res)):
            correct_answer = ideal[i]
            sampled = processed_res[i]
            if "所以答案是 " in sampled:
                sampled = sampled.split("所以答案是 ")[1]
            sampled = parse_option(sampled)
            id = str(i)
            with recorder.as_default_recorder(id):
                record_sampling(prompt=data[i], sampled=results[i])
                evals.record_and_check_match(
                    prompt=data[i],
                    sampled=sampled.lstrip(),
                    expected=correct_answer,
                )

    def run(self, recorder: RecorderBase):
        samples = get_dataset(self.local_dataset, self.dataset)

        if self.eval_in_batch == False:
            self.eval_all_samples(recorder, samples)
        else:
            self.eval_sample_batch(recorder, samples)

        print("OUT OF WINDOW: ", self.OUT_OF_WINDOW)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
            "OUT OF WINDOW": self.OUT_OF_WINDOW,
        }
