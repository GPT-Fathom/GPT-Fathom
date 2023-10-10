import os
import random
from typing import Optional
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset, load_from_disk
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.dataset_specific.lambada import CHAT_MODELS
from evals.elsuite.utils import num_tokens_from_messages
from evals.formatting import make_abc
from evals.record import RecorderBase, record_sampling
from evals.registry import n_ctx_from_model_name


class Sample(BaseModel):
    question: str
    answers: list[str]
    label: int


def get_dataset(local, url: str) -> list[Sample]:
    parsed = urlparse(url)
    if parsed.scheme == "hf":
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}

        path = parsed.netloc
        if not local:
            dataset = load_dataset(path, **query)
        if path == "hellaswag":
            if local:
                path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../..",
                    "registry",
                    "cached_hf_data",
                    "hellaswag",
                    "test" if query["split"] == "validation" else "fewshot",
                )
                dataset = load_from_disk(path)
            return [
                Sample(
                    question=sample["ctx"],
                    answers=sample["endings"],
                    label=int(sample["label"]),
                )
                for sample in dataset
            ]
        elif path == "hendrycks_test":
            if local:
                print(query["split"])
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
        elif path == "winogrande":
            if local:
                path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../..",
                    "registry",
                    "cached_hf_data",
                    "winogrande",
                    "test" if query["split"] == "validation" else "fewshot",
                )
                dataset = load_from_disk(path)
            return [
                Sample(
                    question=sample["sentence"],
                    answers=[sample["option1"], sample["option2"]],
                    label=int(sample["answer"]) - 1,
                )
                for sample in dataset
            ]
    raise ValueError(f"Unknown question dataset {url}")


def load_from_file(file: str) -> list[Sample]:
    data = evals.get_jsonl(file)
    return [
        Sample(
            question=sample["question"],
            answers=sample["choices"],
            label=sample["answer"],
        )
        for sample in data
    ]


class MultipleChoice(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str = None,
        num_few_shot: int = 0,
        few_shot: str = None,
        sample_jsonl: str = None,
        few_shot_jsonl: str = None,
        no_MC_prompt: bool = False,
        CoT: bool = False,
        max_tokens: int = 10,
        temerpature: float = 0.0,
        *args,
        instructions: Optional[str] = "",
        example_q_alias: str = "Q: ",
        example_a_alias: str = "A: ",
        q_alias: str = "Q: ",
        a_alias: str = "A: ",
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
        self.sample_jsonl = sample_jsonl
        self.no_MC_prompt = no_MC_prompt
        self.temerpature = temerpature
        self.max_tokens = max_tokens
        self.prompt_template = [example_q_alias, example_a_alias, q_alias, a_alias]
        self.CoT = CoT
        if self.few_shot > 0:
            if few_shot_jsonl == None:
                self.example = get_dataset(self.local_dataset, few_shot)
            else:
                self.example = load_from_file(few_shot_jsonl)
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
            few_shot_prompt += self.prompt_template[0] + self.example[i].question + "\n\n" + options  #
            if not self.no_MC_prompt:
                few_shot_prompt += (
                    "\nPlease answer with the letter of the correct answer.\n"
                )
            else:
                few_shot_prompt += "\n"
            few_shot_prompt += self.prompt_template[1] + ans_ctx + "\n\n"

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
                + self.prompt_template[2]
                + sample.question
                + "\n\n"
                + options
            )
            if not self.no_MC_prompt:
                prompt += f"\nPlease answer with the letter of the correct answer.\n{self.prompt_template[3]}"
            else:
                prompt += f"\n{self.prompt_template[3]}"
            if self.CoT:
                prompt += "Let's think step by step. "

            if num_tokens_from_messages(
                [prompt], self.completion_fn.model
            ) + 50 + self.max_tokens <= n_ctx_from_model_name(self.completion_fn.model):
                break
            self.OUT_OF_WINDOW = True
        return prompt, correct_answer

    def post_process(self, sampled):
        options = ["(" + chr(ord("A") + idx) + ")" for idx in range(26)]
        ans = sampled
        for option in options:
            if option in sampled:
                ans = option[1]
                break
        if ans == sampled:
            sampled = sampled.lstrip()
            ans = sampled
        sampled = ans
        return sampled

    def eval_sample(self, sample, rng):

        prompt, correct_answer = self.pre_process(sample)

        result = self.completion_fn(
            prompt=prompt,
            temperature=self.temerpature,
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]
        evals.record_and_check_match(
            prompt=prompt,
            sampled=self.post_process(sampled),
            expected=correct_answer,
        )

    def eval_sample_batch(self, recorder, samples):
        data = [self.pre_process(item)[0] for item in samples]
        ideal = [self.pre_process(item)[1] for item in samples]
        results = self.completion_fn(
            inputs=data,
            temperature=self.temerpature,
            max_tokens=self.max_tokens,
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
        if self.sample_jsonl == None:
            samples = get_dataset(self.local_dataset, self.dataset)
        else:
            samples = load_from_file(self.sample_jsonl)

        if self.eval_in_batch == False:
            self.eval_all_samples(recorder, samples)
        else:
            self.eval_sample_batch(recorder, samples)

        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
            "OUT OF WINDOW": self.OUT_OF_WINDOW,
        }
