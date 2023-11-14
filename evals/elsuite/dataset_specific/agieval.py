import json
import os
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
from datasets import load_dataset

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.utils import num_tokens_from_messages
from evals.formatting import make_abc
from evals.record import RecorderBase, record_sampling
from evals.registry import n_ctx_from_model_name
from evals.utils.math_util import (
    clean_numbers,
    is_equiv,
    last_boxed_only,
    last_boxed_only_string,
    remove_boxed,
)


class AGIEVAL(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        num_few_shot: int = 0,
        task: str = None,
        sample_jsonl: str = None,
        few_shot_jsonl: str = None,
        max_tokens: int = 512,
        type: str = None,
        CoT: bool = False,
        Cn: bool = False,
        *args,
        instructions: Optional[str] = "",
        temperature: float = 0.0,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert (
            len(completion_fns) == 1
        ), "MultipleChoice only supports one completion fn"
        self.few_shot = num_few_shot
        self.max_tokens = max_tokens
        self.sample_jsonl = sample_jsonl
        self.OUT_OF_WINDOW = False
        self.type = type
        self.cn = Cn
        self.CoT = CoT
        self.temperature = temperature
        self.few_shot_samples = []
        if self.type == "MC":
            self.instructions = (
                "Follow the given samples and answer the following multiple choice question."
                if not self.cn
                else "回答下列选择题"
            )
        elif self.type == "IMC":
            self.instructions = (
                "Follow the given samples and answer the following multiple select question."
                if not self.cn
                else "回答下列多选题"
            )
        else:
            self.instructions = (
                "Follow the given samples and answer the following cloze question."
                if not self.cn
                else "回答下列填空题"
            )
        if self.few_shot > 0:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../..",
                "registry",
                "data",
                few_shot_jsonl,
            )
            data = pd.read_csv(path)
            context_row = [0, 1, 3, 5, 7, 9]
            explanation_row = [0, 2, 4, 6, 8, 10]
            raw_prompts_context = pd.read_csv(
                path,
                header=0,
                skiprows=lambda x: x not in context_row,
                keep_default_na=False,
            )
            raw_prompts_explanation = pd.read_csv(
                path,
                header=0,
                skiprows=lambda x: x not in explanation_row,
                keep_default_na=False,
            ).replace(r"\n\n", "\n", regex=True)
            contexts = []
            import ast

            if task == "sat-en-without-passage":
                task = "sat-en"
            for line in list(raw_prompts_context[task]):
                if line:
                    # print(line)
                    contexts.append(ast.literal_eval(line))
            explanations = [exp for exp in raw_prompts_explanation[task] if exp]
            self.few_shot = len(explanations)
            for i in range(len(explanations)):
                sample = contexts[i]
                self.few_shot_samples.append(
                    {
                        "passage": sample.get("passage", ""),
                        "question": sample.get("question", ""),
                        "answer": sample.get("answer", ""),
                        "options": sample.get("options", ""),
                        "label": sample.get("label", ""),
                        "explanation": explanations[i],
                    }
                )
            self.few_shot_samples = self.few_shot_samples[: self.few_shot]

    def make_few_shot_prompt(self, num_shot):

        few_shot_prompt = ""
        if not self.cn:
            if self.type == "MC":

                for i in range((num_shot)):
                    if self.few_shot_samples[i]["passage"] not in [None, ""]:
                        few_shot_prompt += (
                            "Passage: " + self.few_shot_samples[i]["passage"] + "\n"
                        )
                    few_shot_prompt += (
                        "Q: " + self.few_shot_samples[i]["question"] + "\n"
                    )
                    for item in self.few_shot_samples[i]["options"]:
                        few_shot_prompt += item + "\n"
                    label = self.few_shot_samples[i]["label"]
                    if self.CoT:
                        few_shot_prompt += (
                            "A: Let's think step by step. "
                            + self.few_shot_samples[i]["explanation"]
                            + "\n"
                        )
                        few_shot_prompt += "So the answer is (" + label + ") "
                        few_shot_prompt += (
                            self.few_shot_samples[i]["options"][ord(label) - ord("A")]
                            + "\n\n"
                        )
                    else:
                        few_shot_prompt += (
                            "A: The answer is ("
                            + label
                            + ") "
                            + self.few_shot_samples[i]["options"][ord(label) - ord("A")]
                            + "\n\n"
                        )

            elif self.type == "Cloze":
                for i in range((num_shot)):
                    if self.few_shot_samples[i]["passage"] not in [None, ""]:
                        few_shot_prompt += (
                            "Passage: " + self.few_shot_samples[i]["passage"] + "\n"
                        )
                    few_shot_prompt += (
                        "Q: " + self.few_shot_samples[i]["question"] + "\n"
                    )
                    if self.CoT:
                        few_shot_prompt += (
                            "A: Let's think step by step. "
                            + self.few_shot_samples[i]["explanation"]
                            + "\n"
                        )
                        few_shot_prompt += (
                            "So the answer is "
                            + self.few_shot_samples[i]["answer"]
                            + "\n\n"
                        )
                    else:
                        few_shot_prompt += (
                            "A: " + self.few_shot_samples[i]["answer"] + "\n\n"
                        )
        else:

            if self.type == "MC":
                for i in range((num_shot)):
                    if self.few_shot_samples[i]["passage"] not in [None, ""]:
                        few_shot_prompt += (
                            "文章: " + self.few_shot_samples[i]["passage"] + "\n"
                        )
                    few_shot_prompt += (
                        "问题: " + self.few_shot_samples[i]["question"] + "\n"
                    )
                    for item in self.few_shot_samples[i]["options"]:
                        few_shot_prompt += item + "\n"
                    label = self.few_shot_samples[i]["label"]
                    if self.CoT:
                        few_shot_prompt += (
                            "回答: 让我们一步步思考。 "
                            + self.few_shot_samples[i]["explanation"]
                            + "\n"
                        )
                        few_shot_prompt += "所以答案是" + "(" + label + ") "
                        few_shot_prompt += (
                            self.few_shot_samples[i]["options"][ord(label) - ord("A")]
                            + "\n\n"
                        )
                    else:
                        few_shot_prompt += (
                            "回答：("
                            + label
                            + ") "
                            + self.few_shot_samples[i]["options"][ord(label) - ord("A")]
                            + "\n\n"
                        )

            elif self.type == "IMC":
                for i in range((num_shot)):
                    if self.few_shot_samples[i]["passage"] not in [None, ""]:
                        few_shot_prompt += (
                            "文章: " + self.few_shot_samples[i]["passage"] + "\n"
                        )
                    few_shot_prompt += (
                        "问题: " + self.few_shot_samples[i]["question"] + "\n"
                    )
                    for item in self.few_shot_samples[i]["options"]:
                        few_shot_prompt += item + "\n"
                    if isinstance(self.few_shot_samples[i]["label"], str):
                        self.few_shot_samples[i]["label"] = [
                            self.few_shot_samples[i]["label"]
                        ]

                    if self.CoT:
                        few_shot_prompt += (
                            "回答: 让我们一步步思考。"
                            + self.few_shot_samples[i]["explanation"]
                            + "\n"
                        )
                        few_shot_prompt += "所以答案是\n"
                        for item in self.few_shot_samples[i]["label"]:
                            few_shot_prompt += (
                                f"({item}) "
                                + self.few_shot_samples[i]["options"][
                                    ord(item) - ord("A")
                                ]
                                + "\n"
                            )
                        few_shot_prompt += "\n"
                    else:
                        few_shot_prompt += "回答："
                        for item in self.few_shot_samples[i]["label"]:
                            few_shot_prompt += (
                                f"({item}) "
                                + self.few_shot_samples[i]["options"][
                                    ord(item) - ord("A")
                                ]
                                + "\n"
                            )
                        few_shot_prompt += "\n"

            elif self.type == "Cloze":
                for i in range((num_shot)):
                    if self.few_shot_samples[i]["passage"] not in [None, ""]:
                        few_shot_prompt += (
                            "文章: " + self.few_shot_samples[i]["passage"] + "\n"
                        )
                    few_shot_prompt += (
                        "问题: " + self.few_shot_samples[i]["question"] + "\n"
                    )
                    if self.CoT:
                        few_shot_prompt += (
                            "回答: 让我们一步步思考。"
                            + self.few_shot_samples[i]["explanation"]
                            + "\n"
                        )
                        few_shot_prompt += (
                            "所以答案是" + self.few_shot_samples[i]["answer"] + "\n\n"
                        )
                    else:
                        few_shot_prompt += (
                            "回答：" + self.few_shot_samples[i]["answer"] + "\n\n"
                        )
        return few_shot_prompt

    def pre_process(self, sample):
        for i in range(self.few_shot, -1, -1):
            few_shot_prompt = self.make_few_shot_prompt(i)
            prompt = self.instructions + "\n\n" + few_shot_prompt + "\n\n"

            ##### Process data pair into text

            if not self.cn:
                if self.type == "MC":
                    if sample["passage"] not in [None, ""]:
                        prompt += "Passage: " + sample["passage"] + "\n"
                    prompt += "Q: " + sample["question"] + "\n"
                    for item in sample["options"]:
                        prompt += item + "\n"
                    if self.CoT:
                        prompt += "A: Let's think step by step. "
                    else:
                        prompt += "A: "

                elif self.type == "IMC":
                    if sample["passage"] not in [None, ""]:
                        prompt += "Passage: " + sample["passage"] + "\n"
                    prompt += "Q: " + sample["question"] + "\n"
                    for item in sample["options"]:
                        prompt += item + "\n"
                    if isinstance(sample["label"], str):
                        sample["label"] = [sample["label"]]

                    if self.CoT:
                        prompt += "A: Let's think step by step. "
                    else:
                        prompt += "A: "

                elif self.type == "Cloze":
                    if sample["passage"] not in [None, ""]:
                        prompt += "Passage: " + sample["passage"] + "\n"
                    prompt += "Q: " + sample["question"] + "\n"
                    if self.CoT:
                        prompt += "A: Let's think step by step. "
                    else:
                        prompt += "A: "
            else:

                if self.type == "MC":
                    if sample["passage"] not in [None, ""]:
                        prompt += "文章: " + sample["passage"] + "\n"
                    prompt += "问题: " + sample["question"] + "\n"
                    for item in sample["options"]:
                        prompt += item + "\n"
                    if self.CoT:
                        prompt += "回答: 让我们一步步思考。 "
                    else:
                        prompt += "回答："

                elif self.type == "IMC":
                    if sample["passage"] not in [None, ""]:
                        prompt += "文章: " + sample["passage"] + "\n"
                    prompt += "问题: " + sample["question"] + "\n"
                    for item in sample["options"]:
                        prompt += item + "\n"
                    if isinstance(sample["label"], str):
                        sample["label"] = [sample["label"]]

                    if self.CoT:
                        prompt += "回答: 让我们一步步思考。"
                    else:
                        prompt += "回答："

                elif self.type == "Cloze":
                    if sample["passage"] not in [None, ""]:
                        prompt += "文章: " + sample["passage"] + "\n"
                    prompt += "问题: " + sample["question"] + "\n"
                    if self.CoT:
                        prompt += "回答: 让我们一步步思考。"
                    else:
                        prompt += "回答："

            if num_tokens_from_messages(
                [prompt], self.completion_fn.model
            ) + 50 + self.max_tokens <= n_ctx_from_model_name(self.completion_fn.model):
                break
            self.OUT_OF_WINDOW = True
            if i == 0:
                prompt = "Once upon a time"
        return prompt, sample

    def post_process(self, sampled, sample):
        if self.CoT:
            if "So the answer is " in sampled:
                sampled = sampled.split("So the answer is ")[1]
            elif "所以答案是" in sampled:
                sampled = sampled.split("所以答案是")[1]
        if self.type == "MC":
            for i in range(26):
                if f"({chr(i+65)})" in sampled:
                    sampled = chr(i + 65)
                    break

            correct_answer = sample["label"]
        elif self.type == "Cloze":

            correct_answer = sample["answer"]
        else:
            if not isinstance(sample["label"], list):
                sample["label"] = [sample["label"]]
            correct_answer = ",".join(sample["label"])
            choice = []
            for i in range(26):
                if f"({chr(i+65)})" in sampled:
                    choice.append(chr(i + 65))
            sampled = ",".join(choice)

        return sampled, correct_answer

    def eval_sample(self, sample, rng):

        prompt, sample = self.pre_process(sample)

        result = self.completion_fn(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        sampled = result.get_completions()[0]
        sampled, correct_answer = self.post_process(sampled, sample)
        evals.record_and_check_match(
            prompt=prompt, sampled=sampled, expected=correct_answer
        )

    def eval_sample_batch(self, recorder, samples):
        data = [self.pre_process(item)[0] for item in samples]
        ideal = [self.pre_process(item)[1] for item in samples]
        results = self.completion_fn(
            inputs=data,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        processed_res, correct_answers = zip(
            *[self.post_process(results[i], ideal[i]) for i in range(len(results))]
        )

        for i in range(len(processed_res)):
            id = str(i)
            with recorder.as_default_recorder(id):
                record_sampling(prompt=data[i], sampled=results[i])
                evals.record_and_check_match(
                    prompt=data[i],
                    sampled=processed_res[i],
                    expected=correct_answers[i],
                )

    def run(self, recorder: RecorderBase):
        samples = evals.get_jsonl(self.sample_jsonl)
        if self.eval_in_batch == False:
            self.eval_all_samples(recorder, samples)
        else:
            self.eval_sample_batch(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
            "OUT OF WINDOW": self.OUT_OF_WINDOW,
        }
