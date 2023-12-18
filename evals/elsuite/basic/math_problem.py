import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

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


def check_match(sampled, ans, dataset):
    if dataset == "mgsm":
        ANS_RE = re.compile(r"(-?\d[\d,]*\.?\d*)(?![\d-])")
        matches = ANS_RE.findall(sampled)
        if matches:
            match_str = matches[-1]
            return match_str, ans
        else:
            return None, ans
    if dataset == "gsm8k":
        ANS_RE = re.compile(r"\d+")
        match = re.findall(r"\d+", sampled)
        reference_answer = ans.split("####")[1].strip()
        ### LOOSE
        if "The answer is " not in sampled:
            match = re.findall(r"\d+", sampled)
            match_pred = re.findall(r"\d+", reference_answer)
            if match:
                match_str = match[-1]
                return match_str, match_pred
        else:
            extracted = sampled.split("The answer is ")[1]
            match = re.findall(r"\d+", extracted)
            match_pred = re.findall(r"\d+", reference_answer)
            if match:
                match_str = match[0]
                return match_str, match_pred
        ## STRICT PROCESSING
        # if "The answer is " in sampled:
        #     extracted = sampled.split("The answer is ")[1]
        #     match = re.findall(r'\d+', extracted)
        #     match_pred = re.findall(r'\d+', reference_answer)
        #     if match:
        #         match_str = match[0]
        #         return match_str, match_pred
        # if match:
        # match_str = match[-1]
        return None, reference_answer
        # else:
        #     return None, reference_answer
        # return last_number
    if dataset == "math":

        sampled = last_boxed_only_string(sampled)
        sampled = remove_boxed(sampled)

        if is_equiv(sampled, ans):
            return ans, ans
        else:
            return sampled, ans


class MATH_PROBLEM(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        num_few_shot: int = 0,
        few_shot: str = None,
        samples_jsonl: str = None,
        few_shot_jsonl: str = None,
        max_tokens: int = 1024,
        *args,
        instructions: Optional[str] = "",
        temperature: float = 0.0,
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
        self.instructions = "Follow the given examples and answer the question."
        self.few_shot = num_few_shot
        self.few_shot_prompt = ""
        self.max_tokens = max_tokens
        self.sample_jsonl = samples_jsonl
        self.temperature = temperature
        self.prompt_template = [example_q_alias, example_a_alias, q_alias, a_alias]
        self.OUT_OF_WINDOW = False
        if self.few_shot > 0:
            self.example = evals.get_jsonl(few_shot_jsonl)

    def make_few_shot_prompt(self, num_shot):

        few_shot_prompt = ""
        for i in range((num_shot)):
            few_shot_prompt += self.prompt_template[0] + self.example[i]["question"] + "\n"
            few_shot_prompt += self.prompt_template[1] + self.example[i]["answer"] + "\n\n"

        return few_shot_prompt

    def pre_process(self, sample):
        for i in range(self.few_shot, -1, -1):
            few_shot_prompt = self.make_few_shot_prompt(i)
            prompt = (
                self.instructions
                + "\n\n"
                + few_shot_prompt
                + "\n\n"
                + self.prompt_template[2]
                + sample["question"]
                + f"\n{self.prompt_template[3]}Let's think step by step"
            )
            if num_tokens_from_messages(
                [prompt], self.completion_fn.model
            ) + 50 + self.max_tokens <= n_ctx_from_model_name(self.completion_fn.model):
                break
            self.OUT_OF_WINDOW = True

        return prompt, sample["answer"]

    def post_process(self, sampled, correct_answer):
        detected = False
        extracted, correct_answer = check_match(
            sampled, correct_answer, self.sample_jsonl.split("/")[0]
        )
        if extracted != None:
            sampled = extracted
            detected = True
        return sampled, correct_answer, detected

    def eval_sample(self, sample, rng):
        prompt, correct_answer = self.pre_process(sample)

        result = self.completion_fn(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]
        sampled, correct_answer, detected = self.post_process(sampled, correct_answer)
        evals.record_and_check_match(
            prompt=prompt, sampled=sampled, expected=correct_answer, detected=detected
        )

    def eval_sample_batch(self, recorder, samples):
        data = [self.pre_process(item)[0] for item in samples]
        ideal = [self.pre_process(item)[1] for item in samples]
        results = self.completion_fn(
            inputs=data,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        processed_res, detected, correct_answers = [], [], []
        for i in range(len(results)):
            a, answer, b = self.post_process(results[i], ideal[i])
            processed_res.append(a)
            detected.append(b)
            correct_answers.append(answer)

        # processed_res = [self.post_process(item)[0] for item in results]
        # detected = [self.post_process(item)[1] for item in results]
        for i in range(len(processed_res)):
            id = str(i)
            with recorder.as_default_recorder(id):
                record_sampling(prompt=data[i], sampled=results[i])
                evals.record_and_check_match(
                    prompt=data[i],
                    sampled=processed_res[i],
                    detected=detected[i],
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
            "detected": evals.metrics.get_picked(recorder.get_events("match")),
            "OUT OF WINDOW": self.OUT_OF_WINDOW,
        }
