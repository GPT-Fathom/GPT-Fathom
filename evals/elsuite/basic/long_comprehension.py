import random
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite import utils
from evals.elsuite.utils import num_tokens_from_messages
from evals.formatting import make_abc
from evals.record import RecorderBase, record_sampling
from evals.registry import n_ctx_from_model_name


class LONG_COMPRE(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        num_few_shot: int = 0,
        samples_jsonl: str = None,
        few_shot_jsonl: str = None,
        max_tokens: int = 30,
        MC: bool = False,
        no_MC_prompt: bool = False,
        *args,
        instructions: Optional[str] = "The following are questions (with answers) about reading comprehension. ",
        temerpature: float = 0.0,
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
        self.instructions = instructions
        self.few_shot = num_few_shot
        self.few_shot_prompt = ""
        self.max_tokens = max_tokens
        self.sample_jsonl = samples_jsonl
        self.MC = MC
        self.OUT_OF_WINDOW = False
        self.prompt_template = [example_q_alias, example_a_alias, q_alias, a_alias]
        self.temerpature = temerpature
        if self.few_shot > 0:
            self.example = evals.get_jsonl(few_shot_jsonl)
        self.no_MC_prompt = no_MC_prompt

    def make_few_shot_prompt(self, num_shot):
        few_shot_prompt = ""
        if not self.MC:
            for i in range((num_shot)):
                few_shot_prompt += "Passage: " + self.example[i]["passage"] + "\n"
                few_shot_prompt += "Question: " + self.example[i]["question"] + "\n"
                few_shot_prompt += "Answer: " + self.example[i]["ideal"][0] + "\n\n"
        else:
            rng = random.Random(47)
            for i in range((num_shot)):
                options, correct_answer, ans_ctx = make_abc(
                    answers=self.example[i]["options"],
                    correct_idx=self.example[i]["ideal"],
                    rng=rng,
                )
                few_shot_prompt += "Passage: " + self.example[i]["passage"] + "\n"
                few_shot_prompt += self.prompt_template[0] + self.example[i]["question"] + "\n" + options

                if not self.no_MC_prompt:
                    few_shot_prompt += (
                        "\nPlease answer with the letter of the correct answer.\n"
                    )
                else:
                    few_shot_prompt += "\n"
                few_shot_prompt += self.prompt_template[1] + ans_ctx + "\n\n"

        return few_shot_prompt

    def pre_process(self, sample):
        if not self.MC:
            correct_answers = sample["ideal"]
            if not isinstance(correct_answers, list):
                correct_answers = [correct_answers]
            prompt = (
                self.instructions
                + "\n\n"
                + self.make_few_shot_prompt(self.few_shot)
                + "\n\n"
            )
            max_num_tokens = n_ctx_from_model_name(self.completion_fn.model)
            num_tokens_prompt = num_tokens_from_messages([prompt], self.completion_fn.model)
            num_tokens_question = num_tokens_from_messages([sample["question"]], self.completion_fn.model)
            if num_tokens_from_messages([sample["passage"]], self.completion_fn.model) + num_tokens_question + num_tokens_prompt + self.max_tokens + 30 >= max_num_tokens:
                half = int((max_num_tokens - num_tokens_question - num_tokens_prompt - self.max_tokens)/2)
                sample["passage"] = sample["passage"][:half] + sample["passage"][-half:]
                while num_tokens_from_messages([sample["passage"]], self.completion_fn.model) + num_tokens_question + num_tokens_prompt + self.max_tokens + 30 >= max_num_tokens:
                    half = half - 500
                    sample["passage"] = sample["passage"][:half] + sample["passage"][-half:]
            prompt += (
                "Passage: "
                + sample["passage"]
                + "\n"
                + "Question: "
                + sample["question"]
                + "\nAnswer: "
            )
            return prompt, correct_answers
        else:
            options, correct_answer, _ = make_abc(
                answers=sample["options"],
                correct_idx=sample["ideal"],
                rng=random.Random(47),
            )
            for i in range(self.few_shot, -1, -1):
                few_shot_prompt = self.make_few_shot_prompt(i)
                prompt = self.instructions + "\n\n" + few_shot_prompt + "\n\n"
                prompt += (
                    "Passage: "
                    + sample["passage"]
                    + "\n"
                    + "Q: "
                    + sample["question"]
                    + options
                )

                if not self.no_MC_prompt:
                    prompt += (
                        "\nPlease answer with the letter of the correct answer.\nA: "
                    )
                else:
                    prompt += "\nA: "
                if num_tokens_from_messages(
                    [prompt], self.completion_fn.model
                ) + 50 + self.max_tokens <= n_ctx_from_model_name(
                    self.completion_fn.model
                ):
                    break
                self.OUT_OF_WINDOW = True
            return prompt, correct_answer

    def eval_sample(self, sample, rng):

        if not self.MC:
            prompt, correct_answers = self.pre_process(sample)
            result = self.completion_fn(
                prompt=prompt,
                temperature=self.temerpature,
                max_tokens=self.max_tokens,
            )
            sampled = result.get_completions()[0]

            matches = [
                utils.fuzzy_match(sampled, correct_answer)
                for correct_answer in correct_answers
            ]

            evals.record.record_match(
                True in matches,
                expected=correct_answers,
                picked=[sampled for i in range(len(correct_answers)) if matches[i]],
            )
            if not self.sample_jsonl in ["longbench/multifieldqa_zh.jsonl" ,"longbench/dureader.jsonl"]:
                evals.record.record_metrics(
                    accuracy=float(True in matches),
                    
                    f1_score=utils.f1_score(sampled, correct_answers),
                )
            else:
                score = 0
                for correct_answer in correct_answers:
                    score = max(score, utils.f1_score_chinese(sampled, correct_answer))
                evals.record.record_metrics(
                    accuracy=float(True in matches),
                    
                    f1_score=score,
                )
        else:
            prompt, correct_answer = self.pre_process(sample)
            result = self.completion_fn(
                prompt=prompt,
                temperature=self.temerpature,
                max_tokens=50,
            )
            sampled = result.get_completions()[0]
            options = ["(" + chr(ord("A") + idx) + ")" for idx in range(26)]
            ans = sampled
            for option in options:
                if option in sampled:
                    ans = option[1]
                    break
            sampled = ans
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
            max_tokens=30,
        )
        processed_res = [(item) for item in results]
        for i in range(len(processed_res)):
            id = str(i)
            with recorder.as_default_recorder(id):
                if self.MC:
                    record_sampling(prompt=data[i], sampled=results[i])
                    options = [chr(ord("A") + idx) + ")" for idx in range(26)]
                    sampled, correct_answer = results[i], ideal[i]
                    ans = sampled
                    for option in options:
                        if option in sampled:
                            ans = option[0]
                            break
                    sampled = ans
                    evals.record_and_check_match(
                        prompt=data[i],
                        sampled=sampled,
                        expected=correct_answer,
                    )
                else:
                    record_sampling(prompt=data[i], sampled=results[i])
                    sampled, correct_answers = results[i], ideal[i]

                    matches = [
                        utils.fuzzy_match(sampled, correct_answer)
                        for correct_answer in correct_answers
                    ]

                    evals.record.record_match(
                        True in matches,
                        expected=correct_answers,
                        picked=[
                            sampled for i in range(len(correct_answers)) if matches[i]
                        ],
                    )
                    evals.record.record_metrics(
                        accuracy=float(True in matches),
                        f1_score=utils.f1_score(sampled, correct_answers),
                    )

    def run(self, recorder: RecorderBase):
        samples = evals.get_jsonl(self.sample_jsonl)

        if self.eval_in_batch == False:
            self.eval_all_samples(recorder, samples)
        else:
            self.eval_sample_batch(recorder, samples)
        print("OUT OF WINDOW: ", self.OUT_OF_WINDOW)
        return {
            "accuracy": np.mean(recorder.get_scores("accuracy"))
            if not self.MC
            else evals.metrics.get_accuracy(recorder.get_events("match")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
            "OUT OF WINDOW": self.OUT_OF_WINDOW,
        }
