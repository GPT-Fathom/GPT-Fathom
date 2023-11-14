from typing import Any

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite import utils
from evals.prompt.base import chat_prompt_to_text_prompt, is_chat_prompt
from evals.record import record_sampling
from evals.registry import CHAT_MODELS
# CHAT_MODELS = {
#     "gpt-3.5-turbo",
#     "gpt-3.5-turbo-0301",
#     "gpt-4",
#     "gpt-4-0314",
#     "gpt-4-32k",
#     "gpt-4-32k-0314",
# }


class Includes(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        ignore_case: bool = False,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        temperature: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Includes only supports one completion fn"
        self.samples_jsonl = samples_jsonl
        self.ignore_case = ignore_case
        self.num_few_shot = num_few_shot
        self.temperature = temperature
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

    def pre_process(self, test_sample):
        prompt = self.prompt_to_fewshot(test_sample)
        if self.completion_fns[0].model not in CHAT_MODELS:
            if self.num_few_shot == 0:
                prompt = chat_prompt_to_text_prompt(
                    prompt, Q_A_ENDING=True, new_line_end=False
                )
            else:
                prompt = chat_prompt_to_text_prompt(
                    prompt, Q_A_ENDING=True, new_line_end=False
                )

        correct_answers = test_sample["ideal"]
        if not isinstance(correct_answers, list):
            correct_answers = [correct_answers]
        return prompt, correct_answers

    def eval_sample(self, sample: Any, *_):

        prompt, ideal = self.pre_process(sample)
        result = self.completion_fn(
            prompt=prompt,
            temperature = self.temperature
        )
        sampled = result.get_completions()[0]

        includes_answer = any(
            [
                utils.get_answer(sampled, ref, self.ignore_case) is not None
                for ref in ideal
            ]
        )
        evals.record.record_match(
            includes_answer, expected=sample["ideal"], picked=sampled, sampled=sampled
        )
        return includes_answer

    def eval_sample_batch(self, recorder, samples):
        data = [self.pre_process(item)[0] for item in samples]
        ideal = [self.pre_process(item)[1] for item in samples]
        results = self.completion_fn(inputs=data, temperature=self.temperature, max_tokens=50)
        processed_res = [(item) for item in results]
        for i in range(len(processed_res)):
            correct_answers = ideal[i]
            sampled = processed_res[i]
            includes_answer = any(
                [
                    utils.get_answer(sampled, ref, self.ignore_case) is not None
                    for ref in correct_answers
                ]
            )

            id = str(i)
            with recorder.as_default_recorder(id):
                record_sampling(prompt=data[i], sampled=results[i])
                evals.record.record_match(
                    includes_answer,
                    expected=correct_answers,
                    picked=sampled,
                    sampled=sampled,
                )

    def run(self, recorder):
        samples = self.get_samples()
        if self.eval_in_batch == False:
            self.eval_all_samples(recorder, samples)
        else:
            self.eval_sample_batch(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
        }
