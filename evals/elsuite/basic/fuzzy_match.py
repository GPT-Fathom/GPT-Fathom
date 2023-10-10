import numpy as np

import evals
from evals.api import CompletionFn
from evals.elsuite import utils
from evals.prompt.base import chat_prompt_to_text_prompt, is_chat_prompt
from evals.record import RecorderBase, record_sampling


class FuzzyMatch(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        *args,
        max_tokens: int = 100,
        temerpature: float = 0.0,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "FuzzyMatch only supports one completion fn"
        self.max_tokens = max_tokens
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

    def pre_process(self, test_sample):
        prompt = self.prompt_to_fewshot(test_sample)
        prompt = chat_prompt_to_text_prompt(prompt, Q_A_ENDING=True, new_line_end=False)
        correct_answers = test_sample["ideal"]
        if not isinstance(correct_answers, list):
            correct_answers = [correct_answers]
        return prompt, correct_answers

    def eval_sample(self, test_sample, rng):

        prompt, correct_answers = self.pre_process(test_sample)

        result = self.completion_fn(
            prompt=prompt,
            temperature=self.temerpature,  # Q: why are these hardcoded?
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
        evals.record.record_metrics(
            accuracy=float(True in matches),
            f1_score=utils.f1_score(sampled, correct_answers),
        )

    def eval_sample_batch(self, recorder, samples):
        data = [self.pre_process(item)[0] for item in samples]
        ideal = [self.pre_process(item)[1] for item in samples]
        results = self.completion_fn(
            inputs=data,
            temperature=self.temerpature,
            max_tokens=50,
        )
        processed_res = [(item) for item in results]
        for i in range(len(processed_res)):
            correct_answers = ideal[i]
            sampled = processed_res[i]
            matches = [
                utils.fuzzy_match(processed_res[i], correct_answer)
                for correct_answer in correct_answers
            ]

            id = str(i)
            with recorder.as_default_recorder(id):
                record_sampling(prompt=data[i], sampled=results[i])
                evals.record.record_match(
                    True in matches,
                    expected=correct_answers,
                    picked=[sampled for i in range(len(correct_answers)) if matches[i]],
                )
                evals.record.record_metrics(
                    accuracy=float(True in matches),
                    f1_score=utils.f1_score(sampled, correct_answers),
                )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        if self.eval_in_batch == False:
            self.eval_all_samples(recorder, samples)
        else:
            self.eval_sample_batch(recorder, samples)
        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
