import os
from typing import Optional
from urllib.parse import parse_qs, urlparse

import evaluate
from datasets import load_dataset, load_from_disk
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.utils import num_tokens_from_messages
from evals.elsuite.utils_execute import compute
from evals.formatting import make_abc
from evals.record import RecorderBase, record_sampling
from evals.registry import n_ctx_from_model_name
from evals.registry import CHAT_MODELS


def get_dataset(local, dataset: str):
    if dataset == "humaneval":
        if not local:
            dataset = load_dataset("openai_humaneval", split="test")
        else:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../..",
                "registry",
                "cached_hf_data",
                "humaneval",
                "test",
            )
            dataset = load_from_disk(path)

        return [
            {
                "prompt": sample["prompt"],
                "test": [sample["test"] + f"\ncheck({sample['entry_point']})"],
            }
            for sample in dataset
        ]
    elif dataset == "mbpp":
        if not local:
            dataset = load_dataset("mbpp", "sanitized", split="test")
        else:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../..",
                "registry",
                "cached_hf_data",
                "mbpp",
                "test",
            )
            dataset = load_from_disk(path)

        return [
            {"prompt": sample["prompt"], "test": sample["test_list"]}
            for sample in dataset
        ]


class CODE(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        max_tokens: int = 512,
        dataset: str = None,
        num_samples_per_task: int = 100,
        *args,
        instructions: Optional[str] = "",
        temperature: float = 0.8,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert (
            len(completion_fns) == 1
        ), "MultipleChoice only supports one completion fn"
        self.max_tokens = max_tokens
        self.dataset = dataset
        self.num_samples_per_task = num_samples_per_task
        self.temperature = temperature
        # The "code_eval" metric executes untrusted model-generated code in Python.
        # Although it is highly unlikely that model-generated code will do something
        # overtly malicious in response to this test suite, model-generated code may act
        # destructively due to a lack of model capability or alignment.
        # Users are strongly encouraged to sandbox this evaluation suite so that it
        # does not perform destructive actions on their host or network. For more
        # information on how OpenAI sandboxes its code, see the paper "Evaluating Large
        # Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).

        # Once you have read this disclaimer and taken appropriate precautions,
        # set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
        # with:

        # import os

        # os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        self.OUT_OF_WINDOW = False
        if self.dataset == "humaneval":
            self.k = [1, 10, 100]
        elif self.dataset == "mbpp":
            self.k = [1, 80]

    def make_few_shot_prompts(self, shot):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../..",
            "registry",
            "cached_hf_data",
            "mbpp",
            "fewshot",
        )
        dataset = load_from_disk(path)
        prompt = ""
        for i in range(shot):
            test_case = "\n".join(dataset[i]["test_list"])
            prompt += "You are an expert Python programmer, and here is your task: "
            prompt += f"{dataset[i]['prompt']}. Your code should pass these tests:\n\n{test_case}\n[BEGIN]\n{dataset[i]['code']}\n[DONE]"
            prompt += "\n\n"

        return prompt

    def pre_process(self, sample):
        if self.dataset == "humaneval":
            prompt = (
                "Complete the code:\n" + sample["prompt"]
                if self.completion_fns[0].model not in CHAT_MODELS
                else "Complete the code and only output the completed code:\n"
                + sample["prompt"]
            )
        elif self.dataset == "mbpp":
            for i in range(3, -1, -1):
                prompt = self.make_few_shot_prompts(i)

                test_case = "\n".join(sample["test"])
                prompt += "You are an expert Python programmer, and here is your task: "
                prompt += f"{sample['prompt']}. Your code should pass these tests:\n\n{test_case}\n[BEGIN]\n"

                if num_tokens_from_messages(
                    [prompt], self.completion_fn.model
                ) + 50 + self.max_tokens <= n_ctx_from_model_name(
                    self.completion_fn.model
                ):
                    break
                self.OUT_OF_WINDOW = True
        return prompt, sample["test"]

    def eval_sample(self, sample, rng):

        prompt, correct_answer = self.pre_process(sample)

        if self.dataset == "humaneval":
            if self.completion_fns[0].model not in CHAT_MODELS:
                preds = [
                    prompt[len("Complete the code:\n") :]
                    + self.completion_fn(
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=["\nclass", "\ndef", "\n#", "\nif"],
                    ).get_completions()[0]
                    + "\n"
                    for _ in range(self.num_samples_per_task)
                ]
            elif self.completion_fns[0].model in ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]:
                preds = [
                    self.completion_fn(
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    ).get_completions()[0]
                    + "\n"
                    for _ in range(self.num_samples_per_task)
                ]
                preds = [item.split("```")[1][7:] if "```" in item else item for item in preds]
            else:
                preds = [
                    self.completion_fn(
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=["\nclass", "\ndef", "\n#", "\nif"],
                    ).get_completions()[0]
                    + "\n"
                    for _ in range(self.num_samples_per_task)
                ]
        elif self.dataset == "mbpp":
            preds = [
                    self.completion_fn(
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=["\n[DONE]"],
                    ).get_completions()[0]
                    + "\n"
                    for _ in range(self.num_samples_per_task)
                ]
            if self.completion_fns[0].model in ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]:
                preds = [item.split("```")[1][7:] if "```" in item else item for item in preds]

        sampled = [["sampled = None" if item is None else item for item in preds]]

        pass_at_k, results = compute(
            references=correct_answer, predictions=sampled, k=self.k
        )
        if self.dataset == "humaneval":
            evals.record.record_metrics(
                pass_at_1=pass_at_k["pass@1"],
                pass_at_10=pass_at_k["pass@10"],
                pass_at_100=pass_at_k["pass@100"],
            )
        elif self.dataset == "mbpp":
            evals.record.record_metrics(
                pass_at_1=pass_at_k["pass@1"],
                pass_at_80=pass_at_k["pass@80"],
            )

    def eval_sample_batch(self, recorder, samples):
        id = 0
        # for sample in samples:
        data, ideal = [], []
        for i in range(len(samples)):
            prompt, correct_answer = self.pre_process(samples[i])
            data.extend([prompt] * self.num_samples_per_task)
            ideal.append(correct_answer)

        response = self.completion_fn(
            inputs=data,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.95,
            max_tokens=self.max_tokens,
        )
        for id in range(len(samples)):
            results = response[
                id * self.num_samples_per_task : (id + 1) * self.num_samples_per_task
            ]
            prompt = data[id * self.num_samples_per_task]
            correct_answer = ideal[id]

            stop_sequences = (
                ["\nclass", "\ndef", "\n#", "\nif"]
                if self.dataset == "humaneval"
                else ["\n[DONE]"]
            )
            for i in range(len(results)):
                for x in stop_sequences:
                    if x in results[i]:
                        results[i] = results[i].split(x)[0]

            if self.dataset == "humaneval":
                results = [
                    prompt[len("Complete the code:\n") :] + item for item in results
                ]

            for i in range(len(results)):
                with recorder.as_default_recorder(id):
                    record_sampling(prompt=prompt, sampled=results[i])

            sampled = [["a" if item == None else item for item in results]]
            pass_at_k, results = compute(
                references=correct_answer, predictions=sampled, k=self.k
            )
            with recorder.as_default_recorder(id):
                if self.dataset == "humaneval":
                    evals.record.record_metrics(
                        pass_at_1=pass_at_k["pass@1"],
                        pass_at_10=pass_at_k["pass@10"],
                        pass_at_100=pass_at_k["pass@100"],
                    )
                elif self.dataset == "mbpp":
                    evals.record.record_metrics(
                        pass_at_1=pass_at_k["pass@1"],
                        pass_at_80=pass_at_k["pass@80"],
                    )

    def run(self, recorder: RecorderBase):
        samples = get_dataset(self.local_dataset, self.dataset)
        if self.eval_in_batch == False:
            self.eval_all_samples(recorder, samples)
        else:
            self.eval_sample_batch(recorder, samples)
        import numpy as np

        if self.dataset == "humaneval":
            return {
                "pass@1": np.mean(recorder.get_scores("pass_at_1")),
                "pass@10": np.mean(recorder.get_scores("pass_at_10")),
                "pass@100": np.mean(recorder.get_scores("pass_at_100")),
                "OUT OF WINDOW": self.OUT_OF_WINDOW,
            }
        elif self.dataset == "mbpp":
            return {
                "pass@1": np.mean(recorder.get_scores("pass_at_1")),
                "pass@80": np.mean(recorder.get_scores("pass_at_80")),
                "OUT OF WINDOW": self.OUT_OF_WINDOW,
            }
