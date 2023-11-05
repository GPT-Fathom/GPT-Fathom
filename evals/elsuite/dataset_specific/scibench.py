import re

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase, record_sampling
from evals.utils.math_util import last_boxed_only_string, remove_boxed


class SCIBENCH(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        num_few_shot: int = 0,
        task: str = None,
        sample_jsonl: str = None,
        max_tokens: int = 2048,
        type: str = None,
        CoT: bool = False,
        *args,
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
        self.CoT = CoT
        self.temperature = temperature
        self.few_shot_samples = []
        self.task = task
        self.instruction = """
        Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal number with three digits after the decimal point. Conclude the answer by stating "The answer is therefore \\boxed{[ANSWER]}."
        """

    # helper_function for post processing
    def remove_not(self, x):
        match_number = re.compile("[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?")
        result = re.findall(match_number, x)
        if len(result) != 0:
            return re.split(match_number, x)[-1]
        return None

    def parse_not(self, inputs):
        if not inputs:
            return "", ""
        if "\times" in inputs:
            x, ab = inputs.split("\times", 1)
        elif "\\times" in inputs:
            x, ab = inputs.split("\\times", 1)
        elif "*" in inputs:
            x, ab = inputs.split("*", 1)
        else:
            return inputs
        return x, ab

    def cal_not(self, inputs):
        try:
            x, ab = list(inputs)
            match_number = re.compile("10\^[{]?\ *-?[0-9]+\ *[}]?")
            ab = re.findall(match_number, ab)[0]
            ab = ab[ab.find("^") + 1 :]
            if "{" in ab:
                ab = ab[ab.find("{") + 1 :]
            if "}" in ab:
                ab = ab[: ab.find("}")]
            x = x.strip()
            out = float(x) * 10 ** float(ab)
            return str(out)
        except:
            return None
        return inputs

    def remove_boxed(self, s):
        left = "oxed{"  # change
        try:
            assert s[: len(left)] == left
            assert s[-1] == "}"
            answer = s[len(left) : -1]
            if "=" in answer:
                answer = answer.split("=")[-1].lstrip(" ")
            return answer
        except:
            return None

    def last_boxed_only_string(self, string):
        idx = string.rfind("oxed")  # change
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    def parse_math_answer(self, raw_string):
        if remove_boxed(last_boxed_only_string(raw_string)):
            return remove_boxed(last_boxed_only_string(raw_string))

        else:
            return ""

    def pre_process(self, sample):
        unit_prob = sample["unit"]
        if self.remove_not(sample["unit"]):
            unit_prob = self.remove_not(sample["unit"])
        problem_text = (
            sample["problem_text"] + " The unit of the answer is " + unit_prob + "."
        )

        prompt = self.instruction + "\n" + "Question: " + problem_text + "\n"
        if self.CoT:
            prompt += "Let's think step by step." + "\n"

        return prompt, sample

    def post_process(self, sampled, sample):
        sampled = self.parse_math_answer(sampled)
        correct_answer = sample["answer_number"]

        unit_prob = sample["unit"]
        if self.remove_not(sample["unit"]):
            unit_prob = self.remove_not(sample["unit"])

        if unit_prob != sample["unit"]:
            sampled = self.cal_not(self.parse_not(sampled))
            correct_answer = self.cal_not((correct_answer, sample["unit"]))

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

        evals.record_and_check_match_scibench(
            prompt=prompt, sampled=sampled, expected=correct_answer, unit=sample["unit"]
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
                evals.record_and_check_match_scibench(
                    prompt=data[i],
                    sampled=processed_res[i],
                    expected=correct_answers[i],
                    unit=ideal[i]["unit"],
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
