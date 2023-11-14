import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase, record_sampling


class SUMMEDITS(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        num_few_shot: int = 0,
        sample_jsonl: str = None,
        max_tokens: int = 128,
        type: str = None,
        CoT: bool = False,
        Cn: bool = False,
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
        self.cn = Cn
        self.CoT = CoT
        self.temperature = temperature
        self.few_shot_samples = []

        self.instructions = 'Given the document below, you have to determine if "Yes" or "No", the summary is factually consistent with the document.'

    def pre_process(self, sample):
        prompt = ""
        prompt = (
            self.instructions
            + "\n\n"
            + "Document:"
            + "\n"
            + sample["doc"]
            + "\n\n"
            + "Summary:"
            + "\n"
            + sample["summary"]
            + "\n\n"
        )
        prompt += (
            "Is the summary factually consistent with the document? (Yes/No)" + "\n"
        )
        prompt += (
            'Start your answer explicitly with "Yes" or "No", and if you answer no, explain which sentence is inconsistent and why.'
            + "\n"
        )

        return prompt, sample

    # Helper function to extract answer from model's response
    def starts_with_yes_or_no(self, s):
        lowercased_string = s.lower()
        if lowercased_string.startswith("yes"):
            return "Yes"
        elif lowercased_string.startswith("no"):
            return "No"
        else:
            return "No"

    def post_process(self, sampled, sample):
        correct_answer = "No"
        if sample["label"] == 1:
            correct_answer = "Yes"
        else:
            correct_answer = "No"
        sampled = self.starts_with_yes_or_no(sampled)

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
