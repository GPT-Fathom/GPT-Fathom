import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase, record_sampling


class THEOREMQA(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        num_few_shot: int = 0,
        task: str = None,
        sample_jsonl: str = None,
        max_tokens: int = 1028,
        type: str = None,
        CoT: bool = False,
        Cn: bool = False,
        with_theorem: bool = False,
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
        self.task = task
        self.with_theorem = with_theorem

        self.instructions = """You are a mathematician, you are supposed to answer the given question. You need to output the answer in your final sentence like "Therefore, the answer is ...". The answer can only be one of the following forms:
                1. a numerical value like 0.1, no symbol and no unit at all.
                2. a list of number like [2, 3, 4].
                3. True/False.
                4. an option like (a), (b), (c), (d)
            """

    # Helper function to extract answer from model's response
    def extract_answer(self, result):
        prediction = result.strip().strip("\n").split("\n")[-1]
        tmp = ""
        for entry in prediction.split(" ")[::-1]:
            if entry == "is" or entry == "be" or entry == "are" or entry.endswith(":"):
                break
            tmp = entry + " " + tmp
        prediction = tmp.strip().strip(".")
        return prediction

    def pre_process(self, sample):
        prompt = self.instructions

        prompt += "Question: " + sample["Question"] + "\n\n"

        if self.with_theorem:
            prompt += (
                "Here is the theorem that can help you answer the question: " + "\n"
            )
            prompt += sample["theorem"] + ": " + sample["theorem_def"] + "\n\n"

        if self.CoT:
            prompt += "Let's think step by step." + "\n\n"

        return prompt, sample

    def post_process(self, sampled, sample):
        sampled = self.extract_answer(sampled)
        correct_answer = sample["Answer"]

        return sampled, correct_answer

    def eval_sample(self, sample, rng):
        prompt, sample = self.pre_process(sample)

        result = self.completion_fn(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )

        sampled = result.get_completions()[0]
        sampled, correct_answer = self.post_process(sampled, sample)
        evals.record_and_check_match_theoremqa(
            prompt=prompt,
            sampled=sampled,
            expected=correct_answer,
            answer_type=sample["Answer_type"],
        )

    def eval_sample_batch(self, recorder, samples):
        data = [self.pre_process(item)[0] for item in samples]
        ideal = [self.pre_process(item)[1] for item in samples]
        results = self.completion_fn(
            inputs=data,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        processed_res, correct_answers = zip(
            *[self.post_process(results[i], ideal[i]) for i in range(len(results))]
        )

        for i in range(len(processed_res)):
            id = str(i)
            with recorder.as_default_recorder(id):
                record_sampling(prompt=data[i], sampled=results[i])
                evals.record_and_check_match_theoremqa(
                    prompt=data[i],
                    sampled=processed_res[i],
                    expected=correct_answers[i],
                    answer_type=ideal[i]["Answer_type"],
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
