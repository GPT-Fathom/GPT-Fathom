from typing import Any, Optional, Union

import torch

from evals.api import CompletionFn, CompletionResult
from evals.base import CompletionFnSpec
from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
    chat_prompt_to_text_prompt,
)
from evals.record import record_sampling
from evals.utils.api_utils import (
    openai_chat_completion_create_retrying,
    openai_completion_create_retrying,
)


class OpenAIBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class OpenAICompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and "choices" in self.raw_data:
            for choice in self.raw_data["choices"]:
                if "text" in choice:
                    completions.append(choice["text"])
        return completions


class LLAMACompletion(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = "llama"
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        inputs,
        **kwargs,
    ) -> OpenAICompletionResult:
        from tqdm import tqdm
        from transformers import LlamaForCausalLM, LlamaTokenizer

        batch_size = 32
        text = [sample for sample in inputs]

        num_batches = len(text) // batch_size + 1
        # print(len(text))
        llama_path = None
        assert llama_path!=None, "Fix your llama path here"

        tokenizer = LlamaTokenizer.from_pretrained(
            llama_path, use_fast=False, padding_side="left"
        )
        model = LlamaForCausalLM.from_pretrained(
            llama_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        n_gpus = torch.cuda.device_count()
        import tensor_parallel as tp

        # model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
        tokenizer.pad_token_id = (
            0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        )
        tokenizer.bos_token_id = 1

        model.eval()
        # print(num_batches)
        res = []
        for i in tqdm(range(num_batches)):
            batch = text[i * batch_size : min((i + 1) * batch_size, len(text))]
            if len(batch) == 0:
                continue
            input_texts = batch
            inputs = tokenizer.batch_encode_plus(
                input_texts, return_tensors="pt", padding=True
            ).to("cuda")
            with torch.no_grad():
                if "do_sample" not in kwargs.keys():
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=kwargs["max_tokens"],
                        temperature=kwargs.get("temperature", 0.0),
                    )
                else:
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=kwargs["max_tokens"],
                        temperature=kwargs.get("temperature", 0.0),
                        do_sample=kwargs.get("do_sample", 0.0),
                        top_p=kwargs.get("top_p", 0.95),
                    )
            result = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            res.extend(result)

        for i in range(len(res)):
            res[i] = res[i][len(text[i]) :]

        return res


class LLAMA2Completion(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = "llama2"
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        inputs,
        **kwargs,
    ) -> OpenAICompletionResult:
        from tqdm import tqdm
        from transformers import LlamaForCausalLM, LlamaTokenizer

        batch_size = 32
        text = [sample for sample in inputs]

        num_batches = len(text) // batch_size + 1
        max_len = max([len(item) for item in text])

        tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-70b-hf",
            use_fast=False,
            padding_side="left",
        )
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-70b-hf",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        n_gpus = torch.cuda.device_count()
        import tensor_parallel as tp

        # model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
        tokenizer.pad_token_id = (
            0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        )
        tokenizer.bos_token_id = 1

        model.eval()

        res = []
        for i in tqdm(range(num_batches)):
            batch = text[i * batch_size : min((i + 1) * batch_size, len(text))]
            if len(batch) == 0:
                continue
            input_texts = batch
            inputs = tokenizer.batch_encode_plus(
                input_texts, return_tensors="pt", padding=True
            ).to("cuda")
            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=kwargs["max_tokens"],
                    temperature=kwargs.get("temperature", 0.0),
                )
            result = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            res.extend(result)

        for i in range(len(res)):
            res[i] = res[i][len(text[i]) :]

        return res
