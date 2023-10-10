# How to run evals

We provide two command line interfaces (CLIs): `oaieval` for running a single eval and `oaievalset` for running a set of evals. To perform a quick eval with one line of code, see [quick-evals.md](quick-evals.md). 

## Running an eval

When using the `oaieval` command, you will need to provide the `completion function` you wish to evaluate as well as the `eval` to run:
```sh
oaieval gpt-3.5-turbo test-match
```
The valid eval names are specified in the YAML files under `evals/registry/evals` and their corresponding implementations canbe found in `evals/elsuite`.

In this example, `gpt-3.5-turbo` is an OpenAI model that we dynamically instantiate as a completion function using `OpenAIChatCompletionFn(model=gpt-3.5-turbo)`. Any implementation of the `CompletionFn` protocol can be run against `oaieval`. By default, we support calling `oaieval` with any models available in the [OpenAI API](https://platform.openai.com/docs/models/) and [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models), or with CompletionFunctions available in [evals/registry/completion_fns](../evals/registry/completion_fns/).
- For evaluations using [OpenAI API](https://platform.openai.com/docs/models/), set your API key as an environment variable: ```OPENAI_API_KEY="YOURKEY"```. 
- To evaluate models in [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models), make sure you first deploy the model following [Azure Deploy](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal) to get a `deployment id`. Then specify a `completion function` name to bind with your `deployment id` in [evals/completion_fns/openai.py](https://github.com/yuyuz/GPT-Fathom/blob/fcb21e048aa4a68f5f66fa9079438c465d0d826b/evals/completion_fns/openai.py#L166), and specify your `api_key` and `api_base` in [evals/utils/azure_utils.py](/evals/utils/azure_utils.py). Run evaluation with flag `--azure_eval True`, for example:
```sh
oaieval text-davinci-001 gsm8k-8shotCoT --azure_eval True
```
- To evaluate a model from LLaMA or Llama 2 family, first configure the model following the official HuggingFace documents for [LLaMA](https://huggingface.co/docs/transformers/main/model_doc/llama) and [Llama 2](https://huggingface.co/docs/transformers/main/model_doc/llama2). Then run evaluation with flag `--eval_in_batch True`, for example:
```sh
oaieval llama gsm8k-8shotCoT --eval_in_batch True
```

Refer to [completion-fns.md](completion-fns.md) for more details of `CompletionFn`.

These CLIs accept various flags to modify their default behavior. For example:
- By default, logging locally will write to `tmp/evallogs`, and you can change this by setting a different `--record_path`.

You can run `oaieval --help` to see a full list of CLI options.

## Running an eval set

```sh
oaievalset gpt-3.5-turbo test
```

Similarly, `oaievalset` also expects a model name and an eval set name, for which the valid options are specified in the YAML files under `evals/registry/eval_sets`.

By default we run with 10 threads, and each thread times out and restarts after 40 seconds. You can configure this, e.g.,

```sh
EVALS_THREADS=42 EVALS_THREAD_TIMEOUT=600 oaievalset gpt-3.5-turbo test
```
Running with more threads will make the eval faster, though keep in mind the costs and your [rate limits](https://platform.openai.com/docs/guides/rate-limits/overview). Running with a higher thread timeout may be necessary if you expect each sample to take a long time, e.g., the data contain long prompts that elicit long responses from the model.

If you have to stop your run or your run crashes, we've got you covered! `oaievalset` records the evals that finished in `/tmp/oaievalset/{model}.{eval_set}.progress.txt`. You can simply rerun the command to pick up where you left off. If you want to run the eval set starting from the beginning, delete this progress file.

Unfortunately, you can't resume a single eval from the middle. You'll have to restart from the beginning, so try to keep your individual evals quick to run.

The results of evaluation will be logged to `/tmp/res/{model}.json` to leverage a simpler result extraction. 
