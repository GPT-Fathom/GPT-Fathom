# Completion Functions

## What are completion functions
In [run-evals.md](run-evals.md), we learned how to make calls to `oaieval` to run an `eval` against a `completion function`. Completion Functions are generalizations of model completions, where a "completion" is some text output that would be our answer to the prompt. For example, if "Who played the girl elf in the hobbit?" is our prompt, the correct completion is "Evangeline Lilly". While we can just test a model directly to see if it generates "Evangeline Lilly", we can imagine doing numerous other operations under the hood to improve our ability to answer this question, like giving the model access to a browser to look up the answer before responding. Making it easy to implement this kind of under-the-hood operators before responding is the motivation behind building Completion Functions.

## How to implement completion functions
A `completion function` needs to implement some interfaces that make it usable within `Evals`. At its core, it is just standardizing inputs to be a text string or [Chat conversation](https://platform.openai.com/docs/guides/chat), and the output to be a list of text strings. Implementing this interface will allow you to run your `completion function` against any evals in GPT-Fathom.

The exact interfaces needed are described in detail in [completion-fn-protocol.md](completion-fn-protocol.md).

We support two kinds of completion function: 
- Given a single sample as input, output one response (e.g., OpenAI Completion);
- Given the whole dataset (as a list of samples), output the responses for all samples as a list (e.g., Llama 2).

The two completion functions share the same interfaces, but for the second type of completion function, you need to specify the argument `--eval_in_batch True` when you call `oaieval` or `oaievalset`. 

We provide some example implementations inside `evals/completion_fns`. For example, the [`LLamaCompletion Function`](evals/completion_fns/llama.py) is an implementation to generate completions from LLaMA and Llama 2 LLMs. 

## Registering Completion Functions
Once you have written a completion function, we need to make the class visible to the `oaieval` CLI. Similar to how we register our evals, we also register Completion Functions inside `evals/registry/completion_fns` as `yaml` files. Here is the registration for our Llama LLM completion function:
```yaml
llama:
  class: evals.completion_fns.llama:LLAMACompletion

llama2:
  class: evals.completion_fns.llama:LLAMA2Completion
```
After register your completion functions, you can call oaieval with
```
oaieval llama lambada-0shot --eval_in_batch True
```
