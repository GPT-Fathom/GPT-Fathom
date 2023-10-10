### The Completion Function Protocol

Here are the interfaces needed to implement the completion function protocol. Any implementation of this interface can be used inside `oaieval`.


#### CompletionFn
Completion functions should implement the `CompletionFn` interface:
```python
class CompletionFn(Protocol):
    def __call__(
        self,
        prompt: Union[str, list[dict[str, str]]],
        **kwargs,
    ) -> CompletionResult:
```

We take a `prompt` representing a single sample or a list of samples from an eval. 

We support two kinds of completion function: 
- Given a single sample as input, output one response (eg. OpenAI Completion)
- Given the whole dataset (as a list of samples), output the responses for all samples as a list. (eg. Llama)

The two completion functions share the same interfaces, but for the second type of completion function, the function should return a list of responses, while for the first type only one reponse should be return. 

Reference implementations:
- [OpenAICompletionFn](../evals/completion_fns/openai.py) - Type 1
- [`LLAMACompletionFn`](../evals/completion_fns/llama.py) - Type 2



#### Using your CompletionFn
This is all that's needed to implement a Completion function that works with our existing Evals, allowing you to more easily evaluate your end-to-end logic on tasks.

See [completion-fns.md](completion-fns.md) to see how to register and use your completion function with `oaieval`.
