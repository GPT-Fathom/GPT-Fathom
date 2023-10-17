# Quick Eval with One Line of Code
Modify the ```text-davinci-001``` to the model you want to evaluate. 

Remember to specific your openai api key as an environment variable ```OPENAI_API_KEY="YOURKEY"```. 

By default, we run with 10 threads, you can configure this with an environment variable such as ```EVALS_THREADS=42```. 

To evaluate your own model, see [`completion-fns.md`](docs/completion-fns.md). To modify/create an eval, see [`custom-eval.md`](docs/custom-eval.md). 

If you need to use our local cached huggingface data, specify `--local_dataset True`

More details of running an eval are in [`run-evals.md`](run-evals.md).

## Knowledge
```
# Closed-book QA: 
oaieval text-davinci-001 natural_questions-1shot
oaieval text-davinci-001 web_questions-1shot
oaieval text-davinci-001 triviaqa-1shot

# Multi-subject Test
oaievalset text-davinci-001 mmlu-set-5shot
oaievalset text-davinci-001 agieval-en-set-5shot
oaieval text-davinci-001 arc-e-1shot
oaieval text-davinci-001 arc-c-1shot
```

## Reasoning
```
# Commonsense Reasoning
oaieval text-davinci-001 lambada-1shot
oaieval text-davinci-001 hellaswag-1shot
oaieval text-davinci-001 winogrande-1shot

# Comprehensive Reasoning
oaievalset text-davinci-001 bbh-set-CoT
```

## Comprehension
```
# Reading Comprehension
oaieval text-davinci-001 race-m-1shot
oaieval text-davinci-001 race-h-1shot
oaieval text-davinci-001 drop-3shot
```

## Math
```
# Math Reasoning
oaieval text-davinci-001 gsm8k-8shotCoT
oaieval text-davinci-001 math-4shotCoT
```

## Code
```
# Coding Problems
oaieval text-davinci-001 humaneval-0shot
oaieval text-davinci-001 mbpp-3shot
```

## Multilingual
```
# Multi-subject Test
oaievalset text-davinci-001 agieval-cn-set-5shot
oaievalset text-davinci-001 ceval-set-5shot

# Math Reasoning
oaievalset text-davinci-001 mgsm-all-set

# QA
oaievalset text-davinci-001 tydiqa-nocontext-set
```

## Safety
```
# Truthfulness
oaieval text-davinci-001 truthfulqa-0shot

# Toxicity
oaieval text-davinci-001 realtoxic
```
