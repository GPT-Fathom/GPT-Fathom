# How to customize your eval


## Customize your eval

To customize your `eval`, you can modify the arguments of `eval` in the registery. We support the modification of num of shots, prompt template, temperature and maximum tokens. Below is an example of custom `eval`:

```yaml
mmlu-abstract-algebra-3shot:
  id: mmlu-abstract-algebra.val.ab-v1
  metrics: [accuracy]
mmlu-abstract-algebra.val.ab-v1:
  class: evals.elsuite.basic.multiple_choice:MultipleChoice
  args:
    dataset: hf://hendrycks_test?name=abstract_algebra&split=test
    few_shot: hf://hendrycks_test?name=abstract_algebra&split=dev
    # Prompt Template are specified using the following five parameters, as:
    # [instructions]. 
    # [example_q_alias] Sample_k_question
    # [example_a_alias] Sample_k_answer
    # ...
    # [q_alias] question
    # [a_alias]
    instructions: The following are multiple choice questions (with answers) about abstract algebra.
    example_q_alias: Q: 
    example_a_alias: A: 
    q_alias: Q: 
    a_alias: A: 
    # Other settings: 
    no_MC_prompt: True # Avoid output the MCQ instruction at the end of the prompt: "Please answer with the letter of the correct answer."
    num_few_shot: 3    # Determine the number of shots
    max_tokens: 10    # Maximum output token length
    temperature: 0.0  # Set the temperature

```

The `args` field should match the arguments that your eval class `__init__` method expects. For example, the `args` above should match the arguments specified in [`evals.elsuite.basic.multiple_choice:MultipleChoice`](../evals/elsuite/basic/multiple_choice.py)
