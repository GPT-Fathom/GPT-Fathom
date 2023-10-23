
# Example of evaluating a new model

Here we provide a step-by-step example of how to evaluate a new model using GPT-Fathom.

1. **Clone Repository & Environment Setup**
    - Clone the repo: 
        ``` 
        git clone https://github.com/GPT-Fathom/GPT-Fathom.git
        ```
    - Navigate to the directory: 
        ``` 
        cd GPT-Fathom/
        ```
    - Create and activate the conda environment: 
        ``` 
        conda create -n gptfathom python=3.9
        conda activate gptfathom
        ```

2. **Model Setup for Evaluation**
   - **Completion Functions**
     1. Create a new file `actor.py` in `evals/completion_fns` directory.
     2. Implement the `completion function` for the model in the created file. If not familiar with `completion function`, refer to [completion-fns.md](/docs/completion-fns.md) for details.
   - **Registration**
     1. Create a new file `actor.yaml` in `evals/registry/completion_fns` directory.
     2. Fill in:
        ```
        actor:
          class: evals.completion_fns.actor:ActorCompletion
        ```
     3. Register the new mode in `/evals/registry.py` by adding `"actor":2048` (the context window size 2048 should be modified accordingly) in the `DICT_OF_N_CTX_BY_MODEL_NAME` dict.

3. **Custom Evaluation**
   - **Custom Prompt Template**
     1. Navigate to `evals/registry/evals/arc.yaml`.
     2. Locate the ID `arc_c.test.v2` for `arc-c-1shot`.
     3. Modify as per the instructions on [custom-eval.md](/docs/custom-eval.md).
     4. Similarly, for `gsm8k-8shotCoT`, navigate to `evals/registry/evals/gsm8k.yaml` and make the required modifications.
   - **Running the Custom Evaluation**
     - Run in command line: 
        ```
        oaieval actor arc-c-1shot  --eval_in_batch True
        oaieval actor gsm8k-8shotCoT  --eval_in_batch True
        ```
