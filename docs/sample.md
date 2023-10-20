
**Evaluation of the Actor Model on arc-c-1shot and gsm8k-8shotCoT**

**Setup:**
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
     2. Write the completion_fns for the actor in the created file.
   - **Registration**
     1. Create a new `actor.yaml` in the `evals/registry/completion_fns`.
     2. Fill in:
        ```
        actor:
          class: evals.completion_fns.actor:ActorCompletion
        ```
     3. Update `/evals/registry.py` and add `"actor":2048,` in the `DICT_OF_N_CTX_BY_MODEL_NAME` dictionary on line 47.

3. **Custom Evaluation**
        ```
   - **Custom Prompt Template**
     1. Navigate to `evals/registry/evals/arc.yaml`.
     2. Locate the ID `arc_c.test.v2` for `arc-c-1shot`.
     3. Modify as per the instructions on [Custom Evaluation Instructions](https://github.com/GPT-Fathom/GPT-Fathom/blob/main/docs/custom-eval.md).
     4. Similarly, for `gsm8k-8shotCoT`, navigate to `evals/registry/evals/gsm8k.yaml` and make the required modifications.
   - **Running the Custom Evaluation**
     - Execute: 
        ```
        oaieval actor arc-c-1shot  --eval_in_batch True
        oaieval actor gsm8k-8shotCoT  --eval_in_batch True
        ```