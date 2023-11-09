import concurrent
import logging
import os
import time

import backoff
import openai
import litellm

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))
SLEEP_TIME = float(
    os.environ.get("SLEEP_TIME", "1.5")
)  

@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def azure_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    openai.api_base = "API-BASE"
    openai.api_key = "API-KEY"

    result = openai.Completion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    time.sleep(SLEEP_TIME)
    return result


def request_with_timeout(func, *args, timeout=EVALS_THREAD_TIMEOUT, **kwargs):
    """
    Worker thread for making a single request within allotted time.
    """
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError as e:
                continue


@backoff.on_exception(
    wait_gen=backoff.constant,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
        openai.error.InvalidRequestError,
    ),
    interval=5
    # max_value=60,
)
def azure_chat_completion_create_retrying(GPT_4: bool = False, 
                                          frontend: bool = False,
                                          *args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    openai.api_base = "API-BASE"
    openai.api_key = "API-KEY"
    result = request_with_timeout(litellm.completion, *args, **kwargs)
    time.sleep(SLEEP_TIME)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result
