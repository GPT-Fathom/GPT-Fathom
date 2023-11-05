"""
This file provides common interfaces and utilities used by eval creators to
sample from models and process the results.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt, Prompt
from evals.record import record_match
import math

logger = logging.getLogger(__name__)


class CompletionResult(ABC):
    @abstractmethod
    def get_completions(self) -> list[str]:
        pass


@runtime_checkable
class CompletionFn(Protocol):
    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> CompletionResult:
        """
        ARGS
        ====
        `prompt`: Either a `Prompt` object or a raw prompt that will get wrapped in
            the appropriate `Prompt` class.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The result of the API call.
        The prompt that was fed into the API call as a str.
        """


class DummyCompletionResult(CompletionResult):
    def get_completions(self) -> list[str]:
        return ["This is a dummy response."]


class DummyCompletionFn(CompletionFn):
    def __call__(
        self,
        prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
        **kwargs,
    ) -> CompletionResult:
        return DummyCompletionResult()


def record_and_check_match(
    prompt: Any,
    sampled: str,
    expected: Union[str, list[str], tuple[str]],
    separator: Callable[[str], bool] = None,
    options: Optional[list[str]] = None,
    detected: bool = False,
):
    """
    Records and checks if a sampled response from a CompletionFn matches the expected result.

    Args:
        prompt: The input prompt.
        sampled: The sampled response from the model.
        expected: The expected response or list of responses.
        separator: Optional function to check if a character is a separator.
        options: Optional list of options to match against the sampled response.

    Returns:
        The matched option or None if no match found.
    """
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]
    if options is None:
        options = expected

    picked = None
    for option in options:
        if not sampled.startswith(option):
            continue
        if (
            separator is not None
            and len(sampled) > len(option)
            and not separator(sampled[len(option)])
        ):
            continue
        picked = option
        break

    result = {
        "prompt": prompt,
        "sampled": sampled,
        "options": options,
        "picked": picked,
    }
    match = picked in expected
    result["expected"] = expected
    result["match"] = match
    record_match(
        match,
        expected=expected,
        picked=picked,
        sampled=sampled,
        options=options,
        detected=detected,
    )
    return picked


def equiv_scibench(model_output, answer, unit):
    """
    Helper function for record_and_check_match_scibench. Checks the numerical equivalence between model output and answer.

    Args:
        model_output: The sampled response from the model.
        answer: The expected response.
        unit: The unit of the answer.

    Returns:
        If the model output matches the expected answer.
    """

    model_output = model_output.replace(",", "")
    try:
        ans = float(answer.strip())
        if ans >= 1:
            first = math.isclose(float(model_output.strip()), ans, abs_tol=0.1)
        else:
            first = math.isclose(float(model_output.strip()), ans, rel_tol=0.1)
    except:
        first = False
    try:
        model = model_output.strip().split()[0]
        if ans >= 1:
            second = math.isclose(float(model_output.strip()), ans, abs_tol=0.1)
        else:
            second = math.isclose(float(model_output.strip()), ans, rel_tol=0.1)
    except:
        second = False
    if first or second:
        return True
    return False


def record_and_check_match_scibench(
    prompt: Any,
    sampled: str,
    expected: str,
    unit: str,
):
    """
    Records and checks if a sampled response from a CompletionFn matches the expected result for scibench.

    Args:
        prompt: The input prompt.
        sampled: The sampled response from the model.
        expected: The expected response or list of responses.
        unit: The unit of the answer

    Returns:
        If the model output matches the expected answer.
    """
    result = {"prompt": prompt, "sampled": sampled}
    match = False
    try:
        match = equiv_scibench(sampled, expected, unit)
    except Exception as e:
        match = False

    result["expected"] = expected
    result["match"] = match
    record_match(
        match,
        expected=expected,
        sampled=sampled,
    )
    return match