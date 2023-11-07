"""
This file provides common interfaces and utilities used by eval creators to
sample from models and process the results.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt, Prompt
from evals.record import record_match
from evals.utils import scibench_utils
from evals.utils import theoremqa_utils
import ast


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
        match = scibench_utils.equiv_scibench(sampled, expected, unit)
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


def record_and_check_match_theoremqa(
    prompt: Any,
    sampled: str,
    expected: str,
    answer_type: str,
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
    correct_answer = expected
    prediction = theoremqa_utils.normalize(sampled)

    if (answer_type == 'integer'):
        correct_answer = int(correct_answer)
    elif (answer_type == 'float'):
        correct_answer = float(correct_answer)
    elif (answer_type == 'list of integer'):
        correct_answer = ast.literal_eval(correct_answer)
    elif (answer_type == 'list of float'):
        correct_answer = ast.literal_eval(correct_answer)
    elif (answer_type == 'bool'):
        correct_answer = bool(correct_answer)

        
    if isinstance(prediction, (str, int, float)) or isinstance(prediction, list):
        # Comparing prediction against the reference
        if answer_type in ['bool', 'option', 'Option']:
            if (prediction == correct_answer):
                match = True

        elif answer_type == 'integer':
            if (theoremqa_utils.compare_two_numbers(prediction, correct_answer)):
                match = True
        elif answer_type == 'float':
            if (theoremqa_utils.compare_two_numbers(prediction, correct_answer)):
                match = True

        elif answer_type in ['list of integer', 'list of float']:
            if (theoremqa_utils.compare_two_list(prediction, correct_answer)):
                match = True

    result["expected"] = expected
    result["match"] = match
    record_match(
        match,
        expected=expected,
        sampled=sampled,
    )
    return match
