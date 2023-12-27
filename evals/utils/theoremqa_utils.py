import ast
import os
import re
from time import sleep

import numpy as np
import openai
import litellm
import wolframalpha
from sympy import Rational

wolfram_client = wolframalpha.Client(os.getenv("WOLFRAM_KEY"))


def extract_answer(query: str, use_azure: bool):
    if use_azure:
        openai.api_key = os.getenv("AZURE_KEY")
        openai.api_type = "azure"
        openai.api_base = "https://waterloogpt.openai.azure.com/"
        openai.api_version = "2023-03-15-preview"
        kwargs = {"engine": "ChatGPT"}
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        kwargs = {"model": "gpt-3.5-turbo"}

    SYSTEMQ = "You are supposed to extract the numeric answer (answer or Python formula or latex form) from a given string. If there is a unit in the input, try to remove that and only keep the number. If you think there is no numerical number within the input, just return 0."
    # greedy decoding
    got_result = False
    full_prompt = f"""
Input: 1/8 for both starting positions
Output: 1/8

Input: 0 an Euler homogeneous equation?
Output: 0

Input: based on the evaluation, the answer is [2, 3, 4].
Output: [2, 3, 4]

Input: Therefore, it will take 330 ms for client A to receive the whole file from the server after sending a request
Output: 330

Input: 3.02xmath.pow(10, 16) V
Output: 3.02*math.pow(10, 16)

Input: 4kHz
Output: 4

Input: individual will work 4,800 hours
Output: 4800

Input: the overall margin exceeds $13,133.4
Output: 13133.4

Input: x^y - 2e(x)
Output: 0

Input: 0.3465735 (approximate value)
Output: 0.3465735

Input: 3 and 4
Output: [3, 4]

Input: 3.57 * 10^(-29)
Output: 3.57 * math.pow(10, -29)

Input: {query}
Output:"""
    while not got_result:
        try:
            result = litellm.completion(
                **kwargs,
                messages=[
                    {"role": "system", "content": SYSTEMQ},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=1028,
                temperature=0.0,
                top_p=1,
                n=1,
            )
            got_result = True
        except Exception as e:
            print("Error:", e)
            sleep(3)
    result = result["choices"][0]["message"]["content"].strip()
    return result


def get_decimal_with_wolfram(string: str) -> float:
    for ex in wolfram_client.query(f"compute {string}").pods:
        if ex["@title"] in ["Decimal approximation", "Decimal form"]:
            for sub in ex.subpods:
                try:
                    return float(sub["plaintext"][:20])
                except Exception:
                    pass

    for ex in wolfram_client.query(f"compute {string}").pods:
        if ex["@title"] in ["Result"]:
            for sub in ex.subpods:
                try:
                    return float(sub["plaintext"][:8])
                except Exception:
                    pass

    return None


def find_numbers_in_string(s: str):
    pattern = r"[-+]?(?:\d*\.*\d+)"
    numbers = re.findall(pattern, s)
    tmp = [float(x) for x in numbers]
    if len(tmp) == 0:
        return None
    elif len(tmp) == 1:
        return tmp[0]
    else:
        return tmp


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def parse_number_list(s: str):
    # Check if the string is a valid list by trying to parse it
    parsed_list = ast.literal_eval(s)
    return parsed_list


def compare_two_numbers(p, gt):
    if isinstance(p, int) or isinstance(p, float):
        pass
    elif isinstance(p, list) or isinstance(p, bool) or isinstance(p, str):
        return False
    elif isinstance(p, tuple) or isinstance(p, complex) or isinstance(p, dict):
        return False
    else:
        raise ValueError(p)

    if isinstance(gt, float):
        return within_eps(pred=p, gt=gt)
    else:
        return round(p) == gt


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    elif len(pred) != len(gt):
        return False
    elif any([not isinstance(x, (int, float)) for x in pred]):
        return False
    else:
        pred = sorted(pred)
        gt = sorted(gt)
        return all([compare_two_numbers(p, g) for p, g in zip(pred, gt)])


def is_number(string):
    pattern = r"^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$"
    match = re.match(pattern, string)
    return bool(match)


def is_scientific_number(string):
    pattern = r"^[-+]?\d+(\.\d+)?e[-]?\d+$"
    match = re.match(pattern, string)
    return bool(match)


def contain_num_and_str(string):
    pattern_str = r"[a-zA-Z]"
    pattern_num = r"[0-9]"
    return bool(re.search(pattern_str, string) and re.search(pattern_num, string))


def normalize(prediction: str):
    # Preprocessing the string [Stage 1]
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else "0"

    # Replace special tokens
    if "=" in prediction:
        prediction = prediction.split("=")[-1].strip()
    if "≈" in prediction:
        prediction = prediction.split("≈")[-1].strip()
    if "`" in prediction:
        prediction = prediction.replace("`", "")
    if "$" in prediction:
        prediction = prediction.replace("$", "")
    if "°" in prediction:
        prediction = prediction.replace("°", "")

    # Detect the boolean keyword in the generation
    if prediction in ["true", "yes", "false", "no"]:
        if prediction == "true" or prediction == "yes":
            prediction = "True"
        else:
            prediction = "False"
    if "True" in prediction or "False" in prediction:
        prediction = "True" if "True" in prediction else "False"

    # Detect the approximation keyword
    if "approximately" in prediction:
        prediction = prediction.replace("approximately", "").strip()
    if " or " in prediction:
        prediction = prediction.split(" or ")[0]

    # Drop the units before and after the number
    if re.match(r"[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$", prediction):
        prediction = re.search(r"([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$", prediction).group(
            1
        )
    if re.match(r"[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$", prediction):
        prediction = re.search(r"[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$", prediction).group(
            1
        )
    if re.match(r"[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$", prediction):
        prediction = re.search(r"([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$", prediction).group(
            1
        )
    if re.match(r"[^-+\d]{1,2}(?:[\d,]*\.*\d+)$", prediction):
        prediction = re.search(r"[^-+\d]{1,2}((?:[\d,]*\.*\d+))$", prediction).group(1)

    # Preprocessing the number [Stage 1]
    if "10^" in prediction:
        prediction = re.sub(r"10\^(-?\d+)", r"math.pow(10, \1)", prediction)
    if " x " in prediction:
        prediction = prediction.replace(" x ", "*")
    if " × " in prediction:
        prediction = prediction.replace(" × ", "*")
    if is_number(prediction):
        prediction = prediction.replace(",", "")

    # Preprocessing the option [Stage 3]
    if (
        "(a)" in prediction
        or "(b)" in prediction
        or "(c)" in prediction
        or "(d)" in prediction
    ):
        prediction = '"' + re.search(r"\([a-d]\)", prediction).group(0) + '"'

    # If the prediction is empty, use dummy '0'
    if not prediction:
        prediction = "0"

    # Converting the string answer to a number/list/bool/option
    try:
        prediction = eval(prediction)
    except Exception:
        # extracting the answer with ChatGPT and try again
        prediction = extract_answer(prediction, False)
        try:
            prediction = eval(prediction)
        except Exception:
            tmp = get_decimal_with_wolfram(prediction)
            if tmp is None:
                print("Wolfram Fail: ------------------", prediction)
                prediction = find_numbers_in_string(prediction)
                # If it does not find any number; boil down to base case.
                if prediction is None:
                    prediction = 0
            else:
                prediction = tmp
                print("Wolfram Success: ------------------", prediction)

    # Performing common type conversion
    if isinstance(prediction, (set, tuple)):
        prediction = list(prediction)
        if isinstance(prediction[0], complex):
            prediction = [tmp.real for tmp in prediction]
        elif isinstance(prediction[0], Rational):
            prediction = [float(tmp) for tmp in prediction]
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    else:
        if isinstance(prediction, complex):
            prediction = prediction.real
        elif isinstance(prediction, Rational):
            prediction = float(prediction)

    return prediction
