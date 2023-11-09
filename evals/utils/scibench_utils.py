import math


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
