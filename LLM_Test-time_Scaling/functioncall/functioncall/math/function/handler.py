import json
import time
from parser import extract_answer

from grader import math_equal


def process_results(answer, solution):
    extracted_answer = extract_answer(answer, "math", use_last_number=False)
    extracted_solution = extract_answer(solution, "math", use_last_number=True)

    if extracted_answer is None or extracted_answer.strip() in ["None", "none", ""]:
        retval = 0
    elif extracted_solution is None or extracted_solution.strip() in [
        "None",
        "none",
        "",
    ]:
        retval = 0
    elif math_equal(extracted_answer, extracted_solution, timeout=True):
        retval = 1
    else:
        retval = 0

    return retval, (extracted_answer, extracted_solution)


def handle(event, context):
    answers = event.get("answers", "")
    solutions = event.get("solutions", "")
    query_ids = event.get("query_ids", "")

    results = []
    for answer, solution, query_id in zip(
        answers,
        solutions,
        query_ids,
    ):
        start_time = time.time()
        result = process_results(answer, solution)
        results.append(result)
        print(
            f"query_id: {query_id}, result: {result}, current cost: {(time.time() - start_time) * 1000:.0f} ms"
        )

    return results
