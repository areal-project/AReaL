import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import List

from functioncall.base.call import Language, batch_function_call, get_runtime_name
from functioncall.base.utils import construct_uid, logger


def math_verify(
    id2info, generateds: List, query_ids: List, batch_size=10, timeout=1000
) -> List:
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )

    st = time.monotonic()
    query_indices = []
    parameters = []
    # Collect all (generated, solution) pairs with their original indices
    for idx, (query_id, generated) in enumerate(zip(query_ids, generateds)):
        base_query_id = query_id.split("@idx:")[0]
        info = id2info[base_query_id]
        for cur_solution in info["solutions"]:
            parameters.append((generated, cur_solution, idx))
            query_indices.append(idx)

    # Process in batches
    start_time = time.time()
    batch_args_list = []
    for i in range(0, len(parameters), batch_size):
        end_idx = min(i + batch_size, len(parameters))
        answers, solutions, indices = zip(*parameters[i:end_idx])
        batch_args = {
            "answers": list(answers),
            "solutions": list(solutions),
            "query_ids": [query_ids[i] for i in indices],
        }

        sub_problem = {
            "uid": construct_uid("math", i, end_idx),
            "language": str(Language.MATH).upper(),
            "runtime": get_runtime_name(None, str(Language.MATH)),
            "code": 'print("hello math!")',
            "testcases": [{}] * (end_idx - i),  # required filed
            "timeout": 5,
            "isFastFail": True,
            "extraInfo": batch_args,
        }

        batch_args_list.append(sub_problem)

    results_batch = batch_function_call(batch_args_list, "math", timeout)

    labels = [0] * len(query_ids)
    # Map results back to original indices
    index = 0

    for batch_idx, results in enumerate(results_batch):
        # check result format
        if not (
            isinstance(results, dict)
            and "results" in results
            and isinstance(results["results"], list)
            and results["results"]
            and all(isinstance(item, dict) for item in results["results"])
        ):
            index += len(batch_args_list[batch_idx]["extraInfo"]["query_ids"])
            logger.warning(
                f"Invalid functioncall math results: {results}, batch index:{batch_idx}, query index: {query_indices[index]}, params: {batch_args_list[batch_idx]}."
            )
            continue

        for result in results["results"]:
            query_index = query_indices[index]
            # set label as 1 if any of the solutions matches the answer
            labels[query_index] = (
                int(result.get("success", False)) or labels[query_index]
            )
            index += 1

    logger.info(
        f"verify math with query size={len(query_ids)}, takes {time.time() - start_time:.4f} seconds"
    )
    return labels


if __name__ == "__main__":
    sample = {
        "answers": ["\\boxed{-\\frac{2}{3}}"],
        "solutions": [
            "1. **Apply the operation $\\otimes$ to the innermost parentheses first:**\n   \\[\n   (1 \\otimes 2) \\otimes 3 = \\left(\\frac{1^2}{2}\\right) \\otimes 3 = \\frac{1}{2} \\otimes 3\n   \\]\n   \\[\n   1 \\otimes (2 \\otimes 3) = 1 \\otimes \\left(\\frac{2^2}{3}\\right) = 1 \\otimes \\frac{4}{3}\n   \\]\n\n2. **Calculate each part using the definition of $\\otimes$:**\n   \\[\n   \\frac{1}{2} \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n   \\]\n   \\[\n   1 \\otimes \\frac{4}{3} = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n   \\]\n\n3. **Subtract the two results:**\n   \\[\n   \\left(\\frac{1}{12}\\right) - \\left(\\frac{3}{4}\\right) = \\frac{1}{12} - \\frac{9}{12} = -\\frac{8}{12} = -\\frac{2}{3}\n   \\]\n\n4. **Conclude with the final answer:**\n   \\[\n   \\boxed{A}\n   \\]",
            "\\boxed{-\\frac{2}{3}}",
        ],
    }
    id2info = {"fe11b471-1aa9-4867-958f-a0a811c85f92": sample}

    scale = 50
    start_time = time.time()
    result = math_verify(
        id2info,
        sample["answers"] * scale,
        ["fe11b471-1aa9-4867-958f-a0a811c85f92"] * scale,
    )
    print(result)
