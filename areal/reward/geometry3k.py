# SPDX-License-Identifier: Apache-2.0

import re

from areal.utils import logging

from . import get_math_verify_worker

logger = logging.getLogger("Geometry3KReward")


def extract_answer(pred_str, use_last_number=True):
    matches = re.findall(r"\[([^\]]+)\]", pred_str)
    if matches:
        return matches[-1]

    # Fallback to last number if no bracket format found
    if use_last_number:
        pattern = r"-?\d*\.?\d+"
        nums = re.findall(pattern, pred_str.replace(",", ""))
        if nums:
            return nums[-1]

    return ""


def geometry3k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        sol = extract_answer(str(completions))
        ans = extract_answer(str(answer)) or str(answer)

        if not sol or not ans:
            return 0.0

        worker = get_math_verify_worker()
        return worker.verify(sol, ans)
    except Exception:
        logger.warning("Exception in geometry3k_reward_fn", exc_info=True)
        return 0.0
