# SPDX-License-Identifier: Apache-2.0

import re

from areal.utils import logging

logger = logging.getLogger("CLEVR70KReward")


def extract_answer(pred_str, data_name, use_last_number=True):
    _, opening, tail = pred_str.rpartition("[")
    answer, closing, _ = tail.partition("]")
    answer = answer.strip()
    if (
        opening
        and closing
        and re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", answer)
    ):
        return answer

    return ""


def clevr_count_70k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        sol = extract_answer(str(completions), data_name="")  # str number
        ans = str(answer)

        if not sol or not ans:
            return 0.0

        is_correct = sol == ans.strip()
        if is_correct:
            logger.info(f"completions: {completions}, answer: {answer}")
        return float(is_correct)
    except Exception:
        logger.warning("Exception in clevr_count_70k_reward_fn", exc_info=True)
        return 0.0
