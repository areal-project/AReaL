# SPDX-License-Identifier: Apache-2.0

"""Constraint reward used by the mixed SWE/IF MOPD agent.

The input dataset contains two verifier formats.  ``ifeval_g`` rows use the
``verifiable_instructions`` registry, while ``ifrl_recovery`` rows use the
checkers vendored in IFDataSynthesis.  Both dependencies are optional and are
loaded lazily from the paths supplied by the launch script.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from areal.utils import logging

logger = logging.getLogger("IFGapReward")

STRICT_BONUS = 0.5

_ifeval_registry: dict[str, Any] | None = None
_ifrl_engines: tuple[Any, Any, Any] | None = None


def _load_ifeval_registry() -> dict[str, Any]:
    """Load the generalized IFEval instruction registry lazily."""
    global _ifeval_registry
    if _ifeval_registry is None:
        from verifiable_instructions import instructions_registry

        _ifeval_registry = instructions_registry.INSTRUCTION_DICT
        logger.info(
            "Loaded verifiable_instructions registry with %d entries",
            len(_ifeval_registry),
        )
    return _ifeval_registry


def _load_ifrl_engines() -> tuple[Any, Any, Any]:
    """Load the IFDataSynthesis verifier and recovery checkers lazily."""
    global _ifrl_engines
    if _ifrl_engines is not None:
        return _ifrl_engines

    configured_root = os.getenv("IF_SYNTH_ROOT", "").strip()
    if not configured_root:
        raise FileNotFoundError("IF_SYNTH_ROOT is not configured")
    root = os.path.abspath(configured_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"IF_SYNTH_ROOT does not point to IFDataSynthesis: {root!r}"
        )
    if root not in sys.path:
        sys.path.insert(0, root)

    import verifier as generalized_verifier  # type: ignore[import-not-found]
    from ifrl_recovery import (  # type: ignore[import-not-found]
        bench_map,
        simple_checkers,
    )

    _ifrl_engines = (generalized_verifier, simple_checkers, bench_map)
    logger.info("Loaded IF recovery verifier from %s", root)
    return _ifrl_engines


def extract_visible_answer(completion: str) -> str:
    """Remove model thinking; truncated thinking has no scoreable answer."""
    if "</think>" in completion:
        return completion.split("</think>", 1)[1]
    if "<think>" in completion:
        return ""
    return completion


def _score_ifeval(spec: dict[str, Any], answer: str) -> list[bool]:
    registry = _load_ifeval_registry()
    instruction_ids = spec.get("instruction_id_list") or []
    kwargs_list = spec.get("kwargs") or [{} for _ in instruction_ids]
    results: list[bool] = []
    for instruction_id, instruction_kwargs in zip(
        instruction_ids, kwargs_list, strict=True
    ):
        passed = False
        try:
            instruction = registry[instruction_id](instruction_id)
            params = {
                key: value
                for key, value in (instruction_kwargs or {}).items()
                if value is not None
            }
            instruction.build_description(**params)
            passed = bool(instruction.check_following(answer))
        except KeyError:
            logger.warning(
                "Unknown ifeval_g instruction id (scored as fail): %s",
                instruction_id,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "IFEval instruction failed (scored as fail): %s",
                instruction_id,
                exc_info=True,
            )
        results.append(passed)
    return results


def _score_ifrl_recovery(spec: dict[str, Any], answer: str) -> list[bool]:
    verifier, simple_checkers, bench_map = _load_ifrl_engines()
    if_type = spec.get("if_type")
    is_zh = simple_checkers.is_zh(answer)
    results: list[bool] = []
    for constraint in spec.get("constraints") or []:
        name = constraint.get("constraint_name")
        params = constraint.get("params")
        passed = False
        mapped = (
            bench_map.map_constraint(name, params) if if_type == "if_bench" else None
        )
        if mapped is not None:
            instruction_id, instruction_kwargs = mapped
            full_kwargs = dict(verifier.blank_kwargs())
            full_kwargs.update(instruction_kwargs)
            try:
                passed = bool(
                    verifier.verify([instruction_id], [full_kwargs], answer)[0][1]
                )
            except Exception:  # noqa: BLE001
                passed = False
        elif name in simple_checkers.CHECKERS:
            try:
                passed = bool(simple_checkers.check(name, params, answer, is_zh))
            except Exception:  # noqa: BLE001
                passed = False
        else:
            logger.warning(
                "Unknown ifrl_recovery constraint (scored as fail): %s", name
            )
        results.append(passed)
    return results


def score_if_gap_spec(
    verify_engine: str, spec: dict[str, Any], answer: str
) -> list[bool]:
    """Return one pass/fail result for every constraint in an IF row."""
    if verify_engine == "ifeval_g":
        return _score_ifeval(spec, answer)
    if verify_engine == "ifrl_recovery":
        return _score_ifrl_recovery(spec, answer)
    raise ValueError(f"Unsupported IF verify engine: {verify_engine!r}")


def if_gap_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
    verify_engine: str = "",
    spec: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Score a single IF answer using pass rate plus a strict-pass bonus."""
    del prompt, prompt_ids, completion_ids, kwargs
    try:
        spec = spec or {}
        answer = extract_visible_answer(str(completions)).strip()
        if not answer:
            return 0.0
        results = score_if_gap_spec(verify_engine, spec, answer)
        if not results:
            return 0.0
        pass_rate = sum(results) / len(results)
        strict = float(all(results))
        return float((pass_rate + STRICT_BONUS * strict) / (1.0 + STRICT_BONUS))
    except Exception:  # noqa: BLE001
        logger.warning("Exception in IF gap reward", exc_info=True)
        return 0.0
