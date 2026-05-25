# SYNC: a copy of this file exists at verl/verl/workers/reward_manager/geo_vision_qa.py
# Any changes here must be mirrored there, and vice versa.
import json
import os
import re
import torch
import numpy as np
import logging
from collections import Counter, defaultdict
from verl import DataProto
from verl.workers.reward_manager import register
 
logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"</think>", re.IGNORECASE)
_ACTION_RE = re.compile(r"<action>.*?</action>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str:
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    think_match = re.search(r"</think>\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        return think_match.group(1).strip()
    return ""


_YES_VARIANTS = frozenset({"yes", "yeah", "yep", "yup", "true", "correct"})
_NO_VARIANTS = frozenset({"no", "nope", "nah", "false", "incorrect"})
_NA_VARIANTS = frozenset({"n/a", "na", "none", "not available", "not applicable"})


def _normalize_answer(text: str) -> str:
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    return text.strip().lower().rstrip(".")


def _parse_number(s: str):
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_items_as_set(text: str) -> frozenset | None:
    items = [item.strip() for item in text.split(",")]
    items = [i for i in items if i]
    return frozenset(items) if items else None


def _parse_range(text: str):
    m = re.match(r"^([\d.]+)%?\s*[-–]\s*([\d.]+)%?$", text.strip())
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except ValueError:
            return None
    return None


def compute_score(prediction: str, ground_truth) -> float:
    if isinstance(ground_truth, list):
        return max(compute_score(prediction, gt) for gt in ground_truth)

    pred_norm = _normalize_answer(prediction)
    gt_norm = _normalize_answer(str(ground_truth))

    if pred_norm == gt_norm:
        return 1.0

    if gt_norm in _YES_VARIANTS and pred_norm in _YES_VARIANTS:
        return 1.0
    if gt_norm in _NO_VARIANTS and pred_norm in _NO_VARIANTS:
        return 1.0
    if gt_norm in _NA_VARIANTS and pred_norm in _NA_VARIANTS:
        return 1.0

    if "," in gt_norm:
        gt_set = _parse_items_as_set(gt_norm)
        pred_set = _parse_items_as_set(pred_norm)
        if gt_set and pred_set and gt_set == pred_set:
            return 1.0

    gt_num = _parse_number(gt_norm)
    pred_num = _parse_number(pred_norm)
    if gt_num is not None and pred_num is not None:
        if abs(gt_num - pred_num) < 1e-6:
            return 1.0

    gt_range = _parse_range(gt_norm)
    pred_range = _parse_range(pred_norm)
    if gt_range is not None and pred_range is not None:
        if gt_range == pred_range:
            return 1.0

    return 0.0


def _compute_reasonmap_plus_score(prediction: str, ground_truth, qtype: str = "") -> float:
    """ReasonMap Plus: TorF → Yes/No normalisation, Counting MCQ → ABCD/index, Counting → numeric."""
    if not prediction:
        return 0.0

    gt_str = str(ground_truth).strip()
    pred = prediction.strip().lower()
    pred = re.sub(r"\\boxed\{(.*?)\}", r"\1", pred)

    if qtype.startswith("TorF"):
        try:
            gt_normalised = "yes" if int(gt_str) == 1 else "no"
        except (ValueError, TypeError):
            gt_normalised = gt_str.lower()
        return 1.0 if pred == gt_normalised else 0.0

    if qtype == "Counting1":
        _LETTER_TO_IDX = {"a": "0", "b": "1", "c": "2", "d": "3"}
        _IDX_TO_LETTER = {v: k for k, v in _LETTER_TO_IDX.items()}
        pred_normalised = _LETTER_TO_IDX.get(pred, pred)
        gt_normalised = _LETTER_TO_IDX.get(gt_str.lower(), gt_str)
        if pred_normalised == gt_normalised:
            return 1.0
        gt_as_letter = _IDX_TO_LETTER.get(gt_str, gt_str.lower())
        if pred == gt_as_letter:
            return 1.0
        return 0.0

    return compute_score(prediction, ground_truth)


def _compute_reasonmap_base_score(prediction: str, ground_truth, extra: dict) -> tuple[float, str]:
    """ReasonMap base: route topology verification. Returns (score, failure_reason)."""
    from geo_edit.evaluation.reason_map_verifier import reason_map_score

    station_1 = extra.get("station_1", "")
    station_2 = extra.get("station_2", "")
    metro_raw = extra.get("metro_data", {})

    if isinstance(metro_raw, str):
        try:
            metro_data = json.loads(metro_raw)
        except (json.JSONDecodeError, TypeError):
            return 0.0, "metro_data_parse_failed"
    else:
        metro_data = metro_raw if isinstance(metro_raw, dict) else {}

    if not station_1 or not station_2 or not metro_data:
        gt_str = str(ground_truth).strip()
        pred_lower = prediction.strip().lower()
        try:
            gt_data = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
            if isinstance(gt_data, dict):
                total_routes = 0
                for routes in gt_data.values():
                    if not isinstance(routes, list):
                        continue
                    for route in routes:
                        total_routes += 1
                        rn = route.get("route_name", "").lower()
                        dep = route.get("departure_stop", "").lower()
                        arr = route.get("arrival_stop", "").lower()
                        if not (rn and dep and arr and rn in pred_lower and dep in pred_lower and arr in pred_lower):
                            return 0.0, "no_metro_data_fallback_failed"
                if total_routes > 0:
                    return 1.0, "valid"
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return compute_score(prediction, ground_truth), "no_metro_data_text_match"

    try:
        score, reason = reason_map_score(prediction, station_1, station_2, metro_data)
    except Exception:
        return 0.0, "exception"
    return score, reason


def _compute_cartomapqa_score(prediction: str, ground_truth, data_source: str) -> float:
    try:
        from geo_edit.evaluation.cartomapqa.extractors import extract_structured
        from geo_edit.evaluation.cartomapqa.metrics import mml_match, name_listing_prf1
    except ImportError:
        return compute_score(prediction, ground_truth)

    gt = str(ground_truth).strip()

    if data_source == "cartomapqa_mfs":
        pred = extract_structured("cartomapqa_mfs", prediction)
        return 1.0 if pred and pred == gt.upper() else 0.0

    if data_source == "cartomapqa_stmf_presence":
        pred = extract_structured("cartomapqa_stmf_presence", prediction)
        return 1.0 if pred and pred == gt.lower() else 0.0

    if data_source == "cartomapqa_stmf_counting":
        pred = extract_structured("cartomapqa_stmf_counting", prediction)
        try:
            gt_val = int(gt)
        except (ValueError, TypeError):
            return 0.0
        if pred is None:
            return 0.0
        return 1.0 if abs(pred - gt_val) <= 1 else 0.0

    if data_source == "cartomapqa_stmf_name_listing":
        pred_names = extract_structured("cartomapqa_stmf_name_listing", prediction)
        gt_names = [n.strip() for n in gt.split("\n") if n.strip()]
        if not pred_names:
            return 0.0
        m = name_listing_prf1(gt_names, pred_names)
        return m["f1"]

    if data_source == "cartomapqa_mml":
        pred_data = extract_structured("cartomapqa_mml", prediction)
        if not pred_data:
            return 0.0
        try:
            import json as _json
            gt_data = _json.loads(gt)
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if mml_match(gt_data.get("road_1", ""), gt_data.get("road_2", ""),
                                 pred_data.get("road_1", ""), pred_data.get("road_2", "")) else 0.0

    if data_source == "cartomapqa_rle":
        pred_data = extract_structured("cartomapqa_rle", prediction)
        if not pred_data:
            return 0.0
        import re as _re
        gt_match = _re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)", gt)
        if not gt_match:
            return 0.0
        gt_val = float(gt_match.group(1).replace(",", ""))
        pred_val = pred_data["value"]
        if gt_val == 0:
            return 1.0 if pred_val == 0 else 0.0
        return 1.0 if abs(pred_val - gt_val) / abs(gt_val) <= 0.15 else 0.0

    if data_source == "cartomapqa_srn":
        from geo_edit.evaluation.cartomapqa.metrics import normalize_route, route_eval
        pred_route = extract_structured("cartomapqa_srn", prediction)
        gt_route = [item.strip() for item in gt.replace("[", "").replace("]", "").split(",")]
        if not pred_route:
            return 0.0
        gt_norm = normalize_route(gt_route)
        pred_norm = normalize_route(pred_route)
        is_success, correct_steps = route_eval(gt_norm, pred_norm)
        if is_success:
            return 1.0
        return max(0.0, (correct_steps - 1) / (len(gt_norm) - 1)) if len(gt_norm) > 1 else 0.0

    if data_source == "cartomapqa_mtmf":
        pred_data = extract_structured("cartomapqa_mtmf", prediction)
        if not pred_data:
            return 0.0
        try:
            import json as _json
            gt_data = _json.loads(gt)
        except (ValueError, TypeError):
            return 0.0
        total_f1 = 0.0
        count = 0
        for poi_type, gt_info in gt_data.items():
            pred_info = pred_data.get(poi_type, {})
            gt_names = gt_info.get("true_names", gt_info.get("names", []))
            pred_names = [n for n in pred_info.get("names", []) if n.strip()]
            m = name_listing_prf1(gt_names, pred_names)
            total_f1 += m["f1"]
            count += 1
        return total_f1 / count if count > 0 else 0.0

    if data_source == "mapeval_visual":
        pred_norm = _normalize_answer(prediction)
        gt_norm = _normalize_answer(gt)
        if pred_norm == gt_norm:
            return 1.0
        try:
            if int(float(pred_norm)) == int(float(gt_norm)):
                return 1.0
        except (ValueError, TypeError):
            pass
        return 0.0

    return compute_score(prediction, ground_truth)


def _compute_map_trace_score(
    response: str,
    ground_truth,
    lo: float = 0.5,
    hi: float = 1.0,
    right_edge: float = 0.5,
) -> tuple[float, float]:
    """MapTrace: piecewise-linear reward in NDTW with hard cliff at ``hi``.

    NDTW is the cumulative DTW cost (lower=better, range [0, inf)).

      ndtw <= lo                    -> reward = 1.0
      lo  <  ndtw <= hi             -> reward linearly interpolates from 1.0 down to ``right_edge``
      ndtw >  hi                    -> reward = 0.0          (hard cliff from ``right_edge`` to 0)
      parse failure / not_success   -> reward = 0.0
    """
    from geo_edit.evaluation.map_trace_verifier import map_trace_score

    try:
        ndtw, is_success, _ = map_trace_score(response, str(ground_truth))
    except (ValueError, TypeError):
        return 0.0, -1.0
    if not is_success:
        return 0.0, -1.0
    if ndtw <= lo:
        return 1.0, ndtw
    if ndtw > hi:
        return 0.0, ndtw
    return right_edge + (1.0 - right_edge) * (hi - ndtw) / (hi - lo), ndtw


def _compute_repetition_penalty(text: str, num_turns: int = 2, disable_sentence_penalty: bool = False) -> float:
    if not text or len(text) < 20:
        return 0.0

    num_turns = max(num_turns, 2)

    if re.search(r"(.)\1{49,}", text):
        return -3.0

    if re.search(r"\b(\w+)(?:\s+\1){19,}\b", text):
        return -3.0

    if re.search(r"(\b\w+(?:\s+\w+){3,})(?:\s+\1){9,}", text):
        return -2.0

    if not disable_sentence_penalty:
        sentences = re.split(r"[.!?\n]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if sentences:
            counts = Counter(sentences)
            max_count = counts.most_common(1)[0][1] if counts else 0
            if max_count >= 10:
                return -1.5 * 2 / num_turns
            if max_count >= 7:
                return -1.0 * 2 / num_turns

    if re.search(r"\b(\w+)(?:\s+\1){9,}\b", text):
        return -1.5

    return 0.0


def _compute_format_reward(text: str) -> float:
    """R_format = (𝕀{format_ok} - 0.5) × 2 → {-1.0, 0.0, +1.0}.

    Incomplete trajectories (has <action> but no <answer>) get 0.0 instead of
    -1.0, because the model was still mid-conversation when the token budget
    ran out — it never had a chance to produce <answer>.
    """
    has_think = bool(_THINK_RE.search(text))
    has_answer = bool(_ANSWER_RE.search(text))
    has_action = bool(_ACTION_RE.search(text))

    if not has_think:
        return -1.0

    if not has_answer:
        # Incomplete trajectory: model called tools but was truncated before
        # it could produce <answer>. Don't penalise.
        if has_action:
            return 0.0
        return -1.0

    # If there are <action> blocks, they must be preceded by <think> blocks
    action_blocks = _ACTION_RE.findall(text)
    if action_blocks:
        think_positions = [m.end() for m in _THINK_RE.finditer(text)]
        action_positions = [m.start() for m in _ACTION_RE.finditer(text)]
        for apos in action_positions:
            if not any(tpos <= apos for tpos in think_positions):
                return -1.0

    return 1.0


@register("geo_vision_qa")
class GeoVisionQARewardManager:
    name = "geo_vision_qa"

    def __init__(self, config=None, tokenizer=None, num_examine=3, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.config = config
        self.disable_sentence_rep_penalty = False
        if config is not None:
            self.disable_sentence_rep_penalty = getattr(config, "disable_sentence_rep_penalty", False)

        api_key = os.environ.get("JUDGE_API_KEY")
        if api_key:
            try:
                from geo_edit.evaluation.trajectory_judge import TrajectoryJudge
                self.judge = TrajectoryJudge(
                    api_key=api_key,
                    model=os.environ.get("JUDGE_MODEL", "gpt-5-mini-2025-08-07"),
                    api_base=os.environ.get("JUDGE_API_BASE"),
                )
                logger.info("LLM judge enabled (model=%s)", self.judge.model)
            except ImportError:
                logger.warning("TrajectoryJudge import failed, LLM judge disabled")
                self.judge = None
        else:
            self.judge = None

    def _score_correctness(self, prediction: str, ground_truth, data_source: str, data_item) -> float:
        if not prediction:
            return 0.0

        if data_source == "reason_map_plus":
            reward_model = data_item.non_tensor_batch.get("reward_model", {})
            qtype = ""
            if isinstance(reward_model, dict):
                extra = reward_model.get("extra", {})
                if isinstance(extra, dict):
                    qtype = extra.get("type", "")
            return _compute_reasonmap_plus_score(prediction, ground_truth, qtype)

        if data_source == "reason_map":
            reward_model = data_item.non_tensor_batch.get("reward_model", {})
            extra = {}
            if isinstance(reward_model, dict):
                extra = reward_model.get("extra", {})
                if not isinstance(extra, dict):
                    extra = {}
            score, reason = _compute_reasonmap_base_score(prediction, ground_truth, extra)
            self._last_reasonmap_reason = reason
            return score

        if data_source == "map_trace":
            score, _ = _compute_map_trace_score(prediction, ground_truth)
            return score

        if data_source.startswith("cartomapqa_") or data_source == "mapeval_visual":
            return _compute_cartomapqa_score(prediction, ground_truth, data_source)

        return compute_score(prediction, ground_truth)

    def _llm_judge_fallback(self, prompt_str: str, ground_truth, prediction: str, max_retries: int = 6) -> bool:
        if self.judge is None:
            return False
        import time, random
        for attempt in range(max_retries):
            try:
                is_correct, _ = self.judge.judge_correctness(
                    question=prompt_str,
                    ground_truth=str(ground_truth),
                    prediction=prediction,
                )
                return is_correct
            except Exception as e:
                err_str = str(e)
                is_retryable = "429" in err_str or "rate_limit" in err_str.lower() or "connection" in err_str.lower() or "timeout" in err_str.lower()
                if is_retryable and attempt < max_retries - 1:
                    delay = min(2 ** attempt + random.random(), 16)
                    logger.warning("LLM judge call failed (attempt %d/%d), retrying in %.1fs: %s", attempt + 1, max_retries, delay, e)
                    time.sleep(delay)
                else:
                    logger.warning("LLM judge call failed (attempt %d/%d), giving up: %s", attempt + 1, max_retries, e)
                    return False
        return False

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_printed = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            reward_model = data_item.non_tensor_batch.get("reward_model")
            if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
                logger.warning("Sample %d missing ground_truth, skipping reward calculation", i)
                continue
            ground_truth = reward_model["ground_truth"]
            if ground_truth is None:
                logger.warning("Sample %d has None ground_truth, skipping reward calculation", i)
                continue
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")

            num_turns = data_item.non_tensor_batch.get("__num_turns__", 2)
            if hasattr(num_turns, '__len__'):
                num_turns = int(num_turns[0]) if len(num_turns) > 0 else 2
            model_only_text = re.sub(r"<\|im_start\|>user\n.*?<\|im_end\|>", "", response_str, flags=re.DOTALL)
            r_rep = _compute_repetition_penalty(model_only_text, int(num_turns), self.disable_sentence_rep_penalty)

            # R_format: format compliance
            r_format = _compute_format_reward(response_str)

            # R_correct: answer correctness (dataset-specific)
            prediction = extract_answer(response_str)
            accuracy = self._score_correctness(prediction, ground_truth, data_source, data_item)

            # LLM judge fallback: rule-based says wrong → ask LLM
            # Skip for map_trace: GT is coordinate paths, LLM judge can't evaluate
            llm_called = 0.0
            llm_overturned = 0.0
            is_ood = data_source.startswith("cartomapqa_") or data_source.startswith("mapeval_")
            judge_threshold = (accuracy != 1.0) if is_ood else (accuracy == 0.0)
            if judge_threshold and prediction and self.judge is not None and data_source != "map_trace":
                llm_called = 1.0
                if data_source == "reason_map":
                    reason = getattr(self, "_last_reasonmap_reason", "")
                    judge_gt = (
                        f"Rule-based verifier failed: {reason}\n"
                        f"Ground truth route: {ground_truth}\n"
                        "Compare ONLY: route names, departure/arrival stations, and transfer connections.\n"
                        "Ignore minor differences like 'Station' suffix, '站' suffix, 'Line' vs '号线' naming.\n"
                        "If the predicted route is topologically equivalent to the ground truth, answer YES."
                    )
                    if self._llm_judge_fallback(prompt_str, judge_gt, prediction):
                        accuracy = 1.0
                        llm_overturned = 1.0
                elif data_source == "cartomapqa_srn":
                    judge_gt = (
                        f"Ground truth route: {ground_truth}\n"
                        f"Predicted route: {prediction}\n"
                        "Evaluation rules:\n"
                        "- 'road_1, continue straight, road_1' is equivalent to 'road_1' only. "
                        "Redundant straight continuations on the same road should be ignored.\n"
                        "- The direction sequence matters: compare step by step.\n"
                        "- Road name synonyms or abbreviations (e.g. 'St' vs 'Street') are acceptable.\n"
                        "If the predicted route is equivalent to the ground truth, answer YES."
                    )
                    if self._llm_judge_fallback(prompt_str, judge_gt, prediction):
                        accuracy = 1.0
                        llm_overturned = 1.0
                elif data_source == "cartomapqa_mml":
                    judge_gt = (
                        f"Ground truth: {ground_truth}\n"
                        f"Predicted: {prediction}\n"
                        "Evaluation rules:\n"
                        "- The answer is a JSON with 'road_1' and 'road_2' fields.\n"
                        "- The order of road_1 and road_2 does NOT matter: "
                        "{road_1: A, road_2: B} is equivalent to {road_1: B, road_2: A}.\n"
                        "- Road name abbreviations (e.g. 'Rd' vs 'Road', 'St' vs 'Street') are acceptable.\n"
                        "If the predicted answer matches the ground truth, answer YES."
                    )
                    if self._llm_judge_fallback(prompt_str, judge_gt, prediction):
                        accuracy = 1.0
                        llm_overturned = 1.0
                elif data_source == "cartomapqa_stmf_name_listing":
                    judge_gt = (
                        f"Ground truth names: {ground_truth}\n"
                        f"Predicted names: {prediction}\n"
                        "Evaluation rules:\n"
                        "- The answer is a list of POI names.\n"
                        "- The order of names does NOT matter.\n"
                        "- Minor spelling variations or abbreviations are acceptable.\n"
                        "If the predicted names match the ground truth, answer YES."
                    )
                    if self._llm_judge_fallback(prompt_str, judge_gt, prediction):
                        accuracy = 1.0
                        llm_overturned = 1.0
                else:
                    if self._llm_judge_fallback(prompt_str, ground_truth, prediction):
                        accuracy = 1.0
                        llm_overturned = 1.0

            r_correct = accuracy if data_source == "map_trace" else (1.0 if accuracy > 0 else 0.0)

            # R(U) = R_rep + R_format + R_correct
            reward = r_rep + r_format + r_correct

            ndtw_val = 0.0
            ndtw_success = 0.0
            ndtw_count = 0.0
            if data_source == "map_trace":
                _, raw_ndtw = _compute_map_trace_score(prediction, ground_truth)
                if raw_ndtw >= 0:
                    ndtw_val = raw_ndtw
                    ndtw_success = 1.0 if raw_ndtw < 1.0 else 0.0
                    ndtw_count = 1.0
            reward_extra_info["ndtw"].append(ndtw_val)
            reward_extra_info["ndtw_success"].append(ndtw_success)
            reward_extra_info["ndtw_count"].append(ndtw_count)

            rle_m_error = 0.0
            rle_m_count = 0.0
            rle_ft_error = 0.0
            rle_ft_count = 0.0
            counting_sq_error = 0.0
            counting_count = 0.0
            srn_step_acc = 0.0
            srn_count = 0.0
            mtmf_counting_sq_error = 0.0
            mtmf_counting_count = 0.0
            mtmf_naming_f1 = 0.0
            mtmf_naming_count = 0.0
            if data_source == "cartomapqa_rle":
                try:
                    from geo_edit.evaluation.cartomapqa.extractors import extract_structured
                    pred_data = extract_structured("cartomapqa_rle", prediction)
                    import re as _re
                    gt_str = str(ground_truth)
                    gt_match = _re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)", gt_str)
                    is_feet = "ft" in gt_str.lower() or "feet" in gt_str.lower()
                    if pred_data and gt_match:
                        gt_val = float(gt_match.group(1).replace(",", ""))
                        pred_val = pred_data["value"]
                        rel_err = abs(pred_val - gt_val) / max(abs(gt_val), 1e-9)
                        if is_feet:
                            rle_ft_error = rel_err
                            rle_ft_count = 1.0
                        else:
                            rle_m_error = rel_err
                            rle_m_count = 1.0
                except Exception:
                    pass
            elif data_source == "cartomapqa_stmf_counting":
                try:
                    from geo_edit.evaluation.cartomapqa.extractors import extract_structured
                    pred_val = extract_structured("cartomapqa_stmf_counting", prediction)
                    gt_val = int(str(ground_truth).strip())
                    if pred_val is not None:
                        counting_sq_error = float((pred_val - gt_val) ** 2)
                        counting_count = 1.0
                except Exception:
                    pass
            elif data_source == "cartomapqa_srn":
                srn_step_acc = accuracy
                srn_count = 1.0
            elif data_source == "cartomapqa_mtmf":
                try:
                    from geo_edit.evaluation.cartomapqa.extractors import extract_structured
                    from geo_edit.evaluation.cartomapqa.metrics import name_listing_prf1
                    pred_data = extract_structured("cartomapqa_mtmf", prediction)
                    gt_data = json.loads(str(ground_truth))
                    if pred_data and isinstance(gt_data, dict):
                        total_sq = 0.0
                        total_f1 = 0.0
                        n = 0
                        for poi_type, gt_info in gt_data.items():
                            pred_info = pred_data.get(poi_type, {})
                            gt_count = gt_info.get("true_count", gt_info.get("count", 0))
                            pred_count = pred_info.get("count", 0)
                            total_sq += (pred_count - gt_count) ** 2
                            gt_names = gt_info.get("true_names", gt_info.get("names", []))
                            pred_names = [nm for nm in pred_info.get("names", []) if nm.strip()]
                            m = name_listing_prf1(gt_names, pred_names)
                            total_f1 += m["f1"]
                            n += 1
                        if n > 0:
                            mtmf_counting_sq_error = total_sq / n
                            mtmf_counting_count = 1.0
                            mtmf_naming_f1 = total_f1 / n
                            mtmf_naming_count = 1.0
                except Exception:
                    pass

            reward_extra_info["rle_m_error"].append(rle_m_error)
            reward_extra_info["rle_m_count"].append(rle_m_count)
            reward_extra_info["rle_ft_error"].append(rle_ft_error)
            reward_extra_info["rle_ft_count"].append(rle_ft_count)
            reward_extra_info["counting_sq_error"].append(counting_sq_error)
            reward_extra_info["counting_count"].append(counting_count)
            reward_extra_info["srn_step_acc"].append(srn_step_acc)
            reward_extra_info["srn_count"].append(srn_count)
            reward_extra_info["mtmf_counting_sq_error"].append(mtmf_counting_sq_error)
            reward_extra_info["mtmf_counting_count"].append(mtmf_counting_count)
            reward_extra_info["mtmf_naming_f1"].append(mtmf_naming_f1)
            reward_extra_info["mtmf_naming_count"].append(mtmf_naming_count)

            reward_extra_info["accuracy"].append(accuracy)
            reward_extra_info["score"].append(reward)
            reward_extra_info["has_answer_tag"].append(1.0 if prediction else 0.0)
            reward_extra_info["r_rep"].append(r_rep)
            reward_extra_info["r_format"].append(r_format)
            reward_extra_info["r_correct"].append(r_correct)
            reward_extra_info["llm_judge_called"].append(llm_called)
            reward_extra_info["llm_judge_overturned"].append(llm_overturned)

            if accuracy > 0:
                reward_extra_info["correct_response_length"].append(valid_response_length)
            else:
                reward_extra_info["wrong_response_length"].append(valid_response_length)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_printed:
                already_printed[data_source] = 0
            should_print = already_printed[data_source] < self.num_examine or np.random.random() < 0.03
            if should_print:
                already_printed[data_source] += 1
                print(f"[data_source] {data_source}")
                prompt_display = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image_base64>", prompt_str)
                if np.random.random() < 0.1:
                    print(f"[prompt] {prompt_display}")
                print(f"[response] {response_str}")
                print(f"[ground_truth] {ground_truth}")
                print(f"[prediction] {prediction}")
                print(f"[r_rep] {r_rep} [r_format] {r_format} [r_correct] {r_correct} [reward] {reward}")
                if llm_overturned:
                    print(f"[llm_judge] overturned to correct")

        correct_len = np.mean(reward_extra_info["correct_response_length"]) if reward_extra_info["correct_response_length"] else 0.0
        wrong_len = np.mean(reward_extra_info["wrong_response_length"]) if reward_extra_info["wrong_response_length"] else 0.0
        reward_extra_info["correct_response_length"] = [correct_len] * len(reward_tensor)
        reward_extra_info["wrong_response_length"] = [wrong_len] * len(reward_tensor)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(sorted(reward_extra_info.items()))}
        return reward_tensor
