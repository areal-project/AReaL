# SPDX-License-Identifier: Apache-2.0

"""SWE SFT loading, processing, distributed caching, and public dataset API."""

import hashlib
import json
import os
import shutil
import time
import uuid

from datasets import Dataset

from areal.utils import logging

from .messages import (
    _clean_message,
    _iter_jsonl_records,
    _prepare_trajectory,
    _split_and_filter,
)
from .tokenization import (
    DATASET_NUM_PROC,
    _dump_samples,
    _patch_chat_template_for_training,
    _require_generation_tracking,
    _TokenizeAndMask,
)

logger = logging.getLogger("SWESFTDataset")

_RANK0_CACHE_TIMEOUT = 36000
_RANK0_CACHE_POLL_INTERVAL = 5
_RANK0_STALE_FAILURE_GRACE = 60
_CACHE_SCHEMA_VERSION = 3
_SWE_SFT_ARTIFACT_MARKER = ".areal_swe_sft.json"
_SWE_SFT_ARTIFACT_FORMAT = "areal.swe_sft.pretokenized"
_SWE_SFT_ARTIFACT_VERSION = 1


def _stable_json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _stable_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    return str(value)


def _json_digest(value) -> str:
    payload = json.dumps(
        _stable_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_identity(path: str) -> dict:
    resolved_path = os.path.realpath(os.path.abspath(path))
    try:
        stat = os.stat(resolved_path)
    except FileNotFoundError:
        return {"path": resolved_path, "missing": True}

    identity = {
        "path": resolved_path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    # Correctness takes precedence over startup I/O: size/mtime plus sampled
    # regions can miss an in-place rewrite that preserves metadata and changes
    # an unsampled offset. Stream the entire source in bounded chunks instead.
    digest = hashlib.sha256()
    with open(resolved_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    identity["digest_kind"] = "sha256"
    identity["digest"] = digest.hexdigest()
    return identity


def _tokenizer_identity(tokenizer) -> dict | None:
    if tokenizer is None:
        return None

    tokenizer_class = type(tokenizer)
    identity = {
        "class": f"{tokenizer_class.__module__}.{tokenizer_class.__qualname__}",
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens": _stable_json_value(
            getattr(tokenizer, "special_tokens_map", None)
        ),
        "chat_template_digest": _json_digest(getattr(tokenizer, "chat_template", None)),
    }
    try:
        vocab = tokenizer.get_vocab()
    except (AttributeError, NotImplementedError):
        vocab = None
    if vocab is not None:
        identity["vocab_digest"] = _json_digest(vocab)
        identity["vocab_entries"] = len(vocab)
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    try:
        backend_serialized = backend_tokenizer.to_str()
    except (AttributeError, TypeError, ValueError):
        backend_serialized = None
    if backend_serialized is not None:
        # Captures fast-tokenizer normalizer, pre-tokenizer, decoder, and model
        # state that vocab size/content alone does not describe.
        identity["backend_tokenizer_digest"] = _json_digest(backend_serialized)
    else:
        identity["init_kwargs_digest"] = _json_digest(
            getattr(tokenizer, "init_kwargs", None)
        )
    return identity


def _build_cache_metadata(path: str, tokenizer, process_kwargs: dict) -> dict:
    return {
        "version": _CACHE_SCHEMA_VERSION,
        "source": _source_identity(path),
        "tokenizer": _tokenizer_identity(tokenizer),
        "process_kwargs": {
            key: value
            for key, value in process_kwargs.items()
            if key not in ("dump_dir", "dump_n_samples")
        },
    }


def _atomic_write_json(path: str, value: dict) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_json_marker(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as marker:
            value = json.load(marker)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _failed_marker_path(cache_dir: str, attempt_id: str) -> str:
    attempt_tag = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:32]
    return f"{cache_dir}.failed.{attempt_tag}.json"


def _building_marker_path(cache_dir: str, attempt_id: str | None) -> str:
    if attempt_id is None:
        # Legacy/direct SPMD without a shared launcher identifier needs a
        # discoverable pointer and retains the bounded stale-marker grace.
        return f"{cache_dir}.building.json"
    attempt_tag = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:32]
    return f"{cache_dir}.building.{attempt_tag}.json"


def _unlink_marker_for_attempt(path: str, attempt_id: str) -> None:
    marker = _read_json_marker(path)
    if marker is None or marker.get("attempt_id") != attempt_id:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _resolve_cache_attempt_id(cache_attempt_id: str | None) -> str | None:
    if cache_attempt_id:
        return cache_attempt_id
    elastic_run_id = os.getenv("TORCHELASTIC_RUN_ID")
    if elastic_run_id:
        restart_count = os.getenv("TORCHELASTIC_RESTART_COUNT", "0")
        return f"TORCHELASTIC_RUN_ID:{elastic_run_id}:restart:{restart_count}"
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id:
        step_id = os.getenv("SLURM_STEP_ID", "unknown")
        restart_count = os.getenv("SLURM_RESTART_COUNT", "0")
        return f"SLURM_JOB_ID:{slurm_job_id}:step:{step_id}:restart:{restart_count}"
    return None


def _cache_entry_path(cache_dir: str, cache_key: str) -> str:
    return os.path.join(cache_dir, "entries", cache_key)


def _cache_coordination_prefix(cache_dir: str, cache_key: str) -> str:
    return os.path.join(cache_dir, ".coord", cache_key, "build")


def _write_swe_sft_artifact_marker(path: str) -> None:
    _atomic_write_json(
        os.path.join(path, _SWE_SFT_ARTIFACT_MARKER),
        {
            "format": _SWE_SFT_ARTIFACT_FORMAT,
            "version": _SWE_SFT_ARTIFACT_VERSION,
        },
    )


def _validate_split_mode(split_mode: str) -> None:
    if split_mode not in ("pair", "trajectory"):
        raise ValueError(
            f"split_mode must be either 'pair' or 'trajectory', got {split_mode!r}"
        )


def _has_supervised_tokens(input_ids, loss_mask) -> bool:
    return len(input_ids) > 0 and len(input_ids) == len(loss_mask) and any(loss_mask)


def _filter_trainable_samples(
    dataset,
    *,
    num_proc: int | None,
    context: str,
    keep_in_memory: bool = False,
):
    required = {"input_ids", "loss_mask"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"{context} is missing required columns: {sorted(missing)}")

    before = len(dataset)
    dataset = dataset.filter(
        _has_supervised_tokens,
        input_columns=["input_ids", "loss_mask"],
        num_proc=num_proc,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=False,
    )
    removed = before - len(dataset)
    if removed:
        logger.info(
            "Filtered %d samples without a valid supervised-token mask from %s",
            removed,
            context,
        )
    if len(dataset) == 0:
        raise ValueError(
            f"{context} has no samples with a non-empty, length-aligned "
            "loss_mask containing at least one supervised token"
        )
    return dataset


def _resolve_cache_topology(
    cache_rank: int | None,
    cache_world_size: int | None,
) -> tuple[int, int]:
    if (cache_rank is None) != (cache_world_size is None):
        raise ValueError(
            "cache_rank and cache_world_size must either both be provided or "
            "both be omitted"
        )
    if cache_rank is None:
        cache_rank = int(os.getenv("RANK", "0"))
        cache_world_size = int(os.getenv("WORLD_SIZE", "1"))

    assert cache_world_size is not None
    if cache_world_size < 1:
        raise ValueError(f"cache_world_size must be positive, got {cache_world_size}")
    if cache_rank < 0 or cache_rank >= cache_world_size:
        raise ValueError(
            f"cache_rank must be in [0, {cache_world_size}), got {cache_rank}"
        )
    return cache_rank, cache_world_size


def _load_trajectory_pairs(
    path: str,
    filter_errors: bool = True,
    strip_all_thinking: bool = False,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
):
    """Load trajectory JSONL and split into progressive pairs.

    Supports nested (``conversations`` wrapper) and flat JSONL formats
    (auto-detected per record via ``_iter_jsonl_records``).

    Returns:
        Tuple of ``(all_pairs, tools)`` where *tools* is ``None`` when no
        tool definitions are found.
    """
    all_pairs = []
    all_tools = []
    records_in = 0
    total_filtered_errors = 0
    total_filtered_empty_tc = 0
    total_filtered_bare_tc = 0

    for record_idx, messages, record_tools in _iter_jsonl_records(path):
        records_in = record_idx
        pairs, n_err, n_empty_tc, n_bare_tc = _split_and_filter(
            messages,
            filter_errors=filter_errors,
            strip_all_thinking=strip_all_thinking,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        )
        total_filtered_errors += n_err
        total_filtered_empty_tc += n_empty_tc
        total_filtered_bare_tc += n_bare_tc
        all_pairs.extend(pairs)
        all_tools.extend([record_tools] * len(pairs))

    # Log extracted tools summary.
    n_with_tools = sum(1 for t in all_tools if t is not None)
    if n_with_tools > 0:
        all_tool_names = set()
        for t_list in all_tools:
            if t_list is not None:
                for t in t_list:
                    all_tool_names.add(t.get("function", {}).get("name", "?"))
        logger.info(
            f"Extracted tools from {n_with_tools}/{len(all_tools)} pairs: "
            f"{sorted(all_tool_names)}"
        )

    filter_parts = []
    if total_filtered_errors:
        filter_parts.append(f"{total_filtered_errors} with tool errors")
    if total_filtered_empty_tc:
        filter_parts.append(f"{total_filtered_empty_tc} empty-content tool calls")
    if total_filtered_bare_tc:
        filter_parts.append(f"{total_filtered_bare_tc} bare-text tool calls")
    filter_msg = ", ".join(filter_parts) if filter_parts else "none"

    logger.info(
        f"Loaded {records_in} trajectories, "
        f"generated {len(all_pairs)} pairs "
        f"(filtered: {filter_msg})"
    )

    return all_pairs, all_tools


def _load_presplit_pairs(
    path: str,
    strip_all_thinking: bool = False,
):
    """Load pre-split pair JSONL where each line is ``{"messages": [...]}``.

    Messages are cleaned but no splitting or error-filtering is performed.
    By default, thinking is stripped from context assistant turns but
    preserved for the last assistant turn (the training target).  Set
    *strip_all_thinking* to strip from every assistant turn.

    Also extracts per-record ``tools`` definitions so that each pair
    carries its own tools, same as ``_load_trajectory_pairs``.

    Returns:
        Tuple of ``(all_pairs, all_tools)`` where *all_tools* is a
        parallel list of per-sample tool definitions (may be ``None``).
    """
    all_pairs = []
    all_tools = []

    def _build_pair(messages, last_asst):
        pair = []
        for idx, m in enumerate(messages):
            is_target = m.get("role") == "assistant" and idx == last_asst
            strip = strip_all_thinking or not is_target
            pair.append(_clean_message(m, strip_thinking=strip))
        return pair

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if not messages:
                continue

            record_tools = record.get("tools")

            # Find the last assistant index so we can preserve its thinking.
            last_asst = None
            for i, m in enumerate(messages):
                if m.get("role") == "assistant":
                    last_asst = i

            all_pairs.append(_build_pair(messages, last_asst))
            all_tools.append(record_tools)

    # Log extracted tools summary.
    n_with_tools = sum(1 for t in all_tools if t is not None)
    if n_with_tools > 0:
        all_tool_names = set()
        for t_list in all_tools:
            if t_list is not None:
                for t in t_list:
                    all_tool_names.add(t.get("function", {}).get("name", "?"))
        logger.info(
            f"Extracted tools from {n_with_tools}/{len(all_tools)} pairs: "
            f"{sorted(all_tool_names)}"
        )

    logger.info(f"Loaded {len(all_pairs)} pre-split pairs from {path}")
    return all_pairs, all_tools


def _load_full_trajectories(
    path: str,
    filter_errors: bool = True,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
):
    """Load trajectory JSONL for trajectory-level training.

    Each trajectory becomes a single training sample with all assistant
    turns as targets (``loss_mask=1``).  When *filter_errors* is True,
    assistant segments with error tool responses are identified so
    tokenization can mask them (``loss_mask=0``) instead of discarding
    the entire trajectory.

    Supports nested (``conversations`` wrapper) and flat JSONL formats
    (auto-detected per record via ``_iter_jsonl_records``).

    Returns:
        Tuple of ``(trajectories, error_indices_list, all_tools)`` where
        *trajectories* is a list of cleaned message lists,
        *error_indices_list* is a list of error segment index lists,
        and *all_tools* is a parallel list of per-sample tool definitions.
    """
    trajectories = []
    error_indices_list = []
    all_tools = []
    records_in = 0
    total_masked_errors = 0
    total_masked_empty_tc = 0
    total_masked_bare_tc = 0

    for record_idx, messages, record_tools in _iter_jsonl_records(path):
        records_in = record_idx
        result = _prepare_trajectory(
            messages,
            filter_errors=filter_errors,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        )
        if result is None:
            continue
        cleaned, masked_idxs, n_err, n_empty_tc, n_bare_tc = result
        trajectories.append(cleaned)
        error_indices_list.append(masked_idxs)
        all_tools.append(record_tools)
        total_masked_errors += n_err
        total_masked_empty_tc += n_empty_tc
        total_masked_bare_tc += n_bare_tc

    # Log extracted tools summary.
    n_with_tools = sum(1 for t in all_tools if t is not None)
    if n_with_tools > 0:
        all_tool_names = set()
        for t_list in all_tools:
            if t_list is not None:
                for t in t_list:
                    all_tool_names.add(t.get("function", {}).get("name", "?"))
        logger.info(
            f"Extracted tools from {n_with_tools}/{len(all_tools)} "
            f"trajectories: {sorted(all_tool_names)}"
        )

    parts = []
    if total_masked_errors:
        parts.append(f"{total_masked_errors} with tool errors")
    if total_masked_empty_tc:
        parts.append(f"{total_masked_empty_tc} empty-content tool calls")
    if total_masked_bare_tc:
        parts.append(f"{total_masked_bare_tc} bare-text tool calls")
    mask_msg = ", ".join(parts) if parts else "none"

    logger.info(
        f"Loaded {records_in} trajectories, "
        f"kept {len(trajectories)} for training "
        f"(masked: {mask_msg})"
    )

    return trajectories, error_indices_list, all_tools


def _tokenize_samples(
    messages_list,
    tools_list,
    tokenizer,
    *,
    split_mode: str = "pair",
    error_indices_list: list | None = None,
    max_length: int | None = None,
    num_proc: int | None = None,
    no_tools: bool = False,
    dump_dir: str | None = None,
    dump_n_samples: int = 0,
    parse_tool_call_args: bool = False,
):
    """Tokenize message lists into a training-ready Dataset.

    Works for both progressive pairs (``split_mode="pair"``) and
    full trajectories (``split_mode="trajectory"``).

    In pair mode, only the last assistant turn per sample gets
    ``loss_mask=1``.  In trajectory mode, all assistant turns get
    ``loss_mask=1`` except those at error segment indices.

    Args:
        tools_list: Per-sample tool definitions (parallel to
            *messages_list*).  Each element is either ``None`` or a
            list of tool dicts.
    """
    _validate_split_mode(split_mode)
    if num_proc is None:
        num_proc = max(1, min(os.cpu_count() or 1, DATASET_NUM_PROC))

    # Find representative tools for template detection.
    first_tools = None
    if tools_list:
        first_tools = next((t for t in tools_list if t is not None), None)

    if no_tools:
        tools_list = None
        first_tools = None
        logger.info("Tool definitions disabled (no_tools=True)")
    elif first_tools is not None:
        all_tool_names = set()
        for t_list in tools_list:
            if t_list is not None:
                for t in t_list:
                    all_tool_names.add(t.get("function", {}).get("name", "?"))
        logger.info(f"Using tools for chat template: {sorted(all_tool_names)}")

    if not messages_list:
        raise ValueError("No valid samples to tokenize")

    # Build dataset columns.
    data = {"messages": messages_list}
    # Serialize per-sample tools as JSON strings for the Dataset column.
    data["tools_json"] = (
        [json.dumps(t) if t else "" for t in tools_list]
        if tools_list
        else [""] * len(messages_list)
    )
    remove_cols = ["messages", "tools_json"]
    if split_mode == "trajectory":
        data["error_indices"] = error_indices_list or [[] for _ in messages_list]
        remove_cols.append("error_indices")

    dataset = Dataset.from_dict(data)
    _patch_chat_template_for_training(tokenizer)
    _require_generation_tracking(tokenizer)

    # Dump samples for inspection before the heavy map() pass.
    if dump_dir and dump_n_samples != 0:
        _dump_samples(
            messages_list,
            tokenizer,
            tools_list,
            dump_dir,
            dump_n_samples,
            split_mode=split_mode,
            error_indices_list=error_indices_list,
            parse_tool_call_args=parse_tool_call_args,
        )

    process_fn = _TokenizeAndMask(
        tokenizer,
        max_length=max_length,
        split_mode=split_mode,
        parse_tool_call_args=parse_tool_call_args,
    )

    dataset = dataset.map(process_fn, num_proc=num_proc).remove_columns(remove_cols)

    # Remove template failures, overlength rows, malformed masks, and rows with
    # no supervised token before they can reach the SFT loss normalizer.
    dataset = _filter_trainable_samples(
        dataset,
        num_proc=num_proc,
        context="freshly tokenized SWE dataset",
    )

    logger.info(f"Final dataset: {len(dataset)} samples")
    return dataset


def _process_swe_sft(
    path: str,
    tokenizer,
    *,
    max_length: int | None = None,
    num_proc: int | None = None,
    pre_split: bool = False,
    filter_errors: bool = True,
    strip_all_thinking: bool = False,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
    no_tools: bool = False,
    split_mode: str = "pair",
    dump_dir: str | None = None,
    dump_n_samples: int = 0,
    parse_tool_call_args: bool = False,
):
    """Load JSONL, split into pairs, tokenize, and filter.

    Combines file loading with ``_tokenize_samples`` so that the rank-0-only
    path and the single-process path share the same logic.

    When *split_mode* is ``"trajectory"``, the full trajectory is kept as a
    single training sample with all assistant turns as targets.
    """
    _validate_split_mode(split_mode)
    error_indices_list = None

    if split_mode == "trajectory":
        messages_list, error_indices_list, tools_list = _load_full_trajectories(
            path,
            filter_errors=filter_errors,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        )
    elif pre_split:
        messages_list, tools_list = _load_presplit_pairs(
            path,
            strip_all_thinking=strip_all_thinking,
        )
    else:
        messages_list, tools_list = _load_trajectory_pairs(
            path,
            filter_errors=filter_errors,
            strip_all_thinking=strip_all_thinking,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        )

    return _tokenize_samples(
        messages_list,
        tools_list,
        tokenizer,
        split_mode=split_mode,
        error_indices_list=error_indices_list,
        max_length=max_length,
        num_proc=num_proc,
        no_tools=no_tools,
        dump_dir=dump_dir,
        dump_n_samples=dump_n_samples,
        parse_tool_call_args=parse_tool_call_args,
    )


def get_swe_sft_dataset(
    path: str,
    split: str | None = None,
    tokenizer=None,
    max_length: int | None = None,
    num_proc: int | None = None,
    pre_split: bool = False,
    filter_errors: bool = True,
    strip_all_thinking: bool = False,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
    no_tools: bool = False,
    skip_pretokenized_filter: bool = False,
    split_mode: str = "pair",
    cache_dir: str | None = None,
    dump_dir: str | None = None,
    dump_samples: int = 0,
    parse_tool_call_args: bool = False,
    cache_rank: int | None = None,
    cache_world_size: int | None = None,
    cache_attempt_id: str | None = None,
):
    """Load SWE trajectory data and convert to SFT training pairs.

    By default, tool definitions are auto-extracted from the training data's
    ``conversations[].tools`` field and passed to ``apply_chat_template``
    so that the tokenizer renders tool definitions in the system prompt
    (e.g. Qwen3 ``# Tools`` block), matching the eval-time format.
    Set *no_tools* to skip this and render without tool definitions.

    When *split_mode* is ``"trajectory"``, the full trajectory is kept as a
    single training sample with all assistant turns as targets
    (``loss_mask=1``).  Error segments are masked (``loss_mask=0``)
    when *filter_errors* is True, instead of being discarded.

    In distributed (SPMD) mode, only rank 0 performs the heavy processing
    (JSONL loading, pair splitting, tokenization) and saves the result as
    an Arrow dataset to *cache_dir*.  Other ranks wait for rank 0 to
    finish and then load the cached dataset directly via memory-mapped I/O.

    Args:
        path: Path to the JSONL file containing SWE trajectories, or a
            directory containing a pre-tokenized Arrow dataset (saved by
            ``python -m areal.dataset.swe_sft --save-tokenized``).
        split: Unused, kept for API compatibility.
        tokenizer: Tokenizer with ``apply_chat_template`` support.
            Not required when loading a pre-tokenized dataset.
        max_length: Max token length.  Longer sequences are filtered out.
        num_proc: Number of parallel workers for tokenization.
            Defaults to ``min(os.cpu_count(), DATASET_NUM_PROC)``.
        pre_split: If True, treat input as pre-split pairs (each line is
            ``{"messages": [...]}``) instead of full trajectories.
        filter_errors: If True (default), discard pairs whose current segment
            contains a tool result with ``is_error=True``.  In trajectory
            mode, sets ``loss_mask=0`` for error segments instead.
            Set to False to keep/train all regardless of tool errors.
        strip_all_thinking: If True, strip ``<think>...</think>`` from every
            assistant turn including the training target.
            Ignored in trajectory mode (thinking is always preserved).
        filter_empty_tool_calls: If True, discard pairs whose training-target
            assistant turn has no text content but has tool_calls.
        filter_bare_text_tool_calls: If True, discard pairs whose
            training-target assistant turn has text without ``<think>``
            tags and has tool_calls.
        no_tools: If True, do not pass tool definitions to
            ``apply_chat_template`` even if the data contains them.
        skip_pretokenized_filter: If True, skip the ``max_length`` filter
            when loading a pre-tokenized dataset.  Useful when the dataset
            was already filtered during pretokenization and you want to
            avoid NFS cache conflicts from concurrent ``dataset.filter()``
            calls across ranks.
        split_mode: ``"pair"`` (default) splits trajectories into
            progressive pairs.  ``"trajectory"`` keeps the full trajectory
            as a single sample — all assistant turns are targets with
            ``loss_mask=1``, error segments are masked instead of filtered.
        cache_dir: Directory to save/load the processed Arrow dataset.
            In distributed mode this is a cache root; each preprocessing
            identity is stored as an immutable Arrow entry under
            ``entries/<cache-key>``. Rank 0 publishes only its entry and other
            ranks load that same key without overwriting unrelated settings.
        dump_dir: Directory to write sample dump files (``.txt`` + ``.json``).
            Only rank 0 writes.  Set to None to disable.
        dump_samples: Number of random samples to dump.  ``-1`` = all,
            ``0`` = disabled.
        parse_tool_call_args: If True, convert OpenAI JSON-string
            ``tool_calls.arguments`` to dicts before ``apply_chat_template``.
            Required by GLM-4.x / GLM-5.x templates; leave at the default
            (False) for Qwen / Llama / Bailing.
        cache_rank: Explicit rank for shared-cache coordination. Data-service
            workers pass their worker rank here. Direct SPMD callers may omit
            it to use ``RANK``.
        cache_world_size: Explicit world size for shared-cache coordination.
            Must be provided together with *cache_rank*. Direct SPMD callers
            may omit it to use ``WORLD_SIZE``.
        cache_attempt_id: Shared launch identifier used to isolate cache-build
            failures across concurrent or restarted launchers. Data-service
            workers receive one from the controller. Direct SPMD callers fall
            back to ``TORCHELASTIC_RUN_ID`` or ``SLURM_JOB_ID`` when available.

    Returns:
        A HuggingFace ``Dataset`` with ``input_ids`` and ``loss_mask`` columns.
    """
    from datasets import load_from_disk

    _validate_split_mode(split_mode)
    rank, world_size = _resolve_cache_topology(cache_rank, cache_world_size)
    resolved_attempt_id = _resolve_cache_attempt_id(cache_attempt_id)

    # Pre-tokenized Arrow dataset: load directly, skip all processing.
    if os.path.isdir(path):
        logger.info(f"Loading pre-tokenized dataset from {path}")
        dataset = load_from_disk(path)

        dataset = _filter_trainable_samples(
            dataset,
            num_proc=num_proc,
            context=f"pre-tokenized SWE dataset at {path}",
            keep_in_memory=True,
        )

        if max_length is not None and not skip_pretokenized_filter:
            before_filter = len(dataset)
            dataset = dataset.filter(
                lambda x: len(x["input_ids"]) <= max_length, num_proc=num_proc
            )
            logger.info(
                f"Filtered {before_filter - len(dataset)} samples "
                f"exceeding max_length={max_length}"
            )
            if len(dataset) == 0:
                raise ValueError(
                    f"pre-tokenized SWE dataset at {path} has 0 samples after "
                    f"max_length={max_length} filtering"
                )

        logger.info(f"Final dataset: {len(dataset)} samples")
        return dataset

    # --- Shared kwargs for _process_swe_sft ---
    effective_dump_dir = dump_dir if world_size == 1 or rank == 0 else None
    process_kwargs = dict(
        max_length=max_length,
        num_proc=num_proc,
        pre_split=pre_split,
        filter_errors=filter_errors,
        strip_all_thinking=strip_all_thinking,
        filter_empty_tool_calls=filter_empty_tool_calls,
        filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        no_tools=no_tools,
        split_mode=split_mode,
        dump_dir=effective_dump_dir,
        dump_n_samples=dump_samples,
        parse_tool_call_args=parse_tool_call_args,
    )

    # --- Distributed rank-0-only processing ---
    if cache_dir is not None and world_size > 1:
        # Cache identity must describe the effective template used by
        # preprocessing, not the unpatched tokenizer state held by the caller.
        # The patch is idempotent, so _tokenize_samples can safely call it again.
        _patch_chat_template_for_training(tokenizer)
        cache_meta = _build_cache_metadata(path, tokenizer, process_kwargs)
        cache_key = _json_digest(cache_meta)
        entry_dir = _cache_entry_path(cache_dir, cache_key)
        done_marker = os.path.join(entry_dir, ".done")
        meta_path = os.path.join(entry_dir, ".meta.json")
        coordination_prefix = _cache_coordination_prefix(cache_dir, cache_key)
        building_marker = _building_marker_path(
            coordination_prefix, resolved_attempt_id
        )

        def _filter_by_max_length(ds):
            if max_length is None:
                return ds
            before = len(ds)
            # Length via arrow list offsets: avoids decoding every row to
            # Python lists, which for long-context datasets costs minutes of
            # startup per rank while (on a validated cache) removing nothing —
            # build-time _TokenizeAndMask already filtered with this max_length.
            import pyarrow.compute as pc

            # ds.data is the underlying arrow table; a freshly built dataset
            # carries an indices mapping (from .filter views) whose row count
            # differs. Materialize the view first (no-op for load_from_disk).
            if getattr(ds, "_indices", None) is not None:
                ds = ds.flatten_indices()
            lengths = pc.list_value_length(ds.data.column("input_ids")).to_pylist()
            keep = [i for i, n in enumerate(lengths) if n <= max_length]
            ds = ds.select(keep)
            if len(ds) < before:
                logger.info(
                    f"Rank {rank}: filtered {before - len(ds)} samples "
                    f"exceeding max_length={max_length}"
                )
            if len(ds) == 0:
                raise ValueError(
                    f"processed dataset at {entry_dir} has 0 samples after "
                    f"max_length={max_length} filtering"
                )
            return ds

        def _load_valid_cache():
            if not os.path.exists(meta_path):
                raise ValueError(f"cached dataset metadata is missing: {meta_path}")
            with open(meta_path) as f:
                cached_meta = json.load(f)
            if cached_meta != cache_meta:
                raise ValueError(
                    f"cached dataset metadata does not match current SWE settings: "
                    f"{meta_path}"
                )
            dataset = load_from_disk(entry_dir)
            if len(dataset) == 0:
                raise ValueError(f"cached dataset is empty: {entry_dir}")
            return dataset

        def _wait_for_valid_cache():
            start = time.monotonic()
            last_error = None

            initial_building = _read_json_marker(building_marker)
            initial_attempt = resolved_attempt_id
            if initial_attempt is None and initial_building is not None:
                initial_attempt = initial_building.get("attempt_id")
            initial_failed = (
                _read_json_marker(
                    _failed_marker_path(coordination_prefix, initial_attempt)
                )
                if initial_attempt
                else None
            )
            initial_failed_attempt = None
            if (
                initial_failed is not None
                and initial_failed.get("cache_key") == cache_key
                and initial_failed.get("attempt_id") == initial_attempt
            ):
                initial_failed_attempt = initial_failed.get("attempt_id")

            while True:
                if os.path.exists(done_marker):
                    try:
                        return _load_valid_cache()
                    except Exception as e:
                        last_error = e

                building = _read_json_marker(building_marker)
                current_attempt = resolved_attempt_id
                if current_attempt is None and building is not None:
                    current_attempt = building.get("attempt_id")
                failed = (
                    _read_json_marker(
                        _failed_marker_path(coordination_prefix, current_attempt)
                    )
                    if current_attempt
                    else None
                )
                matching_failure = (
                    failed is not None
                    and failed.get("cache_key") == cache_key
                    and failed.get("attempt_id") == current_attempt
                )
                if matching_failure:
                    attempt_id = failed["attempt_id"]
                    elapsed = time.monotonic() - start
                    # A complete failure pair already present when this worker
                    # starts may belong to a previous launcher run. Give rank 0
                    # a bounded startup window to atomically replace the
                    # building marker with this run's attempt. Failures first
                    # observed after invocation are current and propagate on
                    # the next poll without waiting for the cache timeout.
                    stale_candidate_in_grace = (
                        resolved_attempt_id is None
                        and attempt_id == initial_failed_attempt
                        and elapsed < _RANK0_STALE_FAILURE_GRACE
                    )
                    if not stale_candidate_in_grace:
                        error_type = failed.get("error_type", "Exception")
                        error = failed.get("error", "unknown preprocessing failure")
                        raise RuntimeError(
                            "Rank 0 failed to build SWE dataset cache "
                            f"at {cache_dir}: {error_type}: {error}"
                        )
                elapsed = time.monotonic() - start
                if elapsed > _RANK0_CACHE_TIMEOUT:
                    raise TimeoutError(
                        f"Waited {_RANK0_CACHE_TIMEOUT}s for rank 0 to rebuild "
                        f"a valid dataset cache at {cache_dir}. Last error: {last_error}"
                    )
                time.sleep(_RANK0_CACHE_POLL_INTERVAL)

        # Fast path: cache from a previous run (or rank 0 already finished).
        if os.path.exists(done_marker):
            if rank == 0:
                try:
                    logger.info(
                        f"Rank {rank}: loading cached processed dataset from {cache_dir}"
                    )
                    dataset = _load_valid_cache()
                    dataset = _filter_by_max_length(dataset)
                    logger.info(f"Final dataset: {len(dataset)} samples")
                    return dataset
                except Exception as e:
                    logger.warning(
                        "Rank 0: invalid processed dataset cache at %s (%s); "
                        "rebuilding it.",
                        cache_dir,
                        e,
                    )
            else:
                try:
                    logger.info(
                        f"Rank {rank}: loading cached processed dataset from {cache_dir}"
                    )
                    dataset = _load_valid_cache()
                    dataset = _filter_by_max_length(dataset)
                    logger.info(f"Final dataset: {len(dataset)} samples")
                    return dataset
                except Exception as e:
                    logger.warning(
                        "Rank %d: cached processed dataset at %s is not usable "
                        "(%s); waiting for rank 0 to rebuild it.",
                        rank,
                        cache_dir,
                        e,
                    )
                    dataset = _wait_for_valid_cache()
                    dataset = _filter_by_max_length(dataset)
                    logger.info(
                        f"Rank {rank}: loaded rebuilt dataset ({len(dataset)} samples)"
                    )
                    return dataset

        if rank == 0:
            # Rank 0: do the heavy processing and save for other ranks.
            attempt_id = resolved_attempt_id or uuid.uuid4().hex
            attempt_tag = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:32]
            failed_marker = _failed_marker_path(coordination_prefix, attempt_id)
            entries_dir = os.path.dirname(entry_dir)
            os.makedirs(entries_dir, exist_ok=True)
            temporary_cache = os.path.join(
                entries_dir,
                f".{cache_key}.tmp.{attempt_tag}.{uuid.uuid4().hex}",
            )
            _atomic_write_json(
                building_marker,
                {
                    "attempt_id": attempt_id,
                    "cache_key": cache_key,
                    "started_at_ns": time.time_ns(),
                },
            )
            _unlink_marker_for_attempt(failed_marker, attempt_id)
            try:
                dataset = _process_swe_sft(path, tokenizer, **process_kwargs)
                if len(dataset) == 0:
                    raise RuntimeError(
                        "SWE SFT preprocessing produced 0 samples; refusing to "
                        "cache an empty processed_dataset."
                    )

                shutil.rmtree(temporary_cache, ignore_errors=True)
                dataset.save_to_disk(temporary_cache)
                _atomic_write_json(
                    os.path.join(temporary_cache, ".meta.json"), cache_meta
                )
                with open(
                    os.path.join(temporary_cache, ".done"),
                    "w",
                    encoding="utf-8",
                ) as marker:
                    marker.write(str(len(dataset)))

                try:
                    os.rename(temporary_cache, entry_dir)
                except OSError as publish_error:
                    # Another builder for the same immutable key may have won
                    # publication. Reuse it only after full metadata/load
                    # validation; different keys live in different entries
                    # and are never touched here.
                    try:
                        dataset = _filter_by_max_length(_load_valid_cache())
                    except Exception:
                        quarantine = f"{entry_dir}.invalid.{uuid.uuid4().hex}"
                        try:
                            os.rename(entry_dir, quarantine)
                        except FileNotFoundError:
                            quarantine = None
                        except OSError:
                            raise publish_error
                        try:
                            try:
                                os.rename(temporary_cache, entry_dir)
                            except OSError:
                                # A concurrent repair may have published while
                                # this builder moved the corrupt entry aside.
                                dataset = _filter_by_max_length(_load_valid_cache())
                                shutil.rmtree(temporary_cache, ignore_errors=True)
                        finally:
                            if quarantine is not None:
                                shutil.rmtree(quarantine, ignore_errors=True)
                    else:
                        shutil.rmtree(temporary_cache, ignore_errors=True)
            except Exception as error:
                shutil.rmtree(temporary_cache, ignore_errors=True)
                _atomic_write_json(
                    failed_marker,
                    {
                        "attempt_id": attempt_id,
                        "cache_key": cache_key,
                        "error_type": type(error).__name__,
                        "error": str(error)[:2000],
                        "failed_at_ns": time.time_ns(),
                    },
                )
                raise
            else:
                for marker_path in (building_marker, failed_marker):
                    _unlink_marker_for_attempt(marker_path, attempt_id)

                logger.info(
                    f"Rank 0: published processed dataset "
                    f"({len(dataset)} samples) to {entry_dir}"
                )
                dataset = _filter_by_max_length(dataset)
                return dataset
        else:
            # Other ranks: wait for rank 0, then load with meta validation so a
            # cache rebuilt for different settings (or mid-rmtree) is never
            # silently loaded as this rank's dataset.
            logger.info(f"Rank {rank}: waiting for rank 0 to process dataset...")
            dataset = _wait_for_valid_cache()
            dataset = _filter_by_max_length(dataset)
            logger.info(f"Rank {rank}: loaded cached dataset ({len(dataset)} samples)")
            return dataset

    # --- Non-distributed or no cache_dir: process in current process ---
    return _process_swe_sft(path, tokenizer, **process_kwargs)
