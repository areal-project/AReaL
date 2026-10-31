# SPDX-License-Identifier: Apache-2.0
"""Decision-model SFT dataset: labelled questions as one causal row each.

See ``docs/en/customization/decision_dataset.md`` for the JSONL schema and
the meaning of ``augment``, ``p_none``, ``choice_label``, and ``soft_target``.

Each question becomes one row ``state + branch_k`` with per-token fields that
pack through AReaL's existing micro-batch path:

* ``token_roles``: ``0`` regular, ``1`` ``</opt>``, ``2`` ``<decide>``
* ``choice_label``: integer ``y`` at ``<decide>``, else ``-1``
* ``question_type``: ``0`` choice / ``1`` noul / ``2`` score at ``<decide>``
* ``soft_target``: option probabilities at ``</opt>`` when a soft ``target``
  is present, else ``-1.0``
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, model_validator

from areal.utils import logging

logger = logging.getLogger("DecisionDataset")

JSONContent = str | dict | list | int | float | bool | None
MAX_OPTIONS = 255

# Reuse existing rarely-used Qwen special tokens as delimiters
# (state, q, opt, /opt, decide) so no embedding rows need to be added.
SPECIAL = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
]

# Default token budgets: state text, one full question row (state + branch),
# and a packed multi-question limit used by ``fits``.
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
OPT_NONE, OPT_DECIDE = -1, -2

ROLE_REGULAR = 0
ROLE_OPT = 1
ROLE_DECIDE = 2

QUESTION_TYPE_IDS = {"choice": 0, "noul": 1, "score": 2}

NONE_OPTIONS = [
    ("other", "None of the above"),
    ("other", "A reason that fits none of the above"),
    ("none", "None of these"),
    ("other", "Something else"),
    ("not_listed", "Not listed here"),
    ("none_of_the_above", None),
    ("other", "A category that fits none of the above"),
    ("other", "None of the listed options apply"),
    ("unknown", "Cannot be determined from the options given"),
    ("other", "Other"),
    ("none", None),
    ("other", "An answer not covered by the other options"),
    ("no_match", "No option matches"),
]
DISTRACTORS = {
    "weather": "Bad weather caused it",
    "purple": "The colour purple",
    "pancakes": "A recipe for pancakes",
    "taxes": "Unrelated: quarterly tax filing",
}

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

# Suite partition names. Development may be development.jsonl or dev.jsonl.
_SPLIT_FILES = {
    "train": ("train.jsonl",),
    "calibration": ("calibration.jsonl",),
    "development": ("development.jsonl", "dev.jsonl"),
    "dev": ("development.jsonl", "dev.jsonl"),
    "validation": ("development.jsonl", "dev.jsonl", "validation.jsonl"),
    "val": ("development.jsonl", "dev.jsonl", "validation.jsonl"),
    "valid": ("development.jsonl", "dev.jsonl", "validation.jsonl"),
    "test": ("test.jsonl",),
}


class ContextOverflow(ValueError):
    """A record does not encode within its state or branch limit."""


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Noul | Choice | Score


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "decision"
    questions: dict[str, Question] = Field(min_length=1)
    image: str | None = None


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into text the model sees."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        (
            f"{pad}{k}:\n{render(x, indent + 1)}"
            if isinstance(x, (dict, list))
            else f"{pad}{k}: {render(x)}"
        )
        for k, x in v.items()
    )


def option_text(name: str, desc: JSONContent) -> str:
    if desc is None or desc == "":
        return name
    return f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria) -> list[str]:
    """Keys a question's probabilities / labels / targets use, in option order."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(req: SystemOneRequest):
    """Internal record for ``encode()``, plus per-question metadata."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        m = {"id": qid, "type": q.type, "keys": question_keys(q.type, q.criteria)}
        if q.type == "noul":
            c = q.criteria or {}
            opts = [
                option_text("no", c.get("false")),
                option_text("yes", c.get("true")),
            ]
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
        else:
            opts = [render(x) for x in q.criteria]
            m["legend"] = dict(zip(m["keys"], opts))
        qs.append({"instr": render(q.instructions), "options": opts, "label": 0})
        meta.append(m)
    return {"state": render(req.state), "questions": qs}, meta


def source_seed(seed, source):
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big"
    )


def augment(req, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """Choice only: permute option order; sometimes insert none / distractor.

    Soft-target questions are permuted only. Inserting or swapping options
    would change what the target means.
    """
    if min(p_none, p_none_distract, p_distract) < 0 or (
        p_none + p_none_distract + p_distract > 1
    ):
        raise ValueError(
            "augmentation probabilities must be nonnegative and sum to at most one"
        )
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q
            continue
        crit, y = dict(q["criteria"]), q["label"]
        if q.get("target") is not None:
            keys = list(crit)
            rng.shuffle(keys)
            out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}}
            continue
        r = rng.random()
        none_options = [(k, v) for k, v in NONE_OPTIONS if k not in crit]
        distractors = [k for k in DISTRACTORS if k not in crit]
        if len(crit) > 2 and r < p_none and none_options:
            nk, nd = rng.choice(none_options)
            crit.pop(y)
            crit[nk] = nd
            y = nk
        elif (
            p_none <= r < p_none + p_none_distract
            and len(crit) < MAX_OPTIONS
            and none_options
        ):
            nk, nd = rng.choice(none_options)
            crit[nk] = nd
        elif (
            p_none + p_none_distract <= r < p_none + p_none_distract + p_distract
            and len(crit) < MAX_OPTIONS
            and distractors
        ):
            k = rng.choice(distractors)
            crit[k] = DISTRACTORS[k]
        keys = list(crit)
        rng.shuffle(keys)
        out["questions"][qid] = {
            **q,
            "criteria": {k: crit[k] for k in keys},
            "label": y,
        }
    return out


def none_pair(req, rng):
    """Minimal pair for the none-of-the-above shortcut.

    Same state and question twice: once with the true option present (a none
    option is wrong) and once with it removed (the same none option is right).
    Returns ``[]`` when the request has no eligible Choice (``>= 3`` options,
    no none key already present, no soft target).
    """
    eligible = [
        (qid, q)
        for qid, q in req["questions"].items()
        if q["type"] == "choice" and len(q["criteria"]) >= 3 and q.get("target") is None
    ]
    if not eligible:
        return []
    qid, q = rng.choice(eligible)
    nk, nd = rng.choice(
        [o for o in NONE_OPTIONS if o[0] not in q["criteria"]]
        or [("none_of_these", None)]
    )
    keys = list(q["criteria"]) + [nk]
    rng.shuffle(keys)
    present = {
        **q,
        "criteria": {k: (nd if k == nk else q["criteria"][k]) for k in keys},
    }
    absent = {
        **present,
        "criteria": {k: v for k, v in present["criteria"].items() if k != q["label"]},
        "label": nk,
    }
    return [
        {"state": req["state"], "questions": {qid: present}},
        {"state": req["state"], "questions": {qid: absent}},
    ]


def load_records(path, source="custom"):
    """Labelled requests from a JSONL file, one API-shaped record per line.

    Labels: option name for choice, true/false for noul, level index for
    score. ``_meta`` and per-question ``src`` are filled so the records
    behave like a frozen suite (``source`` / line number).
    """
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if (
                "state" not in r
                or not isinstance(r.get("questions"), dict)
                or not r["questions"]
            ):
                raise ValueError(
                    f"{path}:{n + 1}: a record needs a state and a "
                    "non-empty questions object"
                )
            for qid, q in r["questions"].items():
                if "label" not in q:
                    raise ValueError(f"{path}:{n + 1}: question {qid!r} has no label")
                q.setdefault("src", f"{source}_{q['type']}")
            text = (
                json.dumps(r["state"], sort_keys=True, ensure_ascii=False)
                if not isinstance(r["state"], str)
                else r["state"]
            )
            r["_meta"] = {
                **{
                    "source": source,
                    "variant": "clean",
                    "id": f"{source}/{n}",
                    "group_id": f"{source}/{n}",
                    "row": n,
                    "split": "custom",
                    "text_sha256": hashlib.sha256(
                        " ".join(text.casefold().split()).encode()
                    ).hexdigest(),
                },
                **r.get("_meta", {}),
            }
            records.append(r)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def api_request(record):
    """Serving-shaped body: state and typed questions, never labels or meta."""
    return {
        "state": record["state"],
        "questions": {
            qid: {
                k: v for k, v in q.items() if k in ("type", "instructions", "criteria")
            }
            for qid, q in record["questions"].items()
        },
    }


def materialize(req):
    """Labelled request -> internal record via the serving path."""
    rec, meta = to_record(SystemOneRequest.model_validate(api_request(req)))
    for q, m, (qid, src_q) in zip(rec["questions"], meta, req["questions"].items()):
        y = src_q["label"]
        # noul labels are bools, score labels are level indices
        q["label"] = m["keys"].index(y) if m["type"] == "choice" else int(y)
        q["src"] = src_q["src"]
        q["qtype"] = m["type"]
        q["qid"] = qid
        q["keys"] = m["keys"]
        if src_q.get("target") is not None:
            t = [float(src_q["target"].get(k, 0.0)) for k in q["keys"]]
            if sum(t) <= 0:
                raise ValueError(f"target for {qid} puts no mass on any option")
            q["target"] = [x / sum(t) for x in t]
    return rec


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path) -> list[dict[str, Any]]:
    # Split on "\n" only: str.splitlines() also breaks on U+2028 / U+2029.
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def _partition_name(directory: Path, split: str) -> str:
    for name in _SPLIT_FILES.get(split, (f"{split}.jsonl",)):
        if (directory / name).exists():
            return name
    expected = " / ".join(_SPLIT_FILES.get(split, (f"{split}.jsonl",)))
    raise FileNotFoundError(
        f"{directory} has no partition for split {split!r} ({expected})"
    )


def load_suite(directory, split="train", verify=True):
    """Load labelled requests from a frozen suite directory.

    Expects ``manifest.json`` plus ``train.jsonl`` / ``development.jsonl``
    (or ``dev.jsonl``). When the manifest lists a sha256 / record count for
    the partition, both are checked.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{directory}: missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = _partition_name(directory, split)
    path = directory / name
    info = (manifest.get("files") or {}).get(name, {})
    if verify and info.get("sha256") and file_sha256(path) != info["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = read_jsonl(path)
    if verify and "records" in info and len(records) != info["records"]:
        raise ValueError(
            f"suite record count mismatch: {path} has {len(records)}, "
            f"manifest says {info['records']}"
        )
    source = (
        (manifest.get("name") or directory.name)
        if isinstance(manifest, dict)
        else directory.name
    )
    for n, record in enumerate(records):
        if (
            "state" not in record
            or not isinstance(record.get("questions"), dict)
            or not record["questions"]
        ):
            raise ValueError(
                f"{path}:{n + 1}: a record needs a state and a "
                "non-empty questions object"
            )
        for qid, q in record["questions"].items():
            if "label" not in q:
                raise ValueError(f"{path}:{n + 1}: question {qid!r} has no label")
            q.setdefault("src", f"{source}_{q['type']}")
        if "_meta" not in record:
            record["_meta"] = {
                "source": source,
                "variant": "clean",
                "id": f"{source}/{n}",
                "group_id": f"{source}/{n}",
                "row": n,
                "split": split,
            }
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def load_labelled_path(path, split="train", source="custom", verify_suite=True):
    """JSONL file or frozen suite directory -> labelled requests."""
    candidate = Path(path)
    if candidate.is_file():
        return load_records(candidate, source=source)
    if candidate.is_dir():
        if (candidate / "manifest.json").exists():
            return load_suite(candidate, split=split, verify=verify_suite)
        name = _partition_name(candidate, split)
        return load_records(candidate / name, source=source)
    raise FileNotFoundError(f"decision dataset path not found: {path}")


def looks_like_decision_record(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and "state" in record
        and isinstance(record.get("questions"), dict)
        and bool(record["questions"])
    )


def is_decision_path(path: str | None) -> bool:
    """True when ``path`` is a decision JSONL, suite, or decision-named source."""
    if not path:
        return False
    lowered = path.lower()
    if "decision" in lowered:
        return True
    candidate = Path(path)
    if candidate.is_file() and candidate.suffix == ".jsonl":
        return _jsonl_looks_like_decision(candidate)
    if candidate.is_dir() and (candidate / "manifest.json").exists():
        return True
    return False


def _jsonl_looks_like_decision(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                return looks_like_decision_record(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return False
    return False


def user_tokens(tok, text):
    """Tokenize caller text so it can never produce delimiter tokens.

    The fast tokenizer ignores ``split_special_tokens``, so ``<|name|>`` is
    rewritten to ``<¦name¦>`` before tokenizing.
    """
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(
    tok,
    rec,
    max_state=MAX_STATE,
    max_branch=MAX_BRANCH,
    strict=False,
    option_isolation=False,
):
    """Pack one record: ``[<state> ...]`` then per-question
    ``[<q> instr <opt> o </opt>... <decide>]``.

    Returns ``ids``, ``seg`` (0 = state, k = question k), ``pos`` (branch
    positions restart after state), ``decide_idx``, ``opt_idx`` (index of
    each ``</opt>``), and ``opt`` (per-token option index: ``OPT_NONE`` for
    state/instruction, ``0..K-1`` for option spans, ``OPT_DECIDE`` for
    ``<decide>``).
    """
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ContextOverflow(
            f"state exceeds {max_state} tokens: {len(state_tokens) + 1}"
        )
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids, seg, pos, opt = (
        list(S),
        [0] * len(S),
        list(range(len(S))),
        [OPT_NONE] * len(S),
    )
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ContextOverflow(
                f"branch too long: {len(br)} tokens with a {len(S)}-token "
                f"state (row limit {max_branch})"
            )
        base = len(ids)
        p0 = len(S)
        br_opt = (
            [OPT_NONE] * len(instr)
            + [j for j, sp in enumerate(spans) for _ in sp]
            + [OPT_DECIDE]
        )
        if option_isolation:
            longest = max(len(sp) for sp in spans)
            br_pos = (
                list(range(p0, p0 + len(instr)))
                + [p0 + len(instr) + i for sp in spans for i in range(len(sp))]
                + [p0 + len(instr) + longest]
            )
        else:
            br_pos = list(range(p0, p0 + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        ids += br
        seg += [k] * len(br)
        pos += br_pos
        opt += br_opt
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])
    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "opt": opt,
        "option_isolation": option_isolation,
        "decide_idx": decide_idx,
        "opt_idx": opt_idx,
        "labels": [q["label"] for q in rec["questions"]],
        "state_truncated": len(state_tokens) + 1 > max_state,
    }


def fits(
    rec,
    *tokenizers,
    max_state=MAX_STATE,
    max_branch=MAX_BRANCH,
    max_packed=MAX_PACKED,
):
    """True when the internal record encodes strictly within the context."""
    try:
        return all(
            len(
                encode(
                    tok, rec, max_state=max_state, max_branch=max_branch, strict=True
                )["ids"]
            )
            <= max_packed
            for tok in tokenizers
        )
    except ValueError:
        return False


def rows_of(enc):
    """Split a packed encoding into its state and per-question branch rows.

    Returns ``(state_ids, state_pos, rows)`` with
    ``rows[k] = {"ids", "pos", "decide", "opts"}``: the branch tokens of
    question ``k`` and the readout offsets *within the branch*. Feeding
    ``state + rows[k]`` as one causal row is the form Qwen3.5 GDN can run.
    """
    seg = enc["seg"]
    ls = seg.count(0)
    rows, start = [], ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1
        if seg[start] != k or seg[end - 1] != k:
            raise ValueError("branch layout mismatch")
        rows.append(
            {
                "ids": enc["ids"][start:end],
                "pos": enc["pos"][start:end],
                "decide": d - start,
                "opts": [o - start for o in oi],
            }
        )
        start = end
    return enc["ids"][:ls], enc["pos"][:ls], rows


def _question_type_id(q: dict[str, Any]) -> int:
    qtype = q.get("qtype") or q.get("type")
    if qtype not in QUESTION_TYPE_IDS:
        raise ValueError(f"unknown question type: {qtype!r}")
    return QUESTION_TYPE_IDS[qtype]


def row_features(state_ids, row, question) -> dict[str, list]:
    """Per-token lists for one causal row ``state + branch``."""
    ids = list(state_ids) + list(row["ids"])
    seqlen = len(ids)
    token_roles = [ROLE_REGULAR] * seqlen
    choice_label = [-1] * seqlen
    question_type = [-1] * seqlen
    soft_target = [-1.0] * seqlen
    offset = len(state_ids)
    target = question.get("target")
    for j, opt_off in enumerate(row["opts"]):
        pos = offset + opt_off
        token_roles[pos] = ROLE_OPT
        if target is not None:
            soft_target[pos] = float(target[j])
    decide_pos = offset + row["decide"]
    token_roles[decide_pos] = ROLE_DECIDE
    choice_label[decide_pos] = int(question["label"])
    question_type[decide_pos] = _question_type_id(question)
    return {
        "input_ids": ids,
        "token_roles": token_roles,
        "choice_label": choice_label,
        "question_type": question_type,
        "soft_target": soft_target,
    }


def question_row_tensors(features: dict[str, list]) -> dict[str, torch.Tensor]:
    """Turn a row-feature dict into 1D tensors of length ``seqlen``."""
    return {
        "input_ids": torch.tensor(features["input_ids"], dtype=torch.long),
        "token_roles": torch.tensor(features["token_roles"], dtype=torch.long),
        "choice_label": torch.tensor(features["choice_label"], dtype=torch.long),
        "question_type": torch.tensor(features["question_type"], dtype=torch.long),
        "soft_target": torch.tensor(features["soft_target"], dtype=torch.float32),
    }


def encode_question_rows(
    tok,
    rec,
    max_state=MAX_STATE,
    max_branch=MAX_BRANCH,
    strict=False,
    option_isolation=False,
    as_tensors=False,
):
    """``encode`` + ``rows_of`` -> one feature dict per question."""
    enc = encode(
        tok,
        rec,
        max_state=max_state,
        max_branch=max_branch,
        strict=strict,
        option_isolation=option_isolation,
    )
    state_ids, _, rows = rows_of(enc)
    out = [row_features(state_ids, row, q) for q, row in zip(rec["questions"], rows)]
    if as_tensors:
        return [question_row_tensors(row) for row in out]
    return out


def record_variants(
    req,
    rng,
    p_none=0.1,
    p_none_distract=0.12,
    p_distract=0.15,
    p_none_pair=0.0,
    do_augment=True,
):
    """Augmented request plus optional none-present / none-absent siblings.

    ``none_pair`` is drawn from the original request with the same RNG after
    ``augment`` has consumed draws.
    """
    if not do_augment:
        return [req]
    variants = [augment(req, rng, p_none, p_none_distract, p_distract)]
    if p_none_pair > 0 and rng.random() < p_none_pair:
        variants += none_pair(req, rng)
    return variants


def select_labelled_records(
    records: list[dict],
    n_per_source: int | None = None,
    max_records: int | None = None,
) -> list[dict]:
    """Keep the first ``n_per_source`` records of each ``_meta.source``.

    ``max_records`` is applied after that cap. File order is preserved so a
    PoC slice of a frozen suite stays deterministic.
    """
    if n_per_source is not None:
        if n_per_source < 1:
            raise ValueError("n_per_source must be >= 1")
        kept: list[dict] = []
        counts: dict[str, int] = {}
        for record in records:
            source = (record.get("_meta") or {}).get("source", "custom")
            used = counts.get(source, 0)
            if used >= n_per_source:
                continue
            counts[source] = used + 1
            kept.append(record)
        records = kept
    if max_records is not None:
        if max_records < 1:
            raise ValueError("max_records must be >= 1")
        records = records[:max_records]
    return records


def get_decision_sft_dataset(
    path: str,
    split: str | None = None,
    tokenizer=None,
    max_length: int | None = None,
    processor=None,
    seed: int = 0,
    max_state: int = MAX_STATE,
    max_branch: int = MAX_BRANCH,
    p_none: float = 0.1,
    p_none_distract: float = 0.12,
    p_distract: float = 0.15,
    p_none_pair: float = 0.0,
    augment: bool | None = None,
    option_isolation: bool = False,
    source: str = "custom",
    verify_suite: bool = True,
    n_per_source: int | None = None,
    max_records: int | None = None,
    **_unused,
):
    """Load a decision JSONL or frozen suite as one causal row per question.

    The train split shuffles choice options and may insert none / distractor
    alternatives. Other splits encode the file as-is. Override the
    probabilities or set ``augment=false`` via ``dataset_kwargs``.
    """
    from datasets import Dataset

    if tokenizer is None:
        raise ValueError("get_decision_sft_dataset requires a tokenizer")

    split = split or "train"
    do_augment = (split == "train") if augment is None else bool(augment)
    records = select_labelled_records(
        load_labelled_path(path, split=split, source=source, verify_suite=verify_suite),
        n_per_source=n_per_source,
        max_records=max_records,
    )

    rows: list[dict[str, list]] = []
    skipped = 0
    for i, req in enumerate(records):
        meta_id = req.get("_meta", {}).get("id", str(i))
        rng = random.Random(source_seed(seed, meta_id))
        variants = record_variants(
            req,
            rng,
            p_none=p_none,
            p_none_distract=p_none_distract,
            p_distract=p_distract,
            p_none_pair=p_none_pair,
            do_augment=do_augment,
        )
        for variant in variants:
            try:
                rec = materialize(variant)
                encoded = encode_question_rows(
                    tokenizer,
                    rec,
                    max_state=max_state,
                    max_branch=max_branch,
                    strict=True,
                    option_isolation=option_isolation,
                )
            except ContextOverflow as exc:
                skipped += 1
                logger.warning("Skipping decision record %s: %s", meta_id, exc)
                continue
            for row in encoded:
                if max_length is not None and len(row["input_ids"]) > max_length:
                    skipped += 1
                    continue
                rows.append(row)

    if not rows:
        raise ValueError(
            f"Decision dataset at {path} (split={split}) produced no rows"
            + (f" ({skipped} skipped)" if skipped else "")
        )
    if skipped:
        logger.info(
            "Encoded %s decision rows from %s (%s skipped)",
            len(rows),
            path,
            skipped,
        )
    return Dataset.from_list(rows)


__all__ = [
    "SPECIAL",
    "MAX_STATE",
    "MAX_BRANCH",
    "MAX_PACKED",
    "ROLE_REGULAR",
    "ROLE_OPT",
    "ROLE_DECIDE",
    "QUESTION_TYPE_IDS",
    "OPT_NONE",
    "OPT_DECIDE",
    "ContextOverflow",
    "SystemOneRequest",
    "render",
    "to_record",
    "source_seed",
    "augment",
    "none_pair",
    "load_records",
    "load_suite",
    "load_labelled_path",
    "select_labelled_records",
    "materialize",
    "is_decision_path",
    "user_tokens",
    "encode",
    "fits",
    "rows_of",
    "row_features",
    "question_row_tensors",
    "encode_question_rows",
    "record_variants",
    "get_decision_sft_dataset",
]
