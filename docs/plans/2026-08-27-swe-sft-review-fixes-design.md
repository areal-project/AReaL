# SWE SFT review fixes

## Problem

The SWE SFT loader has four correctness gaps found during PR review:

- runtime chat-template patching is not idempotent;
- delimiter regexes can truncate assistant supervision when code contains a literal
  end-of-turn token;
- tokenized datasets retain rows with no supervised tokens;
- remote data workers do not pass their topology to the shared-cache coordinator.

The split CLI also imports `_tokenize_samples` from the wrong module.

## Design

Chat-template masking uses structural Jinja generation tracking exclusively. The known
Qwen3 and Bailing V2.5 patches move the assistant header outside a `generation` block
and keep the assistant body, tool calls, and real end-of-turn token inside it. A stable
Jinja comment marks a completed AReaL patch, making repeated calls a no-op. Templates
without structural tracking fail before dataset mapping instead of falling back to
delimiter parsing.

Tokenized rows must contain equal-length `input_ids` and `loss_mask` arrays and at least
one nonzero loss-mask entry. The same predicate applies to freshly tokenized and
pre-tokenized datasets. An empty result raises at preprocessing time. The distributed
cache metadata version is bumped so caches built before this invariant are rebuilt.

Remote data workers pass `DataWorkerConfig.rank` and `world_size` through private
dataset-loader parameters. SWE cache coordination prefers these explicit values and
falls back to `RANK` and `WORLD_SIZE` only for direct SPMD execution. Explicit topology
is validated before preprocessing, and request-provided dataset kwargs cannot override
worker-owned values.

## Verification

Regression tests cover repeated template patching, rendered-text equivalence, literal
delimiter payloads, all-zero masks, pre-tokenized filtering, cache topology precedence,
invalid topology, worker parameter forwarding, and CLI startup.
