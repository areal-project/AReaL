# Bailing V3 HF Export Hardening

## Context

The Bailing V3 bridge must dispatch by `architectures` because V3 and Bailing V2.5 share
`model_type="bailing_hybrid"`. The production `swe-dev` implementation therefore loads
V3 through a generic `PretrainedConfig` and constructs `BailingV3Bridge` explicitly.
That behavior is retained: changing registration or relying only on `AutoConfig` would
risk routing V2.5 checkpoints through the V3 bridge.

The generic config keeps the source `model_type` on the instance, but Transformers
serializes the class-level value. For config classes without their own class-level
`model_type`, `save_pretrained()` can therefore write an empty or missing value.
Internal `swe-dev` later mitigated normal periodic saves by forwarding the local source
checkpoint directory, but direct saves, disk weight updates, and mbridge's native saver
still bypass that mitigation.

## Design

Keep model loading and weight conversion unchanged. After every mbridge HF config save,
run one shared finalization step that restores the original instance `model_type` in the
written `config.json`. When a local `base_model_path` is available, the same step also
copies tokenizer, generation, chat-template, and remote-code assets and preserves source
config fields. Port the production `Saver` fallback that derives `base_model_path` from
`engine.config.path`; this remains useful for source assets but is no longer required
for `model_type` correctness.

The SWE entrypoint also passes preprocessing controls as call-time keyword arguments.
The single-controller `RDataset` branch will merge those arguments with
`dataset_config.dataset_kwargs`, giving explicit call-time arguments precedence, so its
behavior matches the direct SPMD loader.

## Verification

Add CPU tests for a remote-style config class without a class-level `model_type`, verify
the exported JSON, and reload it through `AutoConfig`. Cover the production Saver
fallback and both mbridge export finalization paths without requiring GPUs. Add a
single-controller test proving SWE preprocessing options reach `RDataset`. Existing
Bailing V3 KDA, MLA, MoE, and weight-layout code remains untouched.

## Review follow-up: dataset scope

The public SWE SFT loader keeps only the preprocessing behavior exercised by the
production recipes: canonical pair mode, full-trajectory mode, error/tool filtering,
thinking preservation or full stripping, tool rendering, and tokenization/cache support.
Optional random thinking augmentation, no-thinking ratio sampling, and task-notification
truncation are removed because all tracked production configurations leave them
disabled. Removing their CLI, config, cache-key, helper, and test surfaces together
avoids shipping unexercised branches while preserving the default dataset exactly.

The delimited SWE path matcher remains intentional. A broad substring check previously
misrouted unrelated paths containing `swe`, including `answer_sft`, `sweep_results`, and
user directories such as `swetha`; internal commit `c84db0bbf` fixed that production
bug.
