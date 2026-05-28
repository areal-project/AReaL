---
name: Refactor
about: Propose an API change
title: '[Refactor]'
labels: ''
assignees: ''
---

## Checklist

- [ ] This refactor maintains backward compatibility with all user-facing APIs.
- [ ] For large-scale refactors, I’ve prepared a phased implementation plan.

## Current Limitations

Describe existing design flaws or technical debt:

**Example:** AReaL currently only supports FSDP for dense models, which limits
extensibility for context parallelism or expert parallelism."

## Proposed Refactor

Explain the improvement and its benefits:

**Example:** "Adopt Megatron-Core’s parallel state management instead of thecustom
topology implementation. This would reduce maintenance overhead and improve support for
emerging parallelism techniques"

## Alternatives Considered

Briefly summarize and compare other potential approaches (if applicable).

## Additional Context

Add references, screenshots, or related discussions.
