# Memory Service reference examples

This directory contains small executable checks for one concrete Memory
evolution mechanism. They are reference examples, not production services or
benchmarks.

## Structured fact policy

`adaptive_codebook_eval.py` uses `StructuredFactPolicy` to turn already
structured, deployment-authorized evidence into immutable fact revisions and a
new release. The policy does **not** extract facts from natural-language
conversations and does not update model weights, system prompts, or skills.

An update uses an exact V1 JSON schema. For example, an `ADD` command is:

```json
{"expected_parent_revision_id":null,"fact_key":"project-aurora","fact_value":"CODE-204","operation":"add","record_kind":"structured_fact_update","schema_version":1}
```

The supported operations are:

- `ADD`: the fact must be absent and `expected_parent_revision_id` must be
  `null`.
- `CONFIRM`: the fact must exist, its value must be unchanged, and
  `expected_parent_revision_id` must equal the active revision ID. It creates
  no new revision.
- `SUPERSEDE`: the fact must exist, the new value must differ, and
  `expected_parent_revision_id` must equal the active revision ID.

This parent check is relative to the supplied base release. It is not a global,
linearizable compare-and-swap over all concurrent writers.

### Decision order and trust boundary

For every evidence record, the policy applies these gates in order:

1. require the requested `MemoryScope`;
2. reject observations after `captured_through`;
3. resolve the exact persisted `EvidenceRecord` and verify its SHA-256
   commitment;
4. parse the canonical structured command;
5. ask the deployment-owned `EvidenceTrustPolicy` for authority;
6. check the active parent revision;
7. append immutable candidate and revision records;
8. publish a new immutable release containing the accepted tips.

Timestamps only decide whether evidence was observable at the cutoff. They do
not select a winner. Updates execute in the caller's explicit tuple order, and
conflicts are resolved by the parent revision check.

`EvidenceKind` is not identity or authority. The example trust policy is only a
fixture: its local “verified tool” marker is forgeable and must not be copied
into production. A real deployment still needs authenticated source identity,
authorization, anti-replay, and provenance verification.

### Run the causal smoke test

From the repository root:

```bash
python examples/memory_service/adaptive_codebook_eval.py
```

The deterministic test evaluates eight paired subjects across seven arms. The
future query omits the answer, so useful information must travel through the
Memory release, assignment, retrieval, rendering, consumer acknowledgement,
and exposure path. The controls include no memory, a frozen release, target
masking, stale memory, an explicit raw-history policy, and an oracle ceiling.

The expected aggregate rewards for the fixture are:

| arm | reward |
| --- | ---: |
| `no_memory` | 0 |
| `static` | 2 |
| `adaptive` | 8 |
| `target_masked` | -8 |
| `stale` | 2 |
| `raw_history` | 4 |
| `oracle` | 8 |

These numbers show that this update mechanism works on the controlled synthetic
task. They are not evidence of general agent self-evolution or statistical
significance.

### Current limitations

- The caller must provide canonical structured commands in deterministic order.
- V1 has string keys and values only; it has no delete/retract, TTL, confidence,
  multi-value merge, or evidence aggregation.
- The base release must contain only unique structured-fact records.
- Candidate, revision, and release appends are not one physical transaction. A
  later failure can leave append-only orphan artifacts for audit and future
  reconciliation.
- Concurrent calls from one base release can create separate valid branches;
  there is no global active-head arbitration in this policy.
- The repository implementations are in-process reference stores, not durable
  production backends.
- Decisions and quarantines are returned to the caller but are not yet persisted
  in a review ledger.
