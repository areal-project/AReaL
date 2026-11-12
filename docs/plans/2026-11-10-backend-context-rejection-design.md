# Confirmed context rejection with backend routing

## Failure and requirements

An EAGLE rollout requested 262148 tokens against a 262144-token context. SGLang counted
four reserved draft slots in addition to AReaL's input and output budget. Its tokenizer
rejected the request with HTTP 400 before sending it to the scheduler. The route client
treated this as an uncertain execution, failed the entire pool, and terminated training
before the next optimizer update.

Preserve the execution fence for ambiguous failures. Release a lease only when the
backend adapter establishes that generation was never scheduled. Preserve the backend
error body so the proxy can apply its existing context-overflow disposition, without
relabeling a rejected request as a successful generation.

## Design

SGLang recognizes only its specific tokenizer context-validation messages in a complete
JSON HTTP 400 response. It does not treat arbitrary 400 responses, status 500, malformed
bodies, or a context error code alone as proof. Other backends have no such
classification unless they explicitly implement it.

The remote engine preserves the bounded HTTP error body and wraps confirmed rejections
in `GenerationRejectedError`. The router releases that execution with the existing
`failed` outcome and propagates the error without retrying generation. A failed release
acknowledgement still fails the pool and retains the lease. The proxy's existing
chained-error classifier handles the context rejection.

For the affected EAGLE recipe, reserve the backend's four draft slots by setting the
generation and agent budgets to 262140 while keeping the model context at 262144. This
follows the deployed SGLang calculation:
`max(speculative_eagle_topk * speculative_num_steps, max_speculative_num_draft_tokens)`.
It does not truncate prompt tokens or change sampling, reward filtering, or LR.

Increasing timeouts would not address this failure. Accepting all HTTP 400s as terminal
would discard the execution fence for errors that can occur after scheduling. Enabling
input truncation would change task context unnecessarily.

## Validation and limits

HTTP integration tests cover both context message forms, malformed JSON, unknown 400s,
code-only errors, 500s, propagation to the proxy classifier, continued use of the pool,
and release acknowledgement failure. Existing cancellation and uncertain execution tests
remain required.

Validate the exact length boundary against the deployed SGLang tokenizer code. CPU tests
do not establish successful GPU recovery, a new optimizer update, checkpoint
publication, convergence, or end-to-end MTP speedup; those require the subsequent
distributed run.
