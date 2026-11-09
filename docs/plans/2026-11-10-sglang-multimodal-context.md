# SGLang multimodal context rejection

SGLang can reject a multimodal prompt after image-token expansion, even when the
client's token budget admits it. A real request reached 131070 input tokens while the
backend required fewer than 131066. The backend returned HTTP 400 with
`Multimodal prompt is too long after expanding multimodal tokens.`; the proxy did not
recognize this format and converted it into a retryable server error.

Extend the existing backend context-error allowlist for this exact SGLang message
prefix. Read it only from the JSON body of a chained HTTP 400 response. Preserve the
existing context-overflow handling and native-receipt checks; this transport
classification alone does not establish model attribution or reward eligibility.

Reducing the configured context window would not address backend image expansion in
general. Treating every multimodal error or every HTTP 400 as context overflow would
hide image-decoding, sampling, and service failures. Both alternatives are excluded from
this change.

Regression coverage exercises HTTP and SSE responses, session overflow state, the same
message returned with HTTP 500, and an unrelated multimodal HTTP 400. Before the fix,
only the two new overflow cases failed. Real distributed rollout verification requires
deploying the fix; CPU regression tests do not establish that already-running frozen
workers have picked it up.

Validation: 728 proxy, context-recovery, and Arena tests passed in the actor image. The
14 HTTP/SSE classification cases also passed in the rollout image, using an existing
pytest-asyncio installation copied into a test-only path with its distribution metadata.
Production dependencies were not changed.
