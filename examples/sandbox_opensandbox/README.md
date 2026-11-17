# OpenSandbox sandbox example

This example shows how to use `OpenSandboxRunner` from plain Python code, such as a
reward function, evaluation script, or data-prep utility.

## Install

```bash
uv sync --extra sandbox
export OPEN_SANDBOX_DOMAIN=your-endpoint
export OPEN_SANDBOX_API_KEY=your-key   # omit when the endpoint needs no key
```

## Run

```bash
uv run python examples/sandbox_opensandbox/reward_example.py
```

## Notes

- `OpenSandboxRunner` is the synchronous entry point for non-async user code. It uses
  the SDK's blocking API, so no background event loop is involved.
- Each runner instance creates one sandbox and keeps it warm across multiple `run()`
  calls.
- Every sandbox is created with a server-side expiration (`ttl_seconds`, one hour by
  default). If the process dies before cleanup runs, the platform deletes the sandbox
  at expiry instead of leaking it; while the runner is alive, the lease is renewed
  after each `run()`.
- Pass `network_block_all=True` to deny all outbound network access from the sandbox,
  and `image=` / `snapshot_id=` to pin the environment the code runs in.
