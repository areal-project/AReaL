# SPDX-License-Identifier: Apache-2.0

"""SDK-only JSON-lines subprocess; run in the Pydantic<2.12 sandbox environment."""

import asyncio
import contextlib
import json
import os
import sys
import traceback
from functools import partial
from pathlib import Path


async def call_tool(env, method: str, params: dict):
    if method == "search_replace":
        # The pinned SDK misspells old_str as old_srt; keep the model schema stable.
        return await env.search_replace(
            params["filepath"], params["old_str"], params["new_str"]
        )
    return await getattr(env, method)(**params)


async def serve() -> None:
    from aweagent.envs import swe as swe_sdk

    env = None

    async def journal_ownership() -> None:
        previous = None
        while True:
            instance_id = env.env_info.get("instance_id") if env else None
            if instance_id and instance_id != previous and len(sys.argv) > 1:
                path = Path(sys.argv[1])
                temporary = path.with_suffix(".tmp")
                with temporary.open("w") as stream:
                    json.dump({"instance_id": instance_id}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                previous = instance_id
            await asyncio.sleep(0.1)

    journal = asyncio.create_task(journal_ownership())
    allowed = {"execute_bash", "search_replace", "submit_final_solution", "get_reward"}
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            request = json.loads(line)
            response = {"id": request["id"]}
            try:
                method, params = request["method"], request.get("params", {})
                if method == "initialize":
                    if env is not None:
                        raise RuntimeError("Sandbox already initialized")
                    data = dict(params["data"])
                    data.pop("patch", None)
                    # The SDK's per-tool timeout does not update FastMCP's HTTP
                    # read timeout. Set the public Environment constructor value
                    # before its first connection, in this isolated SDK process.
                    swe_sdk.Environment = partial(
                        swe_sdk.Environment, timeout=params["transport_timeout"]
                    )
                    env = swe_sdk.AenvSWE(
                        data_id=data["instance_id"],
                        data=data,
                        faas_image=data.get("sandbox_image") or data["instance_id"],
                        aenv_version=params["environment"],
                    )
                    if not await env.check_env():
                        raise RuntimeError("Sandbox initialization failed")
                    result = {"repo_dir": env.config.repo_dir}
                elif method == "release":
                    if env is not None:
                        await env.release()
                        env = None
                    response["result"] = {"released": True}
                    sys.__stdout__.write(json.dumps(response) + "\n")
                    sys.__stdout__.flush()
                    break
                elif method in allowed and env is not None:
                    result = await call_tool(env, method, params)
                else:
                    raise ValueError("Unsupported sandbox operation")
                response["result"] = result
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response["error"] = f"{type(error).__name__}: {error}"
            if env is not None:
                response["instance_id"] = env.env_info.get("instance_id")
                response["tool_call_failed_count"] = env.tool_call_failed_count
            sys.__stdout__.write(json.dumps(response) + "\n")
            sys.__stdout__.flush()
    finally:
        journal.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await journal
        if env is not None:
            await env.release()


if __name__ == "__main__":
    # Upstream diagnostics must never corrupt the protocol on original stdout.
    with contextlib.redirect_stdout(sys.stderr):
        asyncio.run(serve())
