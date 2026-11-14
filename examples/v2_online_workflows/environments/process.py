# SPDX-License-Identifier: Apache-2.0
"""Bounded subprocess execution shared with Python 3.6.8 compute hosts."""

import os
import selectors
import signal
import subprocess
import time


def run_limited(
    command,
    *,
    timeout,
    output_bytes,
    truncate=False,
    process_group=False,
    env=None,
    separate_stderr=False,
    preserve_tail=False,
    keep_stdout_bytes=False,
):
    """Bound output/time; optionally retain exact stdout bytes for artifact export."""
    if preserve_tail and not truncate:
        raise ValueError("Preserving the output tail requires truncate=True")

    def head_tail(data, limit):
        if len(data) <= limit:
            return data
        head = limit // 2
        tail = limit - head
        return data[:head] + (data[-tail:] if tail else data[:0])

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if separate_stderr else subprocess.STDOUT,
        start_new_session=process_group,
        env=env,
    )
    output = bytearray()
    errors = bytearray()
    status = "completed"
    truncated = False
    bytes_seen = 0
    assert process.stdout is not None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, output)
            if separate_stderr:
                selector.register(process.stderr, selectors.EVENT_READ, errors)
            while selector.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    status = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.2)):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if preserve_tail:
                        bytes_seen += len(chunk)
                        key.data.extend(chunk)
                        key.data[:] = head_tail(key.data, output_bytes)
                        truncated = bytes_seen > output_bytes
                        continue
                    available = output_bytes - len(output) - len(errors)
                    key.data.extend(chunk[:available])
                    if len(chunk) > available:
                        truncated = True
                        if not truncate:
                            status = "output_limit"
                            break
                if status != "completed":
                    break
        if status == "completed":
            try:
                process.wait(timeout=max(0.01, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                status = "timeout"
    finally:
        if process_group:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    if preserve_tail:
        # Reserve stderr first so noisy stdout cannot hide the final exception.
        # Each stream's collector is bounded; the returned streams share one cap.
        output = head_tail(output, output_bytes - len(errors))
    result = {
        "status": "output_limit" if truncated and status == "completed" else status,
        "exit_code": process.returncode,
        "output": output.decode(errors="replace"),
        "seconds": time.monotonic() - started,
    }
    if separate_stderr:
        result.update(stdout=result["output"], stderr=errors.decode(errors="replace"))
    if keep_stdout_bytes:
        result["stdout_bytes"] = bytes(output)
    if preserve_tail:
        result["output_truncated"] = truncated
    return result
