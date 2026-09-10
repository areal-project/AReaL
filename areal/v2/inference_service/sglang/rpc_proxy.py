# SPDX-License-Identifier: Apache-2.0
"""Lightweight ZMQ proxy for dispatching RPC to scheduler subprocesses."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import zmq
import zmq.asyncio
from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput
from sglang.srt.server_args import PortArgs


class RpcProxy:
    """ZMQ proxy bridging the HTTP process to scheduler subprocesses.

    Two independent channels:

    * **RPC channel** (DEALER on ``rpc_ipc_name``): sends :class:`RpcReqInput`,
      receives :class:`RpcReqOutput` — same protocol as ``Engine.collective_rpc``.
    * **Result channel** (PULL on ``result_ipc``): receives pyobj results
      pushed by :class:`AwexSchedulerBridge` rank 0 via its PUSH socket.
    """

    def __init__(
        self, port_args: PortArgs, result_ipc: str, default_timeout_s: float = 300.0
    ) -> None:
        from sglang.srt.utils.network import get_zmq_socket

        self._rpc_ctx = zmq.asyncio.Context(1)
        self._rpc_socket = get_zmq_socket(
            self._rpc_ctx, zmq.DEALER, port_args.rpc_ipc_name, True
        )

        self._result_ctx = zmq.asyncio.Context(1)
        self._result_pull = self._result_ctx.socket(zmq.PULL)
        self._result_pull.bind(result_ipc)
        self._lock = asyncio.Lock()
        self._default_timeout_s = default_timeout_s

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("SGLang collective RPC deadline exceeded")
        return remaining

    async def _recv_ack(self, rid: str, deadline: float) -> RpcReqOutput:
        while True:
            resp = await asyncio.wait_for(
                self._rpc_socket.recv_pyobj(), timeout=self._remaining(deadline)
            )
            if isinstance(resp, RpcReqOutput) and resp.rid == rid:
                return resp

    async def _recv_result(self, rid: str, deadline: float) -> Any:
        while True:
            envelope = await asyncio.wait_for(
                self._result_pull.recv_pyobj(), timeout=self._remaining(deadline)
            )
            if isinstance(envelope, dict) and envelope.get("rid") == rid:
                return envelope.get("value")

    async def _collective_rpc(
        self,
        method: str,
        *,
        timeout_s: float | None,
        expect_result: bool,
        kwargs: dict[str, Any],
    ) -> Any:
        deadline = time.monotonic() + (
            self._default_timeout_s if timeout_s is None else timeout_s
        )
        await asyncio.wait_for(self._lock.acquire(), timeout=self._remaining(deadline))
        try:
            rid = uuid4().hex
            req = RpcReqInput(
                method=method,
                parameters=kwargs if kwargs else None,
                rid=rid,
            )
            await asyncio.wait_for(
                self._rpc_socket.send_pyobj(req), timeout=self._remaining(deadline)
            )
            resp = await self._recv_ack(rid, deadline)
            if not resp.success:
                raise RuntimeError(f"RPC {method} failed: {resp.message}")
            if expect_result:
                return await self._recv_result(rid, deadline)
            return None
        finally:
            self._lock.release()

    async def collective_rpc(
        self, method: str, *, timeout_s: float | None = None, **kwargs: Any
    ) -> None:
        await self._collective_rpc(
            method,
            timeout_s=timeout_s,
            expect_result=False,
            kwargs=kwargs,
        )

    async def collective_rpc_with_result(
        self, method: str, *, timeout_s: float | None = None, **kwargs: Any
    ) -> Any:
        return await self._collective_rpc(
            method,
            timeout_s=timeout_s,
            expect_result=True,
            kwargs=kwargs,
        )

    def close(self) -> None:
        for sock in (self._rpc_socket, self._result_pull):
            if sock is not None:
                sock.close(linger=0)
        for ctx in (self._rpc_ctx, self._result_ctx):
            if ctx is not None:
                ctx.term()
