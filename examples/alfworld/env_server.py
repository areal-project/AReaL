"""ALFWorld Environment Server — deadlock-proof lifecycle management.

Each env runs in its own subprocess. If textworld hangs, we kill the
subprocess rather than blocking the server forever.

No more MAX_EPISODES restart — registry leaks and pipe deadlocks are
handled at the source.

Usage:
    pip install fastapi uvicorn textworld alfworld
    python env_server.py --port 8765
"""
import os
import uuid
import asyncio
import logging
import time
import multiprocessing as mp
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("env_server")

MAX_ENVS = 128
OP_TIMEOUT = 30      # seconds for create/reset/step
CLOSE_TIMEOUT = 10   # seconds for close before force-kill
STALE_TIMEOUT = 300  # seconds idle before auto-reap


# ---------------------------------------------------------------------------
# Subprocess worker: each env lives in its own process, communicating via Pipe.
# If textworld deadlocks (Frotz pipe hang, GC __del__ deadlock, etc.),
# we just kill the process — no shared state to corrupt.
# ---------------------------------------------------------------------------

def _env_worker(pipe: mp.connection.Connection):
    """Runs in a child process. Owns one textworld env at a time.

    Uses textworld.start() (the public high-level API) which correctly handles
    .tw-pddl files, TWInform7 wrapper, etc. Wrapped with Limit + Filter for
    step counting and info extraction.

    This avoids TextworldBatchGymEnv which has a close-on-reset bug.
    """
    import textworld
    from textworld.envs.wrappers import Filter, Limit

    env = None

    try:
        while True:
            try:
                cmd = pipe.recv()
            except EOFError:
                break

            op = cmd[0]
            try:
                if op == "create":
                    game_file = cmd[1]
                    request_infos = textworld.EnvInfos(won=True, admissible_commands=False)
                    env = textworld.start(game_file, request_infos)
                    env = Limit(env, max_episode_steps=100)
                    env = Filter(env)
                    pipe.send(("ok", None))

                elif op == "reset":
                    obs, infos = env.reset()
                    safe_info = {}
                    for k, v in infos.items():
                        if isinstance(v, (str, int, float, bool)):
                            safe_info[k] = v
                    pipe.send(("ok", (str(obs), safe_info)))

                elif op == "step":
                    action = cmd[1]
                    obs, reward, done, infos = env.step(action)
                    won = bool(infos.get("won", False))
                    pipe.send(("ok", {
                        "obs": str(obs), "reward": float(reward),
                        "done": bool(done), "won": won,
                    }))

                elif op == "close":
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                        env = None
                    pipe.send(("ok", None))
                    break

                else:
                    pipe.send(("error", f"unknown op: {op}"))

            except Exception as e:
                import traceback
                pipe.send(("error", f"{e}\n{traceback.format_exc()}"))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        pipe.close()


class EnvProcess:
    """Manages a single textworld env in a child process."""

    def __init__(self):
        self._parent_pipe, child_pipe = mp.Pipe()
        self._process = mp.Process(target=_env_worker, args=(child_pipe,), daemon=True)
        self._process.start()
        child_pipe.close()

    def _call(self, *args, timeout: float = OP_TIMEOUT):
        """Send command and wait for result with timeout. Raises on timeout/error."""
        self._parent_pipe.send(args)
        if not self._parent_pipe.poll(timeout):
            raise TimeoutError(f"env subprocess did not respond within {timeout}s")
        status, result = self._parent_pipe.recv()
        if status == "error":
            raise RuntimeError(result)
        return result

    def create(self, game_file: str, timeout: float = OP_TIMEOUT):
        self._call("create", game_file, timeout=timeout)

    def reset(self, timeout: float = OP_TIMEOUT):
        return self._call("reset", timeout=timeout)

    def step(self, action: str, timeout: float = OP_TIMEOUT):
        return self._call("step", action, timeout=timeout)

    def close(self, timeout: float = CLOSE_TIMEOUT):
        try:
            self._call("close", timeout=timeout)
        except (TimeoutError, OSError, EOFError):
            pass

    def kill(self):
        """Force-kill the subprocess. Use when close() times out."""
        try:
            self._parent_pipe.close()
        except Exception:
            pass
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5)

    def safe_shutdown(self, timeout: float = CLOSE_TIMEOUT):
        """Try graceful close, then force-kill if needed."""
        try:
            self.close(timeout=timeout)
        except Exception:
            pass
        self.kill()


@dataclass
class EnvEntry:
    proc: EnvProcess
    last_access: float = field(default_factory=time.time)
    op_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


envs: Dict[str, EnvEntry] = {}
envs_lock = asyncio.Lock()
episode_count = 0


class CreateReq(BaseModel):
    game_file: str

class EnvIdReq(BaseModel):
    env_id: str

class StepReq(BaseModel):
    env_id: str
    action: str


async def _run_in_thread(fn, *args):
    """Run a blocking function in a thread so asyncio loop isn't blocked."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


async def _reap_stale_envs():
    """Background task: reap envs idle for more than STALE_TIMEOUT seconds."""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        stale = []
        async with envs_lock:
            for eid, entry in list(envs.items()):
                if now - entry.last_access > STALE_TIMEOUT:
                    stale.append((eid, envs.pop(eid)))

        for eid, entry in stale:
            logger.info("Reaping stale env %s (idle %.0fs)", eid, now - entry.last_access)
            await _run_in_thread(entry.proc.safe_shutdown)


@asynccontextmanager
async def lifespan(app: FastAPI):
    reaper = asyncio.create_task(_reap_stale_envs())
    yield
    reaper.cancel()
    async with envs_lock:
        entries = list(envs.values())
        envs.clear()
    for entry in entries:
        entry.proc.safe_shutdown(timeout=3)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "num_envs": len(envs),
        "episode_count": episode_count,
    }


@app.post("/create")
async def create(req: CreateReq):
    async with envs_lock:
        if len(envs) >= MAX_ENVS:
            raise HTTPException(503, "Max envs reached")

    proc = EnvProcess()
    try:
        await _run_in_thread(proc.create, req.game_file)
    except TimeoutError:
        proc.kill()
        raise HTTPException(500, "Create timed out — textworld subprocess stuck")
    except Exception as e:
        proc.kill()
        raise HTTPException(500, f"Create failed: {e}")

    eid = str(uuid.uuid4())
    async with envs_lock:
        envs[eid] = EnvEntry(proc=proc)
    return {"env_id": eid}


@app.post("/reset")
async def reset(req: EnvIdReq):
    async with envs_lock:
        entry = envs.get(req.env_id)
        if not entry:
            raise HTTPException(404, "Env not found")
        entry.last_access = time.time()
    async with entry.op_lock:
        try:
            obs, info = await _run_in_thread(entry.proc.reset)
        except TimeoutError:
            logger.error("Reset timed out for %s, killing subprocess", req.env_id)
            async with envs_lock:
                envs.pop(req.env_id, None)
            await _run_in_thread(entry.proc.kill)
            raise HTTPException(500, "Reset timed out — env killed and removed")
        except Exception as e:
            logger.error("Reset failed for %s: %s", req.env_id, e)
            async with envs_lock:
                envs.pop(req.env_id, None)
            await _run_in_thread(entry.proc.kill)
            raise HTTPException(500, f"Reset failed: {e}")
    return {"obs": obs, "info": info}


@app.post("/step")
async def step(req: StepReq):
    async with envs_lock:
        entry = envs.get(req.env_id)
        if not entry:
            raise HTTPException(404, "Env not found")
        entry.last_access = time.time()
    async with entry.op_lock:
        try:
            result = await _run_in_thread(entry.proc.step, req.action)
        except TimeoutError:
            logger.error("Step timed out for %s, killing subprocess", req.env_id)
            async with envs_lock:
                envs.pop(req.env_id, None)
            await _run_in_thread(entry.proc.kill)
            return {"obs": "", "reward": 0.0, "done": True, "won": False, "error": "step_timeout"}
        except Exception as e:
            logger.error("Step failed for %s: %s", req.env_id, e)
            async with envs_lock:
                envs.pop(req.env_id, None)
            await _run_in_thread(entry.proc.kill)
            return {"obs": "", "reward": 0.0, "done": True, "won": False, "error": str(e)}
    return result


@app.post("/close")
async def close(req: EnvIdReq):
    global episode_count
    async with envs_lock:
        entry = envs.pop(req.env_id, None)
    if not entry:
        return {"success": False}
    await _run_in_thread(entry.proc.safe_shutdown)
    episode_count += 1
    return {"success": True}


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
