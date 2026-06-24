from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging, stats_tracker

logger = logging.getLogger("OSWorldWorkflow")

OSWORLD_SYSTEM_PROMPT = (
    "You are a computer-use agent operating an Ubuntu desktop through pyautogui. "
    "At each turn you will receive the user instruction and the current screenshot "
    "of the desktop. Produce a short plan and then a single ```python``` code block "
    "containing pyautogui code to execute. Use absolute pixel coordinates relative "
    "to the screenshot. When the task is fully completed, reply with the single "
    "token DONE on its own line (no code block). If the task is infeasible reply "
    "with FAIL."
)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)


def _ensure_osworld_on_path(osworld_root: str | None) -> None:
    if osworld_root and osworld_root not in sys.path:
        sys.path.insert(0, osworld_root)


def _screenshot_to_data_uri(screenshot: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(screenshot).decode("utf-8")


def _parse_actions(text: str) -> list[str]:
    """Extract executable actions from the model reply.

    Returns a list with one pyautogui snippet, or a single control token
    ("DONE" / "FAIL" / "WAIT") to be forwarded to DesktopEnv.step.
    """
    text = text.strip()
    code = _CODE_BLOCK_RE.findall(text)
    if code:
        snippet = code[-1].strip()
        if snippet:
            return [snippet]
    for line in reversed(text.splitlines()):
        token = line.strip().upper()
        if token in {"DONE", "FAIL", "WAIT"}:
            return [token]
    return []


class OSWorldWorkflow(RolloutWorkflow):
    """Multi-turn VLM rollout workflow backed by OSWorld's DesktopEnv.

    Each episode spins up a fresh DesktopEnv for the given task, drives the
    agent<->env loop through ``ArealOpenAI`` so completions/tokens are cached
    for training, and returns interaction samples tagged with the final
    env.evaluate() reward.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        evaluation_examples_dir: str,
        osworld_root: str | None = None,
        provider_name: str = "docker",
        path_to_vm: str | None = None,
        os_type: str = "Ubuntu",
        headless: bool = True,
        screen_size: tuple[int, int] = (1920, 1080),
        observation_type: str = "screenshot",
        action_space: str = "pyautogui",
        cache_dir: str = "cache",
        max_steps: int = 15,
        n_trajs: int = 1,
        sleep_after_execution: float = 1.0,
        env_reset_wait_secs: float = 60.0,
        max_workers: int = 4,
        turn_discount: float = 0.9,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        remote_server_url: str = "",
        remote_request_timeout_secs: float = 1800.0,
        gateway_endpoint: str = "",
        gateway_token: str = "",
        gateway_timeout_secs: int = 1800,
        text_only: bool = False,
        processor_path: str | None = None,
    ):
        # Three transport modes; pick one. Precedence: gateway sandbox >
        # custom remote server > in-process DesktopEnv. Only the in-process
        # path needs OSWorld already importable in this container, so we can
        # defer that for the remote modes — except the gateway path
        # re-imports OSWorld lazily for PythonController/DesktopEnv
        # subclassing, so we still need the path.
        self.remote_server_url = remote_server_url.strip()
        self.remote_request_timeout_secs = remote_request_timeout_secs
        self.gateway_endpoint = gateway_endpoint.strip()
        self.gateway_token = gateway_token.strip()
        self.gateway_timeout_secs = gateway_timeout_secs
        self.osworld_root = osworld_root
        if self.gateway_endpoint and self.gateway_token:
            _ensure_osworld_on_path(osworld_root)
        elif not self.remote_server_url:
            _ensure_osworld_on_path(osworld_root)

        self.gconfig = gconfig.new(n_samples=1)
        self.tokenizer = tokenizer
        self.evaluation_examples_dir = evaluation_examples_dir
        self.provider_name = provider_name
        self.path_to_vm = path_to_vm
        self.os_type = os_type
        self.headless = headless
        self.screen_size = tuple(screen_size)
        self.observation_type = observation_type
        self.action_space = action_space
        self.cache_dir = cache_dir
        self.max_steps = max_steps
        self.n_trajs = n_trajs
        self.sleep_after_execution = sleep_after_execution
        self.env_reset_wait_secs = env_reset_wait_secs
        self.turn_discount = turn_discount
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        if self.dump_dir and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        self.text_only = text_only
        # Lazy-loaded multimodal processor for VL training. We only need it
        # when text_only=False; loading is deferred to first use so the
        # text-only smoke path doesn't pay the import / disk-read cost.
        self._processor_path = processor_path
        self._processor = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _load_task_config(self, data: dict[str, Any]) -> dict[str, Any]:
        if "config" in data and "evaluator" in data and "instruction" in data:
            return dict(data)
        domain = data["domain"]
        example_id = data.get("example_id") or data.get("id")
        path = (
            Path(self.evaluation_examples_dir)
            / "examples"
            / domain
            / f"{example_id}.json"
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _build_env(self):
        if self.gateway_endpoint and self.gateway_token:
            from .gateway_sandbox import make_sandbox_desktop_env

            return make_sandbox_desktop_env(
                osworld_root=self.osworld_root or "",
                cluster_endpoint=self.gateway_endpoint,
                secret_token=self.gateway_token,
                cache_dir=self.cache_dir,
                screen_size=self.screen_size,
                require_a11y_tree=self.observation_type != "screenshot",
                os_type=self.os_type,
            )

        if self.remote_server_url:
            from .remote_desktop_env import RemoteDesktopEnv

            return RemoteDesktopEnv(
                server_url=self.remote_server_url,
                provider_name=self.provider_name,
                path_to_vm=self.path_to_vm,
                action_space=self.action_space,
                cache_dir=self.cache_dir,
                screen_size=self.screen_size,
                headless=self.headless,
                os_type=self.os_type,
                require_a11y_tree=self.observation_type != "screenshot",
                request_timeout_secs=self.remote_request_timeout_secs,
            )

        from desktop_env.desktop_env import DesktopEnv

        return DesktopEnv(
            provider_name=self.provider_name,
            path_to_vm=self.path_to_vm,
            action_space=self.action_space,
            cache_dir=self.cache_dir,
            screen_size=self.screen_size,
            headless=self.headless,
            os_type=self.os_type,
            require_a11y_tree=self.observation_type != "screenshot",
        )

    def _build_user_turn(self, text: str, screenshot: bytes) -> dict[str, Any]:
        if self.text_only:
            # Smoke ablation: skip the screenshot. The agent operates blind;
            # we just want a real PPO step against a text-only base model.
            stub = (
                f"\n[screenshot omitted in text_only mode; "
                f"{len(screenshot or b'')} bytes available]"
            )
            return {"role": "user", "content": text + stub}
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": _screenshot_to_data_uri(screenshot)},
                },
            ],
        }

    async def _run_single_trajectory(
        self,
        engine,
        task_config: dict[str, Any],
        traj_idx: int,
    ) -> tuple[dict[str, Any] | None, float | None]:
        instruction = task_config["instruction"]
        task_id = task_config.get("id", "unknown")
        loop = asyncio.get_running_loop()

        env = None
        try:
            env = await loop.run_in_executor(self.executor, self._build_env)
            await loop.run_in_executor(
                self.executor, partial(env.reset, task_config=task_config)
            )
            await asyncio.sleep(self.env_reset_wait_secs)
            obs = await loop.run_in_executor(self.executor, env._get_obs)

            client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": OSWORLD_SYSTEM_PROMPT},
                self._build_user_turn(
                    f"Task instruction: {instruction}\nHere is the current screenshot.",
                    obs["screenshot"],
                ),
            ]

            terminated = False
            step_idx = 0
            last_response_id: str | None = None
            for step_idx in range(self.max_steps):
                response = await client.chat.completions.create(
                    messages=messages,
                    **self.gconfig.to_openai_args_dict(),
                )
                last_response_id = response.id
                reply = response.choices[0].message
                reply_text = reply.content or ""
                messages.append(reply.model_dump(exclude_none=True))

                actions = _parse_actions(reply_text)
                if not actions:
                    logger.warning(
                        f"[task={task_id} traj={traj_idx}] step {step_idx + 1}: "
                        "no parseable action; feeding back a reminder."
                    )
                    messages.append(
                        self._build_user_turn(
                            "Your previous reply did not contain a valid pyautogui "
                            "code block or control token. Please issue an action.",
                            obs["screenshot"],
                        )
                    )
                    continue

                for action in actions:
                    obs, _, done, info = await loop.run_in_executor(
                        self.executor,
                        partial(env.step, action, self.sleep_after_execution),
                    )
                    if done:
                        terminated = True
                        break
                if terminated:
                    break
                messages.append(
                    self._build_user_turn(
                        f"Step {step_idx + 1} executed. Here is the new screenshot.",
                        obs["screenshot"],
                    )
                )

            final_reward = float(
                await loop.run_in_executor(self.executor, env.evaluate)
            )
            logger.info(
                f"[task={task_id} traj={traj_idx}] finished after "
                f"{step_idx + 1} steps, reward={final_reward:.3f}"
            )

            stats_tracker.get(self.rollout_stat_scope).scalar(
                reward=final_reward, num_steps=step_idx + 1
            )

            if last_response_id is not None:
                client.set_reward(last_response_id, final_reward)
            client.apply_reward_discount(turn_discount=self.turn_discount)
            completions = client.export_interactions(style="individual")

            # Inject multimodal training tensors when running against a Qwen-VL
            # base. ArealOpenAI's cache only stores text token / logp / reward
            # data; FSDPEngine._prepare_mb_list (areal/engine/fsdp_engine.py)
            # additionally requires `mm_token_type_ids` and `multi_modal_input`
            # for any `is_qwen_vl_model`. We re-run the HF processor on each
            # turn's prefix to recover those fields and stash them on the
            # interaction's `_cache` so the downstream `to_tensor_dict()` call
            # returns them as-is. Skipped in text_only mode.
            if completions and not self.text_only:
                self._attach_vl_tensor_dicts(completions)

            if self.dump_dir is not None:
                self._dump_trajectory(task_id, traj_idx, messages, final_reward)

            return completions, final_reward
        except Exception as e:
            logger.error(f"[task={task_id} traj={traj_idx}] trajectory failed: {e!r}")
            return None, None
        finally:
            if env is not None:
                try:
                    await loop.run_in_executor(self.executor, env.close)
                except Exception as e:
                    logger.warning(f"Failed to close env: {e!r}")

    def _dump_trajectory(
        self,
        task_id: str,
        traj_idx: int,
        messages: list[dict[str, Any]],
        reward: float,
    ) -> None:
        # Strip base64 image bytes before dumping so the log stays readable.
        def _sanitize(msg: dict[str, Any]) -> dict[str, Any]:
            if isinstance(msg.get("content"), list):
                parts = []
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        parts.append({"type": "image_url", "image_url": "<elided>"})
                    else:
                        parts.append(part)
                return {**msg, "content": parts}
            return msg

        path = os.path.join(
            self.dump_dir, f"{task_id}_traj{traj_idx}_{uuid.uuid4().hex[:6]}.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_id": task_id,
                    "traj_idx": traj_idx,
                    "reward": reward,
                    "messages": [_sanitize(m) for m in messages],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ------------------------------------------------------------------
    # VL bridge: hand FSDPEngine the multimodal tensors it expects.
    # ------------------------------------------------------------------

    def _get_processor(self):
        """Lazy-load HF processor for the VL base model."""
        if self._processor is None:
            path = self._processor_path
            if not path:
                raise RuntimeError(
                    "OSWorldWorkflow needs `processor_path` set when "
                    "text_only=False — point it at the same HuggingFace dir "
                    "as actor.path so we can recover mm_token_type_ids etc."
                )
            self._processor = AutoProcessor.from_pretrained(path)
        return self._processor

    @staticmethod
    def _decode_data_uri_image(url: str) -> Image.Image | None:
        if not url.startswith("data:image"):
            return None
        try:
            _, b64 = url.split(",", 1)
        except ValueError:
            return None
        try:
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to decode image data URI: {e!r}")
            return None

    def _split_messages_to_text_and_images(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        """Convert workflow messages → processor messages + PIL image list.

        Workflow messages carry images as ``image_url`` data URIs; the HF
        processor wants ``{"type": "image"}`` placeholders alongside an
        external ``images=`` list. Iterate the conversation, peel out each
        decoded image, replace the part with the placeholder.
        """
        out_messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                # Plain string content (system / assistant text) — pass through.
                out_messages.append(msg)
                continue
            new_parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    img = self._decode_data_uri_image(
                        part.get("image_url", {}).get("url", "")
                    )
                    if img is None:
                        # Drop unreadable images rather than mis-aligning the
                        # token stream.
                        continue
                    images.append(img)
                    new_parts.append({"type": "image"})
                elif part.get("type") == "text":
                    new_parts.append({"type": "text", "text": part.get("text", "")})
                # ignore other part types (audio, video, etc.)
            out_messages.append({"role": msg["role"], "content": new_parts})
        return out_messages, images

    def _build_vl_tensor_dict(
        self, interaction
    ) -> dict[str, torch.Tensor | list] | None:
        """Run processor on the interaction prefix; produce a VL tensor dict.

        Returns a dict shaped like ``InteractionWithTokenLogpReward.to_tensor_dict``
        plus the multimodal extras (``mm_token_type_ids``, ``multi_modal_input``).
        Returns ``None`` if the interaction has no ``model_response`` (we can't
        contribute output tokens) — caller should fall back to the text-only
        export.
        """
        resp = getattr(interaction, "model_response", None)
        if resp is None:
            return None
        prefix_messages = list(interaction.messages or [])
        proc_messages, images = self._split_messages_to_text_and_images(prefix_messages)
        processor = self._get_processor()

        text = processor.apply_chat_template(
            proc_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs: dict[str, Any] = dict(text=[text], padding=False, return_tensors="pt")
        if images:
            kwargs["images"] = images
        proc_out = processor(**kwargs)

        input_ids = proc_out["input_ids"][0].tolist()
        if "mm_token_type_ids" in proc_out:
            mm_token_type_ids = proc_out["mm_token_type_ids"][0].tolist()
        else:
            mm_token_type_ids = [0] * len(input_ids)

        output_ids = list(resp.output_tokens or [])
        output_logprobs = list(resp.output_logprobs or [])
        output_versions = list(resp.output_versions or [])
        # Defensive: pad / truncate the per-token streams to match output_ids.
        # SGLang occasionally returns short logprob arrays; avoid blowing up
        # downstream tensor builds when that happens.
        if len(output_logprobs) < len(output_ids):
            output_logprobs += [0.0] * (len(output_ids) - len(output_logprobs))
        else:
            output_logprobs = output_logprobs[: len(output_ids)]
        if len(output_versions) < len(output_ids):
            output_versions += [-1] * (len(output_ids) - len(output_versions))
        else:
            output_versions = output_versions[: len(output_ids)]

        seq = input_ids + output_ids
        full_mm = mm_token_type_ids + [0] * len(output_ids)
        loss_mask = [0] * len(input_ids) + [1] * len(output_ids)
        logprobs = [0.0] * len(input_ids) + output_logprobs
        versions = [-1] * len(input_ids) + output_versions

        multi_modal_input: list[dict[str, torch.Tensor]] = []
        if images and "pixel_values" in proc_out:
            entry: dict[str, torch.Tensor] = {"pixel_values": proc_out["pixel_values"]}
            if "image_grid_thw" in proc_out:
                entry["image_grid_thw"] = proc_out["image_grid_thw"]
            multi_modal_input.append(entry)

        reward = float(interaction.reward) if interaction.reward is not None else 0.0
        return {
            "input_ids": torch.tensor(seq, dtype=torch.long).unsqueeze(0),
            "mm_token_type_ids": torch.tensor(full_mm, dtype=torch.long).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([reward], dtype=torch.float32),
            "multi_modal_input": multi_modal_input,
        }

    def _attach_vl_tensor_dicts(self, completions: dict[str, Any]) -> None:
        """Pre-populate each interaction's `_cache` with VL-augmented tensors.

        ``InteractionWithTokenLogpReward.to_tensor_dict()`` short-circuits and
        returns ``self._cache`` if set, so this is a non-invasive override.
        """
        bad_ids: list[str] = []
        for iid, interaction in completions.items():
            try:
                tensor_dict = self._build_vl_tensor_dict(interaction)
            except Exception as e:
                logger.warning(f"VL tensor build failed for interaction {iid}: {e!r}")
                tensor_dict = None
            if tensor_dict is None:
                bad_ids.append(iid)
                continue
            interaction._cache = tensor_dict
        for iid in bad_ids:
            # Drop interactions we couldn't bridge — they'd otherwise crash
            # FSDPEngine on missing `mm_token_type_ids`.
            completions.pop(iid, None)

    async def arun_episode(self, engine, data: dict[str, Any]):
        task_config = self._load_task_config(data)

        results = await asyncio.gather(
            *[
                self._run_single_trajectory(engine, task_config, i)
                for i in range(self.n_trajs)
            ]
        )

        stats_tracker.get(self.rollout_stat_scope).scalar(
            num_trajectories_failed=sum(1 for r in results if r[0] is None),
            num_full_passes=sum(1 for r in results if r[1] is not None and r[1] >= 1.0),
        )

        merged: dict[str, Any] = {}
        for completions, _ in results:
            if completions:
                merged.update(completions)

        if not merged:
            logger.warning(f"All trajectories failed for task {task_config.get('id')}.")
            return None
        return merged
