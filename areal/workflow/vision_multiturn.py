# SPDX-License-Identifier: Apache-2.0

"""Env-driven multi-turn tool-calling RL workflow for vision-language models.

The workflow is task-agnostic: it processes the image once (turn 0), then loops
generate -> ``env.step(assistant_text)`` -> append the env's text observation as
the next user turn, until the env signals ``done`` or ``max_turns`` is reached.
All task semantics (initial prompt incl. tool/system instructions, action
parsing, grading, feedback, termination) live in a ``MultiTurnVisionEnv``
(e.g. ``examples/multi_turn_vlm/geo3k_env.py``).

The VLM trajectory mechanics (single fixed image / 1-element multi_modal_input,
``mm_token_type_ids`` threading for the FSDP mRoPE path, the chat-template delta
tokenization, and the vLLM ``messages_chat`` sync) are preserved from the
validated single-turn / retry path.

NOTE: the remote engine echoes the request's ``input_ids`` back as
``resp.input_tokens`` and never returns the server's chat render, so token-path
/ chat-path consistency rests on this workflow rendering both from the same
``messages_chat`` with the model's own chat template.
"""

import uuid
from collections.abc import Callable
from typing import Any, cast

import torch
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, ModelResponse, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.image import image2base64
from areal.utils.perf_tracer import atrace_session_phase, session_context
from areal.workflow.vision_env import EnvStepResult, MultiTurnVisionEnv

logger = logging.getLogger("VisionMultiTurnWorkflow")


class VisionMultiTurnWorkflow(RolloutWorkflow):
    """Env-driven multi-turn tool-calling workflow for vision-language models."""

    def __init__(
        self,
        env_factory: Callable[..., MultiTurnVisionEnv] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        processor: AutoProcessor | str,
        env_args: dict[str, Any] | None = None,
        max_turns: int = 2,
        turn_discount: float = 1.0,
        max_tokens_per_traj: int | None = None,
        export_style: str = "concat",
    ):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if not (0.0 < turn_discount <= 1.0):
            raise ValueError("turn_discount must be in (0, 1].")

        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        if isinstance(processor, str):
            processor = AutoProcessor.from_pretrained(processor)
        self.processor = processor

        self.env_factory = env_factory
        self.env_args = env_args or {}

        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer).new(
            n_samples=1
        )
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        # Hard cap on trajectory length so a multi-turn sequence always fits in
        # one microbatch (a VLM trajectory's image binds it to a single mb and
        # cannot be split). Set to actor mb_spec.max_tokens_per_mb. None = no cap.
        self.max_tokens_per_traj = max_tokens_per_traj
        if export_style not in ("concat", "individual"):
            raise ValueError("export_style must be 'concat' or 'individual'")
        self.export_style = export_style

        # Generated image/video pad tokens would desync the placeholder count
        # from the supplied images (the processor re-render and the VLM forward
        # both count them), so they are dropped from outputs.
        self._banned_output_ids = set()
        for tok_str in ("<|image_pad|>", "<|video_pad|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and self.tokenizer.convert_ids_to_tokens(tid) == tok_str:
                self._banned_output_ids.add(tid)

        # The chat-template delta trick (and its prefix-consistency assumption) is
        # only used by the 'concat' path. Thinking models, whose template strips
        # prior <think>, break it and must use 'individual' export, which re-renders
        # each turn's context from messages_chat with the model's own template.
        if export_style == "concat":
            # A dynamic observation is tokenized as the delta s2[len(s1):]
            # (user turn + generation prompt, starting AFTER the assistant-turn
            # closer, which s1 owns). Prefer a lone assistant turn; some templates
            # require a user message, so fall back to a user+assistant pair.
            self._delta_placeholder, self._delta_s1 = self._build_delta_placeholder()
            # Self-test: a non-empty, prefix-consistent delta. Fails fast with an
            # actionable message for context-dependent templates.
            probe = self._tokenize_observation("probe")
            assert isinstance(probe, list) and len(probe) > 0, "empty observation delta"
            # Tokens the template emits to close an assistant turn (e.g. EOS +
            # "\n"); appended to the token path at turn boundaries so it matches
            # the server's chat render exactly.
            self._turn_closer_ids = self._build_turn_closer()

    def _build_delta_placeholder(self) -> tuple[list[dict[str, str]], list[int]]:
        for placeholder in (
            [{"role": "assistant", "content": "x"}],
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "x"}],
        ):
            try:
                s1 = list(
                    self.tokenizer.apply_chat_template(
                        placeholder, tokenize=True, return_dict=False
                    )
                )
            except Exception:
                continue
            return placeholder, s1
        raise ValueError(
            "The tokenizer's chat template is incompatible with the multi-turn "
            "delta placeholder (it rejects both an assistant and a user+assistant "
            "turn). Use a model with a standard chat template (e.g. Qwen3-VL)."
        )

    def _build_turn_closer(self) -> list[int]:
        """Tokens the chat template appends to close an assistant turn.

        Derived as render(placeholder) minus render(placeholder,
        continue_final_message=True); falls back to a bare EOS when the
        template does not support unclosed final messages.
        """
        closed = list(
            self.tokenizer.apply_chat_template(
                self._delta_placeholder, tokenize=True, return_dict=False
            )
        )
        try:
            opened = list(
                self.tokenizer.apply_chat_template(
                    self._delta_placeholder,
                    tokenize=True,
                    continue_final_message=True,
                    return_dict=False,
                )
            )
        except Exception:
            opened = None
        if opened and closed[: len(opened)] == opened and len(closed) > len(opened):
            return closed[len(opened) :]
        logger.warning(
            "could not derive the assistant-turn closer from the chat template; "
            "falling back to a bare EOS (the token path may miss template tokens "
            "at turn boundaries)"
        )
        return [self.tokenizer.eos_token_id]

    def _assistant_content(self, out_toks: list[int]) -> str:
        """Decoded assistant text for the chat history, without the trailing EOS
        (the template adds its own turn closer when re-rendering; a literal EOS
        inside the content would double it in the server prompt)."""
        text = self.tokenizer.decode(out_toks)
        eos = self.tokenizer.eos_token
        return text.removesuffix(eos) if eos else text

    def _filter_output(
        self, resp: ModelResponse
    ) -> tuple[list[int], list[float], list[int]]:
        """Output tokens with generated image/video pad tokens dropped (along
        with their logprob/version entries), keeping the placeholder count
        aligned with the supplied images."""
        toks = list(resp.output_tokens)
        if not self._banned_output_ids or not any(
            t in self._banned_output_ids for t in toks
        ):
            return toks, list(resp.output_logprobs), list(resp.output_versions)
        kept = [
            (t, lp, v)
            for t, lp, v in zip(toks, resp.output_logprobs, resp.output_versions)
            if t not in self._banned_output_ids
        ]
        logger.warning(
            f"dropped {len(toks) - len(kept)} generated image/video pad tokens"
        )
        return (
            [t for t, _, _ in kept],
            [lp for _, lp, _ in kept],
            [v for _, _, v in kept],
        )

    def _image_token(self) -> str:
        """Model-specific image placeholder, mirroring the dataset loader."""
        proc_type = self.processor.image_processor.image_processor_type.lower()
        if "qwen" in proc_type:
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if "gemma3" in proc_type:
            return self.processor.boi_token
        return getattr(self.processor, "image_token", "<image>")

    @staticmethod
    def _chat_to_struct(
        messages_chat: list[dict[str, Any]], image_token: str
    ) -> list[dict[str, str]]:
        """Flatten a vLLM-style chat list into tokenizer.apply_chat_template form.

        Each image content block becomes the model image placeholder text so the
        token path matches the dataset loader's construction.
        """
        struct: list[dict[str, str]] = []
        for msg in messages_chat:
            content = msg["content"]
            if isinstance(content, str):
                struct.append({"role": msg["role"], "content": content})
                continue
            parts: list[str] = []
            for c in content:
                ctype = c.get("type")
                if ctype == "text":
                    parts.append(c.get("text", ""))
                elif ctype in ("image_url", "image"):
                    parts.append(image_token)
            struct.append({"role": msg["role"], "content": "".join(parts)})
        return struct

    def _tokenize_observation(self, obs_text: str) -> list[int]:
        """Tokenize a dynamic user-turn observation as a chat-template delta."""
        s2 = list(
            self.tokenizer.apply_chat_template(
                self._delta_placeholder + [{"role": "user", "content": obs_text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )
        assert s2[: len(self._delta_s1)] == self._delta_s1, (
            "The model's chat template renders turns context-dependently (e.g. a "
            "thinking model that inserts/strips <think>), which breaks the "
            "multi-turn observation delta. Use a context-independent chat template "
            "(e.g. Qwen3-VL)."
        )
        return s2[len(self._delta_s1) :]

    @session_context()
    async def _collect(
        self, engine: InferenceEngine, req: ModelRequest
    ) -> ModelResponse:
        async with atrace_session_phase("generate"):
            return await engine.agenerate(req)

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # Resolve env_factory lazily (in the rollout worker), one env per episode.
        if isinstance(self.env_factory, str):
            self.env_factory = import_from_string(self.env_factory)
        env: MultiTurnVisionEnv = self.env_factory(**self.env_args)

        reset_out = env.reset(data)
        messages_chat = reset_out.messages_chat
        images = reset_out.images
        if not images:
            raise ValueError("env.reset must return non-empty images for VLM workflow")

        # Image-block / image count must match (fail fast at the workflow boundary).
        n_img_blocks = sum(
            1
            for m in messages_chat
            if isinstance(m["content"], list)
            for c in m["content"]
            if c.get("type") in ("image_url", "image")
        )
        assert n_img_blocks == len(images), (n_img_blocks, len(images))

        # Turn 0: derive the token-path prompt string from the same messages_chat
        # used for the vLLM chat path, so both paths carry the identical system /
        # tool instructions and stay consistent.
        image_token = self._image_token()
        messages_str = self.tokenizer.apply_chat_template(
            self._chat_to_struct(messages_chat, image_token),
            add_generation_prompt=True,
            tokenize=False,
        )
        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)
        processed_input = processor_callable(
            images=images,
            text=messages_str,
            padding=False,
            return_tensors="pt",
        )
        input_ids: list[int] = processed_input["input_ids"].tolist()[0]
        mm_token_type_ids: list[int] = processed_input["mm_token_type_ids"].tolist()[0]
        byte_images = image2base64(images)
        turn0_len = len(input_ids)

        if self.export_style == "individual":
            return await self._build_individual_episode(
                engine,
                env,
                messages_chat,
                images,
                byte_images,
                image_token,
                input_ids,
                mm_token_type_ids,
                processed_input,
            )

        # Accumulators for the single trajectory.
        seq = input_ids.copy()
        logprobs = [0.0] * len(input_ids)
        loss_mask = [0] * len(input_ids)
        versions = [-1] * len(input_ids)
        reward = 0.0
        discount = 1.0
        turn = 0

        for turn in range(self.max_turns):
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=seq,
                image_data=byte_images,
                vision_msg_vllm=[messages_chat] if messages_chat else None,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            resp = await self._collect(engine, req)
            out_toks, out_logprobs, out_versions = self._filter_output(resp)

            seq += out_toks
            logprobs += out_logprobs
            loss_mask += [1] * len(out_toks)
            versions += out_versions
            mm_token_type_ids += [0] * len(out_toks)

            # Step the environment with the decoded assistant output.
            assistant_text = self.tokenizer.decode(out_toks)
            step_out: EnvStepResult = env.step(assistant_text)
            turn_reward = step_out.reward

            # Best discounted reward across turns (turn_discount=1.0 -> flat terminal).
            if turn == 0:
                reward = turn_reward * discount
            else:
                reward = max(reward, turn_reward * discount)

            if step_out.done or turn == self.max_turns - 1:
                break
            # Stop before another turn could push the trajectory past the
            # microbatch budget (+128 covers the appended EOS/observation tokens).
            if (
                self.max_tokens_per_traj is not None
                and len(seq) + self.gconfig.max_new_tokens + 128
                > self.max_tokens_per_traj
            ):
                break
            obs_text = step_out.observation
            if obs_text is None:
                break  # defensive: env should set done=True when no observation

            # Close the assistant turn exactly as the chat template does (EOS +
            # any trailing tokens, e.g. "\n"), so the appended observation delta
            # starts at the same tokens the server renders next turn.
            closer = list(self._turn_closer_ids)
            if (
                out_toks
                and out_toks[-1] == self.tokenizer.eos_token_id
                and closer
                and closer[0] == self.tokenizer.eos_token_id
            ):
                closer = closer[1:]  # natural stop already emitted the EOS
            seq += closer
            logprobs += [0.0] * len(closer)
            loss_mask += [0] * len(closer)
            versions += [-1] * len(closer)
            mm_token_type_ids += [0] * len(closer)
            if messages_chat:
                messages_chat = messages_chat + [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": self._assistant_content(out_toks),
                            }
                        ],
                    }
                ]

            # Append the dynamic observation as a non-trainable user turn.
            obs_text = obs_text.strip()
            obs_ids = self._tokenize_observation(obs_text)
            seq.extend(obs_ids)
            logprobs += [0.0] * len(obs_ids)
            loss_mask += [0] * len(obs_ids)
            versions += [-1] * len(obs_ids)
            mm_token_type_ids += [0] * len(obs_ids)
            if messages_chat:
                messages_chat = messages_chat + [
                    {"role": "user", "content": [{"type": "text", "text": obs_text}]}
                ]

            discount *= self.turn_discount

        metrics = {"reward": reward, **env.get_metrics()}
        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)

        # Invariants: all accumulators length-aligned, and image pads (non-zero
        # mm_token_type_ids) confined to the turn-0 prefix (text-only obs after).
        assert len(mm_token_type_ids) == len(seq), (len(mm_token_type_ids), len(seq))
        assert all(t == 0 for t in mm_token_type_ids[turn0_len:]), (
            "non-zero mm_token_type_ids appended after turn 0"
        )

        # Single fixed image -> 1-element multi_modal_input.
        multi_modal_input = [{"pixel_values": processed_input["pixel_values"]}]
        if "image_grid_thw" in processed_input:
            multi_modal_input[0]["image_grid_thw"] = processed_input["image_grid_thw"]

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long).unsqueeze(0),
            "mm_token_type_ids": torch.tensor(
                mm_token_type_ids, dtype=torch.long
            ).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "multi_modal_input": multi_modal_input,
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
        }

    async def _build_individual_episode(
        self,
        engine: InferenceEngine,
        env: MultiTurnVisionEnv,
        messages_chat: list[dict[str, Any]],
        images: list[Any],
        byte_images: list[str],
        image_token: str,
        turn0_input_ids: list[int],
        turn0_mm_token_type_ids: list[int],
        processed_input: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Per-turn ('individual') export for thinking models.

        Each turn becomes an independent training sample. The engine does not
        return the server's tokenization (``resp.input_tokens`` echoes the
        request), so each turn's context is rendered locally from the same
        ``messages_chat`` the vLLM chat path templates server-side: the model's
        own chat template strips prior ``<think>`` and inserts the observation
        turns, and the processor expands the image placeholder — keeping the
        trained prefix aligned with the generation prefix (and supplying the
        processor's exact ``mm_token_type_ids`` per row). The fixed image is
        attached to every per-turn sample; the episode reward is broadcast to
        all turns (outcome credit).
        """
        image_dict: dict[str, Any] = {"pixel_values": processed_input["pixel_values"]}
        if "image_grid_thw" in processed_input:
            image_dict["image_grid_thw"] = processed_input["image_grid_thw"]
        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)

        rows: list[dict[str, list]] = []
        in_toks = list(turn0_input_ids)
        mm_in = list(turn0_mm_token_type_ids)
        reward = 0.0
        discount = 1.0
        for turn in range(self.max_turns):
            if turn > 0:
                # Re-render the grown conversation (think-stripping applied by
                # the template; image pads expanded by the processor) — the same
                # construction the server applies to these messages.
                messages_str = self.tokenizer.apply_chat_template(
                    self._chat_to_struct(messages_chat, image_token),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                processed = processor_callable(
                    images=images,
                    text=messages_str,
                    padding=False,
                    return_tensors="pt",
                )
                in_toks = processed["input_ids"].tolist()[0]
                mm_in = processed["mm_token_type_ids"].tolist()[0]
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=in_toks,
                image_data=byte_images,
                vision_msg_vllm=[messages_chat] if messages_chat else None,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            resp = await self._collect(engine, req)
            out_toks, out_logprobs, out_versions = self._filter_output(resp)
            n_in, n_out = len(in_toks), len(out_toks)
            assert len(mm_in) == n_in, (len(mm_in), n_in)
            rows.append(
                {
                    "input_ids": in_toks + out_toks,
                    "loss_mask": [0] * n_in + [1] * n_out,
                    "logprobs": [0.0] * n_in + out_logprobs,
                    "versions": [-1] * n_in + out_versions,
                    "mm_token_type_ids": mm_in + [0] * n_out,
                }
            )

            assistant_text = self.tokenizer.decode(out_toks)
            step_out = env.step(assistant_text)
            turn_reward = step_out.reward
            reward = (
                turn_reward * discount
                if turn == 0
                else max(reward, turn_reward * discount)
            )

            if step_out.done or turn == self.max_turns - 1:
                break
            if (
                self.max_tokens_per_traj is not None
                and n_in + n_out + self.gconfig.max_new_tokens + 128
                > self.max_tokens_per_traj
            ):
                break
            obs_text = step_out.observation
            if obs_text is None:
                break
            messages_chat = messages_chat + [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": self._assistant_content(out_toks)}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": obs_text.strip()}],
                },
            ]
            discount *= self.turn_discount

        metrics = {"reward": reward, **env.get_metrics()}
        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)

        # Stack the per-turn rows into [K, max_len], right-padded; attention_mask
        # marks real tokens. concat_padded_tensors re-pads across episodes later.
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        k = len(rows)
        max_len = max(len(r["input_ids"]) for r in rows)

        def stack(key: str, value: float, dtype: torch.dtype) -> torch.Tensor:
            padded = [r[key] + [value] * (max_len - len(r[key])) for r in rows]
            return torch.tensor(padded, dtype=dtype)

        attn = [
            [True] * len(r["input_ids"]) + [False] * (max_len - len(r["input_ids"]))
            for r in rows
        ]
        return {
            "input_ids": stack("input_ids", pad_id, torch.long),
            "mm_token_type_ids": stack("mm_token_type_ids", 0, torch.long),
            "loss_mask": stack("loss_mask", 0, torch.int32),
            "logprobs": stack("logprobs", 0.0, torch.float32),
            "multi_modal_input": [image_dict for _ in range(k)],
            "versions": stack("versions", -1, torch.int32),
            "attention_mask": torch.tensor(attn, dtype=torch.bool),
            "rewards": torch.tensor([reward] * k, dtype=torch.float32),
        }
