# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import re
from enum import Enum
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Controlled at runtime via AgentLoopConfig.verbose; default keeps current behavior.
print_debug = print


class GenerationState(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DONE = "done"


class ParallelBranchingController:
    """Orchestrate seq→par→seq rollouts on top of the existing vLLM server."""

    OUTLINE_PATTERN = re.compile(r"<(?:Outline|Subtask|Trial)>\s*([0-9]+(?:\.[0-9]+)*)\s*:")

    def __init__(
        self,
        *,
        tokenizer,
        server_manager,
        base_sampling_params: Dict[str, Any],
        request_id: str,
        prompt_length: int,
        response_length: int,
        max_total_tokens: Optional[int],
        no_conclusion: bool,
        metrics: Dict[str, float],
    ) -> None:
        self.tokenizer = tokenizer
        self.server_manager = server_manager
        self.request_id = request_id
        self.prompt_length = prompt_length
        self.response_length = response_length
        self.metrics = metrics
        self.no_conclusion = no_conclusion

        self.base_sampling_params = dict(base_sampling_params)
        # Branch control does not surface per-token logprobs; disable upstream request to save work.
        self.base_sampling_params.pop("logprobs", None)

        self.max_total_tokens = (
            max_total_tokens if max_total_tokens is not None else prompt_length + response_length
        )
        print_debug(f"[Branching] controller initialized | prompt_length={prompt_length} | response_length={response_length} | max_total_tokens={self.max_total_tokens}")
        self.special_tokens = self._build_special_token_map()

        self.current_tokens: List[int] = []
        self.generated_tokens: List[int] = []
        self.generated_blocks: List[Dict[str, Any]] = []
        self.expanded_prompt_ids: List[List[int]] = []
        self.expanded_response_ids: List[List[int]] = []
        self.expanded_response_mask: List[List[int]] = []
        self.state = GenerationState.SEQUENTIAL
        self.outline_nums: List[str] = []
        self.eos_seen = False
        self.block_counter = 0

    def _build_special_token_map(self) -> Dict[str, int]:
        def _get_id(token: str) -> int:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Special token '{token}' must map to exactly one token id; got {token_ids}")
            return token_ids[0]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must define eos_token_id for parallel branching mode")

        return {
            "outlines_start_id": _get_id("<Outlines>"),
            "outlines_end_id": _get_id("</Outlines>"),
            "thread_end_id": _get_id("</Thread>"),
            "eos_id": eos_id,
        }

    async def run(
        self, prompt_tokens: List[int]
    ) -> Tuple[
        List[int],
        List[Dict[str, Any]],
        List[List[int]],
        List[List[int]],
        List[List[int]],
    ]:
        self.current_tokens = list(prompt_tokens)
        self.generated_tokens = []
        self.generated_blocks = []
        self.expanded_prompt_ids = []
        self.expanded_response_ids = []
        self.expanded_response_mask = []
        self.state = GenerationState.SEQUENTIAL
        self.outline_nums = []
        self.eos_seen = False
        self.block_counter = 0

        max_iterations = 32  # safeguard against pathological loops
        iteration = 0

        print_debug(
            f"[Branching] start run | prompt_tokens={len(prompt_tokens)} | max_response={self.response_length}"
        )

        while self.state != GenerationState.DONE:
            iteration += 1
            if iteration > max_iterations:
                logger.warning("Parallel branching controller reached max iterations; terminating early.")
                print_debug("[Branching] reached max iterations; stopping controller")
                break

            if len(self.generated_tokens) >= self.response_length:
                print_debug("[Branching] generated response reached configured length; stopping")
                break

            if len(self.current_tokens) >= self.max_total_tokens:
                print_debug("[Branching] context hit max_total_tokens; stopping")
                break

            if self.state == GenerationState.SEQUENTIAL:
                print_debug(
                    f"[Branching] iteration={iteration} entering SEQUENTIAL | ctx={len(self.current_tokens)}"
                )
                await self._sequential_step()
            elif self.state == GenerationState.PARALLEL:
                print_debug(
                    f"[Branching] iteration={iteration} entering PARALLEL | ctx={len(self.current_tokens)}"
                )
                await self._parallel_step()
            else:
                break

        return (
            self.generated_tokens[: self.response_length],
            list(self.generated_blocks),
            [list(ids) for ids in self.expanded_prompt_ids],
            [list(ids) for ids in self.expanded_response_ids],
            [list(mask) for mask in self.expanded_response_mask],
        )

    async def _sequential_step(self) -> None:
        stop_ids = [self.special_tokens["outlines_end_id"], self.special_tokens["eos_id"]]
        print_debug(
            f"[Branching] sequential step | stop_ids={stop_ids} | ctx_len={len(self.current_tokens)}"
        )
        block_tag = self._start_new_block()
        output = await self._generate(self.current_tokens, stop_ids, suffix=block_tag)

        tokens = list(output.token_ids)
        if not tokens:
            self.state = GenerationState.DONE
            print_debug(f"[Branching] sequential produced no tokens; marking DONE at {len(self.generated_tokens)} tokens")
            return

        self._record_block("sequential", tokens)
        self.current_tokens.extend(tokens)
        self.generated_tokens.extend(tokens)
        print_debug(
            f"[Branching] sequential produced {len(tokens)} tokens | finish_reason={output.finish_reason} | matched_stop={output.matched_stop}"
        )

        if self.special_tokens["outlines_end_id"] in tokens:
            outline_nums = self._extract_outline_numbers()
            if outline_nums:
                self.outline_nums = outline_nums
                self.state = GenerationState.PARALLEL
                self.eos_seen = False
                print_debug(f"[Branching] detected outlines {outline_nums}; switching to PARALLEL")
                # Note: this is not implemented, since the last sequential step only covers the tokens generated in the last sequential step (not the other sequential parts, which are in the prompt and do not participate in loss computation).
                # if self.keep_simplified_expanded_response:
                #     # If we only keep simplified expanded response, we discard the
                #     # expanded response for the sequential step, unless it's the last step (next state is DONE).
                #     call_idx = output.call_idx
                #     self.expanded_prompt_ids[call_idx] = []
                #     self.expanded_response_ids[call_idx] = []
                #     self.expanded_response_mask[call_idx] = []
                # return

        if (
            output.finish_reason in {"length", "eos_token"}
            or output.matched_stop == self.special_tokens["eos_id"]
            or self.special_tokens["eos_id"] in tokens
        ):
            # We always keep the expanded response for the last sequential step.
            self.state = GenerationState.DONE
            print_debug(f"[Branching] sequential hit EOS/length (with {output.finish_reason=}); marking DONE at {len(self.generated_tokens)} tokens")

    async def _parallel_step(self) -> None:
        if not self.outline_nums:
            self.state = GenerationState.SEQUENTIAL
            print_debug("[Branching] parallel step found no outlines; returning to SEQUENTIAL")
            return

        outlines = list(dict.fromkeys(self.outline_nums))
        if len(outlines) != len(self.outline_nums):
            print_debug(
                f"[Branching] deduplicated outlines from {self.outline_nums} to {outlines}"
            )
        self.outline_nums = outlines

        headers: List[List[int]] = []
        tasks = []
        print_debug(f"[Branching] parallel step spawning {len(self.outline_nums)} branches")
        block_tag = self._start_new_block()
        for idx, outline in enumerate(self.outline_nums):
            prompt_header_text = "\n<Thread>\n" + f"{outline}:"
            prompt_header_tokens = self._encode(prompt_header_text)
            branch_prompt = self.current_tokens + prompt_header_tokens

            # In the first branch, both prompt header and header is "\n<Thread>\n{outline}:".
            # For subsequent branches, the prompt header is the same, but the header is
            # "<Thread>\n{outline}:" so that the returned text has ... </Thread><Thread> ... (no newline between threads).
            if idx == 0:
                header_text = prompt_header_text
                header_tokens = prompt_header_tokens
            else:
                header_text = "<Thread>\n" + f"{outline}:"
                header_tokens = self._encode(header_text)
            headers.append(header_tokens)
            tasks.append(
                self._generate(
                    branch_prompt,
                    [self.special_tokens["thread_end_id"], self.special_tokens["eos_id"]],
                    suffix=f"{block_tag}_path_{outline}",
                )
            )

        branch_outputs = await asyncio.gather(*tasks)

        additional_tokens: List[int] = []
        current_block_entries: List[List[int]] = []
        for outline, header_tokens, output in zip(self.outline_nums, headers, branch_outputs):
            branch_tokens = list(output.token_ids)

            if output.finish_reason == "length" or output.matched_stop == self.special_tokens["eos_id"]:
                self.eos_seen = True
                print_debug(
                    f"[Branching] branch {outline} finished by length/eos | len={len(branch_tokens)}"
                )

            if self.special_tokens["eos_id"] in branch_tokens:
                eos_index = branch_tokens.index(self.special_tokens["eos_id"])
                branch_tokens = branch_tokens[:eos_index]
                self.eos_seen = True
                print_debug(f"[Branching] branch {outline} truncated at EOS index {eos_index}")

            # Ensure proper closure of <Thread> blocks. The following cases need closure:
            if output.matched_stop != self.special_tokens["thread_end_id"]:
                branch_tokens.extend(self._encode("\n</Thread>"))
                print_debug(f"[Branching] branch {outline} appended </Thread> for proper closure")

            combined = list(header_tokens) + branch_tokens
            additional_tokens.extend(combined)
            current_block_entries.append(combined.copy())

        self.current_tokens.extend(additional_tokens)
        self.generated_tokens.extend(additional_tokens)
        if current_block_entries:
            self._record_block("parallel", current_block_entries)
        print_debug(f"[Branching] merged branch tokens with outlines {self.outline_nums=} | added={len(additional_tokens)}")
        self.outline_nums = []

        if self.eos_seen or len(self.generated_tokens) >= self.response_length:
            if self.no_conclusion:
                closing = self._encode("\n</Parallel>")
            else:
                closing = self._encode("\n<Conclusion>\n</Conclusion>\n</Parallel>")
            self.current_tokens.extend(closing)
            self.generated_tokens.extend(closing)
            self._record_block("meta", closing)
            self.state = GenerationState.DONE
            print_debug(f"[Branching] closing parallel block; marking DONE at {len(self.generated_tokens)} tokens")
        else:
            if self.no_conclusion:
                continuation = self._encode("\n")
            else:
                continuation = self._encode("\n<Conclusion>\n")
            self.current_tokens.extend(continuation)
            self.generated_tokens.extend(continuation)
            self._record_block("meta", continuation)
            self.state = GenerationState.SEQUENTIAL
            if self.no_conclusion:
                print_debug("[Branching] appended newline continuation; returning to SEQUENTIAL")
            else:
                print_debug("[Branching] appended <Conclusion>; returning to SEQUENTIAL")

    async def _generate(
        self,
        prompt_tokens: List[int],
        stop_token_ids: List[int],
        *,
        suffix: Optional[str],
    ) -> Any:
        sampling_kw = dict(self.base_sampling_params)
        existing_stop_ids = sampling_kw.get("stop_token_ids")
        if existing_stop_ids:
            # Preserve order while ensuring uniqueness.
            merged = list(dict.fromkeys(list(existing_stop_ids) + list(stop_token_ids)))
            sampling_kw["stop_token_ids"] = merged
        else:
            sampling_kw["stop_token_ids"] = list(stop_token_ids)

        request_id = self._request_id(suffix)
        call_idx = self._register_expanded_call(prompt_tokens)
        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_tokens,
            sampling_params=sampling_kw,
        )
        output.call_idx = call_idx
        tokens = list(output.token_ids)
        output.token_ids = tokens
        if output.log_probs is not None:
            output.log_probs = list(output.log_probs)
        self.expanded_response_ids[call_idx] = list(tokens)
        self.expanded_response_mask[call_idx] = [1] * len(tokens)
        return output

    def _request_id(self, suffix: Optional[str]) -> str:
        if suffix is None:
            return self.request_id
        return f"{self.request_id}:{suffix}"

    def _start_new_block(self) -> str:
        self.block_counter += 1
        return str(self.block_counter)

    def _record_block(self, block_type: str, content: Any) -> None:
        """Append a generation block while copying token payloads."""
        if block_type == "parallel":
            block_content = [list(path) for path in content]
        else:
            block_content = list(content)
        self.generated_blocks.append({"type": block_type, "content": block_content})

    def _register_expanded_call(self, prompt_tokens: List[int]) -> int:
        self.expanded_prompt_ids.append(list(prompt_tokens))
        self.expanded_response_ids.append([])
        self.expanded_response_mask.append([])
        return len(self.expanded_prompt_ids) - 1

    def _extract_outline_numbers(self) -> List[str]:
        outlines_start_id = self.special_tokens["outlines_start_id"]
        try:
            last_outlines_start = len(self.current_tokens) - 1 - self.current_tokens[::-1].index(outlines_start_id)
        except ValueError:
            return []

        outlines_tokens = self.current_tokens[last_outlines_start:]
        outlines_text = self.tokenizer.decode(outlines_tokens)
        extracted_outlines = self.OUTLINE_PATTERN.findall(outlines_text)
        print_debug(f"[Branching] extracting outlines from outlines text: {outlines_text!r} -> {extracted_outlines}")
        return extracted_outlines

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure debug verbosity from config
        verbosity = int(self.config.actor_rollout_ref.rollout.agent.verbose)
        self._verbose = verbosity
        # Map verbosity to print_debug behavior
        global print_debug
        if self._verbose >= 1:
            print_debug = print
            # Optionally, raise logger level for more details if desired in future
        else:
            print_debug = lambda *args, **kwargs: None

        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        configured_max_total = self.config.actor_rollout_ref.rollout.max_model_len
        if configured_max_total is not None:
            self.max_total_tokens = configured_max_total
        elif self.prompt_length is not None and self.response_length is not None:
            self.max_total_tokens = self.prompt_length + self.response_length
        else:
            self.max_total_tokens = None
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )

        completions: List[int]
        log_probs: Optional[List[float]] = None
        generated_blocks: Optional[List[Dict[str, Any]]] = None
        expanded_prompt_ids: Optional[List[List[int]]] = None
        expanded_response_ids: Optional[List[List[int]]] = None
        expanded_response_mask: Optional[List[List[int]]] = None

        if self.config.actor_rollout_ref.rollout.agent.enable_parallel_branching:
            try:
                controller = ParallelBranchingController(
                    tokenizer=self.tokenizer,
                    server_manager=self.server_manager,
                    base_sampling_params=sampling_params,
                    request_id=request_id,
                    prompt_length=self.prompt_length,
                    response_length=self.response_length,
                    max_total_tokens=self.max_total_tokens,
                    no_conclusion=self.config.actor_rollout_ref.rollout.agent.no_conclusion,
                    metrics=metrics,
                )
                with simple_timer("generate_sequences", metrics):
                    (
                        completions,
                        generated_blocks,
                        expanded_prompt_ids,
                        expanded_response_ids,
                        expanded_response_mask,
                    ) = await controller.run(prompt_ids)
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning(
                    "Parallel branching controller failed (%s); falling back to single pass generation.",
                    exc,
                )
                (
                    completions,
                    log_probs,
                    generated_blocks,
                    expanded_prompt_ids,
                    expanded_response_ids,
                    expanded_response_mask,
                ) = await self._fallback_generate(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    metrics=metrics,
                )
        else:
            (
                completions,
                log_probs,
                generated_blocks,
                expanded_prompt_ids,
                expanded_response_ids,
                expanded_response_mask,
            ) = await self._fallback_generate(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                metrics=metrics,
            )

        allowed_response_len = min(self.response_length, self.max_total_tokens - len(prompt_ids)) if self.max_total_tokens is not None else self.response_length
        assert allowed_response_len >= 0, f"allowed_response_len={allowed_response_len} | prompt_len={len(prompt_ids)} | max_total_tokens={self.max_total_tokens} | response_length={self.response_length}"

        response_ids = completions[: allowed_response_len]
        response_mask = self._make_response_mask(len(response_ids))

        # Fallback to default values if any of the expanded_* is None or empty: this is not needed since it likely will not happen.
        # if not expanded_prompt_ids:
        #     expanded_prompt_ids = [list(prompt_ids)]

        # if not expanded_response_ids:
        #     expanded_response_ids = [list(response_ids)]

        # if not expanded_response_mask:
        #     expanded_response_mask = [self._make_response_mask(len(response_ids))]

        for idx in range(len(expanded_response_ids)):
            expanded_allowed_response_len = min(
                self.response_length,
                self.max_total_tokens - len(expanded_prompt_ids[idx])
            ) if self.max_total_tokens is not None else self.response_length

            if len(expanded_response_ids[idx]) > expanded_allowed_response_len:
                print_debug(f"[AgentLoop] Truncating expanded response {idx} from {len(expanded_response_ids[idx])} to {expanded_allowed_response_len}. Prompt content: {self.tokenizer.decode(expanded_prompt_ids[idx], skip_special_tokens=False)}. Response content: {self.tokenizer.decode(expanded_response_ids[idx], skip_special_tokens=False)}")
    
            expanded_response_ids[idx] = expanded_response_ids[idx][:expanded_allowed_response_len]
            expanded_response_mask[idx] = expanded_response_mask[idx][:expanded_allowed_response_len]
        
        assert len(response_ids) == len(response_mask), f"response_ids and response_mask length mismatch: {len(response_ids)} vs {len(response_mask)}"

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=log_probs[: allowed_response_len] if log_probs else None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            generated_blocks=generated_blocks,
            expanded_prompt_ids=expanded_prompt_ids,
            expanded_response_ids=expanded_response_ids,
            expanded_response_mask=expanded_response_mask,
        )

        # Expose generated_blocks through extra_fields so it is carried in
        # DataProto.non_tensor_batch and available to downstream savers.
        if kwargs.get("return_generated_blocks", False):
            if self._verbose >= 1:
                print(f"Returning {len(generated_blocks)} generated_blocks through extra_fields")
            output.extra_fields["generated_blocks"] = generated_blocks

        output.extra_fields.setdefault("_base_non_tensor", kwargs)

        return output

    @staticmethod
    def _make_response_mask(length: int) -> List[int]:
        return [1] * length

    async def _fallback_generate(
        self,
        *,
        prompt_ids: List[int],
        sampling_params: dict[str, Any],
        request_id: str,
        metrics: Dict[str, float],
    ) -> tuple[
        List[int],
        Optional[List[float]],
        Optional[List[Dict[str, Any]]],
        List[List[int]],
        List[List[int]],
        List[List[int]],
    ]:
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )
        tokens = list(output.token_ids)
        log_probs = list(output.log_probs) if output.log_probs is not None else None
        generated_blocks = output.generated_blocks
        if generated_blocks is None and tokens:
            generated_blocks = [{"type": "sequential", "content": list(tokens)}]
        expanded_prompt_ids = [list(prompt_ids)]
        expanded_response_ids = [list(tokens)]
        expanded_response_mask = [[1] * len(tokens)]
        return (
            tokens,
            log_probs,
            generated_blocks,
            expanded_prompt_ids,
            expanded_response_ids,
            expanded_response_mask,
        )
