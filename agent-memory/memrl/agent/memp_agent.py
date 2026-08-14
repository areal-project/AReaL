# FILE: memp/agent/memp_agent.py

import logging
from typing import List, Dict, Any, Tuple, Optional
import copy
import ast
import json
import re
import os

from .base import BaseAgent
from .history import EpisodeHistory
from . import prompts
from memrl.providers.llm import OpenAILLM

logger = logging.getLogger(__name__)

# Returned only when the transport produced a non-empty but unusable completion.
# Runners must treat this as an infrastructure event, not as a model decision.
INVALID_LLM_ACTION = "__memrl_invalid_llm_response__"

class MempAgent(BaseAgent):
    """
    A stateless agent that uses an LLM to make decisions.
    It receives all necessary context (history, retrieved memories) from an
    external controller (the Runner) at the moment of action.
    """
    def __init__(self, llm_provider: OpenAILLM, few_shot_examples: Dict[str, Any],
                 max_recent_turns: int = 20, max_history_response_chars: int = 0,
                 no_think: bool = False, force_think: bool = False):
        # The agent is now independent of the memory service.
        self.llm = llm_provider
        self.few_shot_examples = few_shot_examples
        self.max_recent_turns = max_recent_turns
        self._max_history_response_chars = max_history_response_chars
        self._no_think = no_think
        self._force_think = force_think
        self.prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }

    def reset(self, task_description: str) -> None:
        """Resets the agent for a new episode and retrieves relevant long-term memories."""
        self.task_description = task_description.strip()
        logger.info(f"Agent has been reset for new task: '{self.task_description}'")
        
    def _get_examples_for_task(self, task_type: str) -> str:
        """
        [NEW] Selects the relevant few-shot examples based on the task type.
        """
        for prefix, key in self.prefixes.items():
            if task_type.startswith(prefix):
                # This logic mirrors your example script: load two relevant examples
                for example in self.few_shot_examples:
                    if example['task'] == key:
                        return copy.deepcopy(example['example'])
        return "No specific examples found for this task type."

    def _split_retrieved_memory_content(self, raw_content: str) -> Tuple[str, str, str]:
        """Split a retrieved memory into header/body and classify its body."""
        for marker, body_type in (
            ("\n\nCOMPACT_TRAJECTORY_V1:\n", "compact_trajectory"),
            ("\n\nTRAJECTORY:\n", "trajectory"),
            ("\n\nFailed approach:\n", "failure_summary"),
        ):
            if marker in raw_content:
                header, body = raw_content.split(marker, 1)
                return header, body, body_type
        if raw_content.startswith("Task:") and "\n\n" in raw_content:
            header, body = raw_content.split("\n\n", 1)
            return header, body, "unknown"
        return "", raw_content, "raw"

    @staticmethod
    def _strip_trajectory_fence(value: str) -> str:
        value = value.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json|python|text)?\s*", "", value, count=1,
                           flags=re.IGNORECASE)
            value = re.sub(r"\s*```$", "", value, count=1)
        return value.strip()

    @staticmethod
    def _extract_balanced_list(value: str) -> Optional[str]:
        """Return the first balanced list literal, respecting quoted strings."""
        start = value.find("[")
        if start < 0:
            return None
        quote = None
        escaped = False
        depth = 0
        for idx in range(start, len(value)):
            char = value[idx]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in ("'", '\"'):
                quote = char
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return value[start : idx + 1]
        return None

    @staticmethod
    def _normalise_trajectory_payload(payload: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(payload, dict):
            for key in ("trajectory", "messages", "steps"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            return None
        messages = [item for item in payload if isinstance(item, dict)]
        if len(messages) != len(payload):
            raise ValueError("Retrieved memory trajectory contains non-message entries.")
        return messages

    def _parse_trajectory_list(self, trajectory_str: str) -> Optional[List[Dict[str, Any]]]:
        """Parse JSON/Python trajectory payloads, including fenced or prefixed values."""
        value = self._strip_trajectory_fence(trajectory_str)
        candidates = [value]
        balanced = self._extract_balanced_list(value)
        if balanced and balanced != value:
            candidates.append(balanced)
        parse_errors = []
        for candidate in candidates:
            for parser in (json.loads, ast.literal_eval):
                try:
                    result = self._normalise_trajectory_payload(parser(candidate))
                    if result is not None:
                        return result
                except Exception as exc:
                    parse_errors.append(exc)
        if value.startswith("[") or balanced is not None:
            raise ValueError("Could not parse trajectory list: %s" % (parse_errors[-1] if parse_errors else "invalid payload"))
        return None

    @staticmethod
    def _compact_text(value: str, limit: int = 420) -> str:
        value = re.sub(r"\s+", " ", (value or "").strip())
        return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"

    def _clean_trajectory_messages(self, trajectory_list: List[Dict[str, Any]]) -> str:
        """Render trajectories as a bounded action/observation summary, never raw prompts."""
        turn_idx = -1
        for i, msg in enumerate(trajectory_list):
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if msg.get("role") == "user" and isinstance(content, str) and "Now, it's your turn" in content:
                turn_idx = i
        if turn_idx >= 0:
            trajectory_list = trajectory_list[turn_idx:]

        pairs: List[Dict[str, str]] = []
        pending_observation = ""
        for message in trajectory_list:
            role = message.get("role")
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            if role == "user":
                if "Now, it's your turn" in content:
                    content = content.split("Now, it's your turn", 1)[-1]
                text = self._compact_text(content)
                if text:
                    pending_observation = text
            elif role == "assistant":
                action_match = re.search(r"(?im)^\s*Action\s*:\s*(.+)$", content)
                action = self._compact_text(action_match.group(1) if action_match else content, 240)
                if action:
                    pairs.append({"observation": pending_observation, "action": action})
                    pending_observation = ""

        if not pairs:
            return ""
        pairs = pairs[-24:]
        lines = ["Archived Action/Observation Summary:"]
        for index, pair in enumerate(pairs, 1):
            if pair["observation"]:
                lines.append(f"{index}. Observation: {pair['observation']}")
            lines.append(f"   Action: {pair['action']}")
        result = "\n".join(lines)
        return result if len(result) <= 5000 else result[-5000:]

    def _format_retrieved_memory(self, raw_content: str) -> str:
        """Format scripts or trajectories without leaking raw prompt/thought noise."""
        raw_content = (raw_content or "").strip()
        if not raw_content:
            return ""
        header, body, body_type = self._split_retrieved_memory_content(raw_content)
        header, body = header.strip(), body.strip()
        clean_parts = []
        if "SCRIPT:" in header:
            script_part = header.split("SCRIPT:", 1)[1].strip()
            if script_part:
                clean_parts.append(f"Archived Script:\n{script_part}")
        if "What went wrong:" in header:
            reflection_part = header.split("What went wrong:", 1)[1].strip()
            if reflection_part:
                clean_parts.append(f"Archived Script:\n{reflection_part}")
        if body_type == "compact_trajectory":
            clean_parts.append(body)
            return "\n\n".join(clean_parts) or raw_content

        # Parse only explicit trajectory payloads. Failure summaries can contain
        # bracketed prose such as "[open the cabinet first]"; scanning for the
        # first bracket anywhere falsely treated such prose as a Python list.
        stripped_body = body.lstrip()
        json_trajectory_envelope = bool(re.match(
            r'^\{\s*"(?:trajectory|messages|steps)"\s*:', stripped_body,
            flags=re.IGNORECASE,
        ))
        should_parse_trajectory = (
            body_type == "trajectory"
            or stripped_body.startswith("[")
            or stripped_body.startswith("```")
            or json_trajectory_envelope
        )
        if should_parse_trajectory:
            try:
                trajectory_list = self._parse_trajectory_list(body)
            except Exception as exc:
                logger.warning("Could not parse retrieved memory trajectory; using bounded fallback. Error: %s", exc)
                if body:
                    clean_parts.append(f"Archived Trajectory:\n{self._compact_text(body, 5000)}")
                return "\n\n".join(clean_parts) or raw_content
            if trajectory_list is not None:
                compact = self._clean_trajectory_messages(trajectory_list)
                if compact:
                    clean_parts.append(compact)
                return "\n\n".join(clean_parts) or raw_content
        if body and body != raw_content:
            clean_parts.append(f"{'Archived Trajectory' if body_type == 'trajectory' else 'Archived Script'}:\n{body}")
        return "\n\n".join(clean_parts) or raw_content

    def _construct_messages(self, task_description: str, retrieved_memories: List[Dict], task_type: str) -> List[Dict[str, str]]:
        """
        [REFACTORED]
        Builds the message list in a conversational ReAct style.
        """
        # 1. Start with the system prompt
        system_content = prompts.SYSTEM_PROMPT
        if getattr(self, '_no_think', False):
            system_content += "\n/no_think"
        elif getattr(self, '_force_think', False):
            system_content += "\n/think"

        if os.environ.get("MEMRL_ALFWORLD_STATE_GUARD_PROMPT", "0").strip().lower() not in {"0", "false", "no"}:
            system_content += """

Before choosing an action, maintain a compact symbolic state from the task and observations:
- required object(s), current inventory, known open/closed receptacles;
- whether clean/heat/cool is required and already complete;
- the remaining target placement.
Never repeat a completed transformation, never take an object already held, never put before holding, and open a closed target before putting. If the last action produced Nothing happened, choose a different valid action. Output exactly one executable command after Action:.
"""

        if os.environ.get("MEMRL_ALFWORLD_PROGRAM_GUIDE", "0").strip().lower() not in {"0", "false", "no"}:
            task_key = task_type.split('/', 1)[0]
            programs = {
                'pick_and_place_simple': 'PROGRAM: locate object -> take object -> locate target -> open target if closed -> put object.',
                'pick_and_place_with_movable_recep': 'PROGRAM: locate object -> take object -> put object in movable receptacle -> take/move receptacle -> put it at target.',
                'pick_clean_then_place_in_recep': 'PROGRAM: locate object -> take object -> go to sink -> clean object -> locate target -> open if closed -> put object.',
                'pick_cool_then_place_in_recep': 'PROGRAM: locate object -> take object -> go to fridge -> open if needed -> cool object -> locate target -> open if closed -> put object.',
                'pick_heat_then_place_in_recep': 'PROGRAM: locate object -> take object -> go to microwave -> open if needed -> place object inside -> close/use as required -> retrieve heated object -> locate target -> open if closed -> put object.',
                'pick_two_obj_and_place': 'PROGRAM: place the first required object at target -> locate a second instance -> place it at the same target; do not stop after one.',
                'look_at_obj_in_light': 'PROGRAM: locate target object -> take if needed -> locate light source -> activate if needed -> examine target under light.',
            }
            guide = programs.get(task_key)
            if guide:
                system_content += "\n\n" + guide + "\nBind names from the current task and observations; follow this execution skeleton without copying example objects."

        messages = [{"role": "system", "content": system_content}]

        # 2. Add the selected few-shot example as a complete dialogue
        example_dialogue = self._get_examples_for_task(task_type)
        if example_dialogue:
            # Modify the first user message in the example to introduce it
            example_dialogue[0]['content'] = "Here is an example of how to solve the task:\n" + example_dialogue[0]['content']
            messages.extend(example_dialogue)

        # 3. Add retrieved memories as additional context for the agent
        if retrieved_memories:
            successful_mems = retrieved_memories.get('successed', [])
            failed_mems = retrieved_memories.get('failed', [])

            # Region success summaries are pre-aggregated strategy text — render
            # them verbatim (not through the SCRIPT/trajectory parser).
            success_summary_mems = [
                m for m in successful_mems
                if isinstance(m, dict) and m.get('_region_success_summary')
            ]
            raw_success_mems = [
                m for m in successful_mems
                if not (isinstance(m, dict) and m.get('_region_success_summary'))
            ]

            successful_mems_formatted = [
                self._format_retrieved_memory(mem['content']) for mem in raw_success_mems
            ] if raw_success_mems else []

            success_summary_formatted = [
                mem['content'] for mem in success_summary_mems
            ] if success_summary_mems else []

            failed_mems_formatted = [
                self._format_retrieved_memory(mem['content']) for mem in failed_mems
            ] if failed_mems else []

            memory_parts = [
                "In addition to the example, you have the following memories from your own past experiences. "
                "Use them to help you if they are relevant:"
            ]

            if success_summary_formatted:
                memory_parts.append(
                    "--- EFFECTIVE STRATEGIES (aggregated from past successes) ---\n" +
                    "\n\n".join(success_summary_formatted)
                )

            if successful_mems_formatted:
                memory_parts.append(
                    "--- SUCCESSFUL MEMORIES (Examples to follow) ---\n" +
                    "\n".join(successful_mems_formatted)
                )

            if failed_mems_formatted:
                memory_parts.append(
                    "--- FAILED MEMORIES (Examples to avoid or learn from) ---\n" +
                    "\n".join(failed_mems_formatted)
                )

            if successful_mems_formatted or failed_mems_formatted or success_summary_formatted:
                memory_context = "\n\n".join(memory_parts)
                messages.append({"role": "user", "content": memory_context})

        # 4. Add the current task description as the new user prompt
        # The history of the current task will be appended in the `act` method
        current_task_prompt = f"Now, it's your turn to solve a new task.\n{task_description}"
        messages.append({"role": "user", "content": current_task_prompt})

        # Periodic prompt sampling for debugging (every 100th call)
        if not hasattr(self, '_construct_messages_counter'):
            self._construct_messages_counter = 0
        self._construct_messages_counter += 1
        if self._construct_messages_counter <= 3 or self._construct_messages_counter % 200 == 0:
            # Log memory injection details
            if retrieved_memories and isinstance(retrieved_memories, dict):
                s_mems = retrieved_memories.get('successed', [])
                f_mems = retrieved_memories.get('failed', [])
                logger.info(
                    "[MEMORY INJECT #%d] task=%s | success_mems=%d, failed_mems=%d",
                    self._construct_messages_counter,
                    task_description[:80],
                    len(s_mems), len(f_mems),
                )
                for idx, m in enumerate(s_mems[:2]):
                    content = m.get('content', '') or ''
                    logger.info(
                        "[MEMORY INJECT #%d] SUCCESS[%d] (%d chars): %s",
                        self._construct_messages_counter, idx, len(content), content[:300],
                    )
                for idx, m in enumerate(f_mems[:2]):
                    content = m.get('content', '') or ''
                    logger.info(
                        "[MEMORY INJECT #%d] FAILED[%d] (%d chars): %s",
                        self._construct_messages_counter, idx, len(content), content[:300],
                    )

        return messages

    def _parse_action(self, llm_response: str) -> str:
        """
        Extracts the 'Action:' part from the ReAct response.
        Handles DeepSeek-R1 <think>...</think> blocks and markdown formatting.
        """
        if not llm_response or not str(llm_response).strip():
            logger.warning(
                "LLM returned an empty/whitespace completion; marking for deferred repair."
            )
            return INVALID_LLM_ACTION

        import re
        text = re.sub(r'<think>.*?</think>', '', llm_response, flags=re.DOTALL).strip()
        if not text:
            # Entire response was inside <think>. Try to extract action from think content.
            text = re.sub(r'</?think>', '', llm_response).strip()
            if not text:
                return 'look'

        # Match Action: with optional markdown bold, bullets, numbering
        # Handles: "Action: go to", "**Action**: Go to", "- **Action**: go to", "1. Action: go to"
        action_match = re.search(
            r'(?i)^[\s\-\d.]*\*{0,2}action\*{0,2}[\s]*:[\s]*(.+)',
            text, re.MULTILINE
        )
        if action_match:
            action = action_match.group(1).strip().split("\n")[0].strip()
            action = action.rstrip('.')
            return action.lower() if action[0].isupper() else action

        # Match "Thought: ...\nAction: ..." pattern (ReAct format)
        thought_action = re.search(
            r'(?i)thought\s*:.*?\n\s*\*{0,2}action\*{0,2}\s*:\s*(.+)',
            text, re.DOTALL
        )
        if thought_action:
            action = thought_action.group(1).strip().split("\n")[0].strip()
            action = action.rstrip('.')
            return action.lower() if action[0].isupper() else action

        # Fallback: find any lowercase ALFWorld command
        alf_commands = ('go to ', 'take ', 'put ', 'open ', 'close ', 'use ',
                        'clean ', 'heat ', 'cool ', 'examine ', 'look', 'inventory')
        for line in text.split("\n"):
            line = line.strip()
            line = re.sub(r'^[\d.\-\*\s]+', '', line).strip()
            line_lower = line.lower()
            for cmd in alf_commands:
                if line_lower.startswith(cmd):
                    return line_lower.rstrip('.')

        logger.warning(
            "Could not extract action from LLM response (len=%d, repr=%r). "
            "Marking as invalid transport/model response for deferred repair.",
            len(llm_response), llm_response[:160],
        )
        return INVALID_LLM_ACTION
    def act(self, observation: str, history_messages: List[Dict[str, str]], first_step: bool = False):
        """
        Agent performs one step of action generation.
        Ensures robustness: if LLM fails or returns invalid output, action=None is returned.

        Thread-safety contract: This method does NOT mutate `history_messages`. Callers
        receive the messages to append via the second return value and must apply them
        themselves under any necessary locking. Previously this method appended in place,
        which raced with concurrent slot-level retries from ThreadPoolExecutor.

        Returns:
            (action: Optional[str], new_messages: List[Dict[str, str]])
                new_messages is the user-observation + assistant-response pair to append
                to the caller's history (or just the assistant response if first_step).
        """
        import json

        current_messages = copy.deepcopy(history_messages)
        if not first_step:
            current_messages.append({"role": "user", "content": f"Observation: {observation.strip()}"})

        # Truncate conversation to keep prompt within model context limit.
        # Keep all messages before the first Observation (static prefix: system, few-shot,
        # memories, task), then only the last K interaction turns.
        MAX_RECENT_TURNS = getattr(self, 'max_recent_turns', 20)
        # Find the CURRENT task marker first. Few-shot examples also contain
        # "Observation:" messages; using the first one globally truncates away the
        # real task prompt once an episode grows long, leaving the model to think the
        # example just finished ("OK, ready for the next task").
        task_marker = "Now, it's your turn to solve a new task."
        current_task_idx = None
        for idx in range(len(current_messages) - 1, -1, -1):
            m = current_messages[idx]
            content = m.get("content", "")
            if m.get("role") == "user" and isinstance(content, str) and content.strip().startswith(task_marker):
                current_task_idx = idx
                break
        interaction_start = len(current_messages)
        search_from = (current_task_idx + 1) if current_task_idx is not None else 0
        for idx in range(search_from, len(current_messages)):
            m = current_messages[idx]
            content = m.get("content", "")
            if m.get("role") == "user" and isinstance(content, str) and content.strip().startswith("Observation:"):
                interaction_start = idx
                break
        n_interaction = len(current_messages) - interaction_start
        if n_interaction > MAX_RECENT_TURNS:
            # Preserve the entire static prefix through the actual current-task prompt;
            # only truncate live observation/action turns.
            head_end = (current_task_idx + 1) if current_task_idx is not None else interaction_start
            head = current_messages[:head_end]
            tail = current_messages[-MAX_RECENT_TURNS:]
            current_messages = head + tail
        filtered_messages = []
        for i, m in enumerate(current_messages):
            if m.get("content") is None:
                logger.warning(f"[Message Filter] Message {i} has None content, removed: {m}")
                continue
            if isinstance(m.get("content"), str) and not m["content"].strip():
                logger.warning(f"[Message Filter] Message {i} has empty content, removed: {m}")
                continue
            filtered_messages.append(m)
        current_messages = filtered_messages

        logger.debug("Querying LLM for the next action...")

        response = None
        try:
            response = self.llm.generate(current_messages)
        except Exception as e:
            logger.error("LLM generation failed: %s", str(e))
            logger.error("Messages before failure:\n%s", json.dumps(current_messages, indent=2, ensure_ascii=False))
            response = None  # fallback

        # Build the messages to be appended (returned to caller; not applied here).
        # Strip <think>...</think> blocks from response before storing in history
        # to prevent context length explosion from accumulated reasoning tokens.
        import re
        raw_len = len(response) if response else 0
        has_think_tags = bool(response and '<think>' in response)
        clean_response = response if response is None else re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        think_stripped = raw_len - len(clean_response) if clean_response else 0

        is_fallback = False
        if clean_response is not None and not clean_response:
            clean_response = response  # fallback if entire response was thinking
            is_fallback = True
            logger.warning(
                "[ACT] Response was entirely thinking (raw_len=%d, has_tags=%s). "
                "Using raw response as fallback.",
                raw_len, has_think_tags,
            )

        # Cap only fallback responses (thinking leaked into content) to avoid context explosion.
        # Normal responses (short action text from reasoning-parser) are not capped.
        max_hist_chars = getattr(self, '_max_history_response_chars', 0)
        capped = False
        if is_fallback and max_hist_chars > 0 and clean_response and len(clean_response) > max_hist_chars:
            clean_response = clean_response[-max_hist_chars:]
            capped = True

        # Periodic diagnostic logging (first 5 calls + every 200th)
        if not hasattr(self, '_act_call_count'):
            self._act_call_count = 0
        self._act_call_count += 1
        if self._act_call_count <= 5 or self._act_call_count % 200 == 0:
            logger.info(
                "[ACT DIAG #%d] raw_len=%d has_think_tags=%s think_stripped=%d "
                "clean_len=%d capped=%s history_msgs=%d",
                self._act_call_count, raw_len, has_think_tags, think_stripped,
                len(clean_response) if clean_response else 0, capped,
                len(current_messages),
            )

        action = None
        if response:
            try:
                action = self._parse_action(response)
            except Exception as e:
                logger.warning(f"Action parsing failed for response='{response}': {e}")
                action = "inventory"
        elif response is not None:
            action = INVALID_LLM_ACTION

        new_messages: List[Dict[str, str]] = []
        if action != INVALID_LLM_ACTION:
            if not first_step:
                new_messages.append({"role": "user", "content": f"Observation: {observation.strip()}"})
            new_messages.append({"role": "assistant", "content": clean_response if clean_response is not None else "No response."})
        else:
            logger.warning("[ACT] Invalid completion excluded from episode history to prevent self-reinforcing prompt drift.")

        return action, new_messages



    def get_trajectory(self) -> List[Dict[str, str]]:
        """Returns the complete trajectory for the finished episode."""
        pass
