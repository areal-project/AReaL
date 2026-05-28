import asyncio
import atexit
import json
import os
import re
import textwrap
import threading
import time
import uuid
from typing import (
    TYPE_CHECKING,
)

from camel.agents import ChatAgent
from camel.agents._types import ToolCallRequest
from camel.logger import get_logger
from camel.messages import (
    BaseMessage,
    FunctionCallingMessage,
)
from camel.prompts import TextPrompt
from camel.responses import ChatAgentResponse
from camel.types import (
    OpenAIBackendRole,
)
from camel.types.agents import ToolCallingRecord
from pydantic import BaseModel

from areal.utils.perf_tracer import (
    Category,
    atrace_scope,
    atrace_session_phase,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Cleanup temp files on exit
_temp_files: set[str] = set()
_temp_files_lock = threading.Lock()


def _cleanup_temp_files():
    with _temp_files_lock:
        for path in _temp_files:
            try:
                os.unlink(path)
            except Exception:
                pass


atexit.register(_cleanup_temp_files)

SIMPLE_FORMAT_PROMPT = TextPrompt(
    textwrap.dedent(
        """\
        Please format the following content:

        {content}
        """
    )
)


class ChatAgentTrace(ChatAgent):
    """A ChatAgent with performance tracing capabilities."""

    def __init__(self, *args, **kwargs):
        """Initialize ChatAgentTrace with parse error tracking."""
        super().__init__(*args, **kwargs)
        self.max_parse_errors = kwargs.get("max_parse_errors", 3)
        self.parse_error_count = 0

    async def adetect_tool_calls_parse_error(self, response):
        r"""
        Asynchronously detect tool calls in the response content using Qwen25Detector.
        if the model is Qwen 2.5 or Qwen 3.
        if there's tool call tokens detected, but got json parse failure, format the information into a tool call record,
        so that the agent can handle the error next step.
        add a self.count_parse_error, so that we can limit the number of parse errors we handle in one step. if max reached, just
        break the loop.

        Args:
            response: The model response to check for parse errors

        Returns:
            Optional[ToolCallingRecord]: A tool calling record with error information if parse error detected, None otherwise
        """
        bot_token = "<tool_call>\n"
        eot_token = "\n</tool_call>"

        # Check if we've reached max parse errors
        if self.parse_error_count >= self.max_parse_errors:
            logger.warning(
                f"Max parse errors ({self.max_parse_errors}) reached, stopping error handling"
            )
            return None

        # Extract content from response
        if not response.output_messages:
            return None

        content = response.output_messages[0].content
        if not content or bot_token not in content:
            return None

        # Find all potential tool call blocks
        pattern = rf"{re.escape(bot_token)}(.*?){re.escape(eot_token)}"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return None

        # Check each match for JSON parse errors
        for match_text in matches:
            try:
                # Try to parse the JSON
                json.loads(match_text.strip())
                # If successful, no error for this match
                continue
            except json.JSONDecodeError as e:
                # Found a parse error
                self.parse_error_count += 1
                logger.warning(
                    f"Detected JSON parse error (count: {self.parse_error_count}/{self.max_parse_errors}): {str(e)}"
                )
                logger.warning(f"Problematic content: {match_text[:200]}...")

                # Create an error tool calling record
                error_message = (
                    f"JSON Parse Error: {str(e)}\n"
                    f"The tool call format is incorrect. Please ensure:\n"
                    f"1. The JSON is valid and properly formatted\n"
                    f"2. All quotes are properly escaped\n"
                    f"3. The structure matches: {{'name': 'function_name', 'arguments': {{}}}}\n"
                    f"Problematic content (first 200 chars): {match_text[:200]}..."
                )

                # Generate a unique error tool call ID
                error_tool_call_id = f"error_{uuid.uuid4().hex[:8]}"

                # Create the error record
                error_record = ToolCallingRecord(
                    tool_name="json_parse_error",
                    args={"raw_content": match_text, "error": str(e)},
                    result=error_message,
                    tool_call_id=error_tool_call_id,
                )

                # Record this in memory so the model can see the error
                assist_msg = FunctionCallingMessage(
                    role_name=self.role_name,
                    role_type=self.role_type,
                    meta_dict=None,
                    content="",
                    func_name="json_parse_error",
                    args={"raw_content": match_text[:200], "error": str(e)},
                    tool_call_id=error_tool_call_id,
                )

                func_msg = FunctionCallingMessage(
                    role_name=self.role_name,
                    role_type=self.role_type,
                    meta_dict=None,
                    content="",
                    func_name="json_parse_error",
                    result=error_message,
                    tool_call_id=error_tool_call_id,
                )

                # Use precise timestamps
                current_time_ns = time.time_ns()
                base_timestamp = current_time_ns / 1_000_000_000

                self.update_memory(
                    assist_msg, OpenAIBackendRole.ASSISTANT, timestamp=base_timestamp
                )
                self.update_memory(
                    func_msg,
                    OpenAIBackendRole.FUNCTION,
                    timestamp=base_timestamp + 1e-6,
                )

                return error_record

        return None

    async def _astep_non_streaming_task(
        self,
        input_message: BaseMessage | str,
        response_format: type[BaseModel] | None = None,
    ) -> ChatAgentResponse:
        r"""Internal async method for non-streaming astep logic."""

        # try to extract task name if exists in input_message
        if isinstance(input_message, str):
            task_name_match = re.search(r"Task name:(.*)\n", input_message)
            if task_name_match:
                task_name = task_name_match.group(1).strip()
            else:
                task_name = "default"
        else:
            task_name = "default"

        # Reset parse error counter at the start of each step
        self.parse_error_count = 0

        try:
            from camel.utils.langfuse import set_current_agent_session_id

            set_current_agent_session_id(self.agent_id)
        except ImportError:
            pass  # Langfuse not available

        # Check if this call is from a RegisteredAgentToolkit to prevent tool
        # use
        disable_tools = self._is_called_from_registered_toolkit()

        # Handle response format compatibility with non-strict tools
        original_response_format = response_format
        input_message, response_format, used_prompt_formatting = (
            self._handle_response_format_with_non_strict_tools(
                input_message, response_format
            )
        )

        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: list[ToolCallingRecord] = []
        external_tool_call_requests: list[ToolCallRequest] | None = None
        accumulated_context_tokens = (
            0  # This tracks cumulative context tokens, not API usage tokens
        )

        # Initialize token usage tracker
        step_token_usage = self._create_token_usage_tracker()
        iteration_count: int = 0
        prev_num_openai_messages: int = 0
        while True:
            if self.pause_event is not None and not self.pause_event.is_set():
                if isinstance(self.pause_event, asyncio.Event):
                    await self.pause_event.wait()
                elif isinstance(self.pause_event, threading.Event):
                    # For threading.Event in async context, run in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.pause_event.wait)
            try:
                openai_messages, num_tokens = self.memory.get_context()
                accumulated_context_tokens += num_tokens
            except RuntimeError as e:
                return self._step_terminate(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )

            async with atrace_scope(
                f"agent_astep._aget_model_response:{task_name}",
                category=Category.COMM,
                args={"agent_id": self.agent_id, "iteration": iteration_count},
            ):
                async with atrace_session_phase("generate"):
                    response = await self._aget_model_response(
                        openai_messages,
                        # num_tokens=num_tokens,
                        current_iteration=iteration_count,
                        response_format=response_format,
                        tool_schemas=[]
                        if disable_tools
                        else self._get_full_tool_schemas(),
                        prev_num_openai_messages=prev_num_openai_messages,
                    )

            prev_num_openai_messages = len(openai_messages)
            iteration_count += 1

            # Accumulate API token usage
            self._update_token_usage_tracker(step_token_usage, response.usage_dict)

            # Terminate Agent if stop_event is set
            if self.stop_event and self.stop_event.is_set():
                # Use the _step_terminate to terminate the agent with reason
                logger.info(f"Termination triggered at iteration {iteration_count}")
                return self._step_terminate(
                    accumulated_context_tokens,
                    tool_call_records,
                    "termination_triggered",
                )

            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if tool_call_request.tool_name in self._external_tool_schemas:
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        if (
                            self.pause_event is not None
                            and not self.pause_event.is_set()
                        ):
                            if isinstance(self.pause_event, asyncio.Event):
                                await self.pause_event.wait()
                            elif isinstance(self.pause_event, threading.Event):
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, self.pause_event.wait)
                        async with atrace_scope(
                            f"agent_astep._aexecute_tool:{task_name}",
                            category=Category.IO,
                            args={
                                "agent_id": self.agent_id,
                                "iteration": iteration_count,
                                "tool_name": tool_call_request.tool_name,
                            },
                        ):
                            async with atrace_session_phase("toolcall"):
                                tool_call_record = await self._aexecute_tool(
                                    tool_call_request
                                )
                        tool_call_records.append(tool_call_record)

                # If we found an external tool call, break the loop
                if external_tool_call_requests:
                    break

                if (
                    self.max_iteration is not None
                    and iteration_count >= self.max_iteration
                ):
                    break

                # If we're still here, continue the loop
                continue

            # Check for JSON parse errors in tool calls (Qwen 2.5/3 specific)
            parse_error_record = await self.adetect_tool_calls_parse_error(response)
            if parse_error_record:
                print(
                    f"Task {task_name}: Detected tool call parse error, prompting model to correct."
                )
                tool_call_records.append(parse_error_record)

                # Check if we've reached max parse errors
                if self.parse_error_count >= self.max_parse_errors:
                    logger.error(
                        f"Max parse errors reached ({self.max_parse_errors}), "
                        "terminating step to prevent infinite loop"
                    )
                    break

                # Continue to let the model try again with the error feedback
                continue

            break

        await self._aformat_response_if_needed(response, response_format)

        # Apply manual parsing if we used prompt-based formatting
        if used_prompt_formatting and original_response_format:
            self._apply_prompt_based_parsing(response, original_response_format)

        self._record_final_output(response.output_messages)

        # Clean tool call messages from memory after response generation
        if self.prune_tool_calls_from_memory and tool_call_records:
            self.memory.clean_tool_calls()

        return self._convert_to_chatagent_response(
            response,
            tool_call_records,
            accumulated_context_tokens,
            external_tool_call_requests,
            step_token_usage["prompt_tokens"],
            step_token_usage["completion_tokens"],
            step_token_usage["total_tokens"],
        )
