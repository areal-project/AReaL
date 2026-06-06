"""LiteLLM service implementation for unified LLM access."""

import asyncio
import os
import logging
import random
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import time

import litellm
from litellm import acompletion, token_counter
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMResponse, LLMService, Message

logger = logging.getLogger(__name__)

# Configure success call logger for dedicated log file
# Use module-level timestamp to ensure each program instance gets its own log file
_setup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_success_logger_configured = False
_success_logger_lock = threading.Lock()


def _setup_success_logger():
    """Setup a dedicated logger for successful LLM calls.
    
    Each program instance gets its own log file based on the setup timestamp,
    ensuring different programs don't write to the same log file.
    """
    global _success_logger_configured, _setup_timestamp
    
    with _success_logger_lock:
        if _success_logger_configured:
            return
        
        success_logger = logging.getLogger(f"{__name__}.success_calls")
        success_logger.setLevel(logging.INFO)
        
        # Avoid adding multiple handlers if already configured
        if success_logger.handlers:
            _success_logger_configured = True
            return
        
        # Create log directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create file handler for success calls with timestamp
        # Each program instance gets its own log file based on setup timestamp
        log_file = log_dir / f"llm_success_calls_{_setup_timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        success_logger.addHandler(file_handler)
        success_logger.propagate = False  # Don't propagate to root logger
        
        # Print to console so user knows where logs are being written
        print(f"Success call logger initialized, logging to: {log_file}", flush=True)
        _success_logger_configured = True

# Default timeout for LLM requests (90 minutes for long content generation)
DEFAULT_TIMEOUT_SECONDS = 60 * 90



class LiteLLMService(LLMService):
    """LiteLLM service for unified access to multiple LLM providers."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize LiteLLM service.

        Args:
            model_name: Name of the model (e.g., "gpt-4", "claude-3-opus", "gemini/gemini-3-pro")
            api_key: API key for authentication
            api_base: Base URL for API (for custom endpoints).
                     Can be a single URL or multiple URLs separated by commas (e.g., "url1,url2,url3").
                     When multiple URLs are provided, they will be randomly selected for load balancing.
            **kwargs: Additional configuration
        """
        super().__init__(model_name, api_key, **kwargs)

        # litellm._turn_on_debug()

        # Setup success call logger
        _setup_success_logger()

        # Parse api_base: support multiple comma-separated URLs
        if api_base:
            # Split by comma and strip whitespace
            self.api_bases = [base.strip() for base in api_base.split(',') if base.strip()]
        else:
            self.api_bases = []

        # Request budgeting context limit. Override with LLM_CONTEXT_LIMIT_TOKENS when needed.
        configured_context_limit = kwargs.get("context_limit")
        if isinstance(configured_context_limit, int) and configured_context_limit > 0:
            self.context_limit = configured_context_limit
        else:
            context_limit_env = os.getenv("LLM_CONTEXT_LIMIT_TOKENS")
            if context_limit_env and context_limit_env.isdigit() and int(context_limit_env) > 0:
                self.context_limit = int(context_limit_env)
            else:
                self.context_limit = 131072

        # Optional default completion cap from service configuration.
        configured_max_tokens = kwargs.get("max_tokens")
        if isinstance(configured_max_tokens, int) and configured_max_tokens > 0:
            self.default_max_tokens: Optional[int] = configured_max_tokens
        else:
            self.default_max_tokens = None

    def _get_api_base(self) -> Optional[str]:
        """Get an API base URL.

        If multiple API bases are configured, randomly select one for load balancing.

        Returns:
            Selected API base URL or None if no API bases configured
        """
        if not self.api_bases:
            return None
        if len(self.api_bases) == 1:
            return self.api_bases[0]
        # Random selection for load balancing across multiple endpoints
        return random.choice(self.api_bases)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def generate_(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using LiteLLM."""
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        call_kwargs = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": temperature,
            "timeout": 60*90,
            **kwargs,
        }

        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        else:
            # Calculate max_tokens as context_limit - input_tokens to avoid exceeding max model length
            # Leave a safety margin to account for token_counter inaccuracies and API formatting overhead
            CONTEXT_LIMIT = self.context_limit
            SAFETY_MARGIN = 1000  # Safety margin for token_counter inaccuracies and API overhead
            try:
                input_tokens = token_counter(model=self.model_name, messages=formatted_messages)
                
                # Check if input exceeds limit (with safety margin)
                if input_tokens > (CONTEXT_LIMIT - SAFETY_MARGIN):
                    # print("input_tokens: ", input_tokens)
                    # print("input", formatted_messages)
                    raise ValueError(f"Input tokens ({input_tokens}) exceed safe limit ({CONTEXT_LIMIT - SAFETY_MARGIN})")
                
                # Calculate max_tokens with safety margin to account for:
                # 1. token_counter inaccuracies (may underestimate actual tokens)
                # 2. API formatting overhead (extra tokens added during API call)
                # 3. extra_body parameters (e.g., reasoning_effort) that may affect token count
                available_tokens = CONTEXT_LIMIT - input_tokens - SAFETY_MARGIN
                call_kwargs["max_tokens"] = max(1, available_tokens)  # Ensure at least 1 token
            except Exception:
                # If token counting fails, don't set max_tokens and let API use default
                pass

        if self.api_key:
            call_kwargs["api_key"] = self.api_key

        # Get API base (randomly selected if multiple are configured)
        api_base = self._get_api_base()
        if api_base:
            call_kwargs["api_base"] = api_base

        # Measure LLM call time
        llm_start_time = time.time()
        response = await acompletion(**call_kwargs)
        llm_end_time = time.time()
        llm_duration = llm_end_time - llm_start_time

        content = response.choices[0].message.content or ""
        reasoning_content = None

        if "qwen3" in self.model_name.lower() and "</think>" in content:
            reasoning_content = content.split("</think>")[0] + "</think>"
            content = content[content.rfind("</think>") + len("</think>"):]
        
        if "gpt-oss" in self.model_name.lower() and "<|start|>assistant<|channel|>final<|message|>" in content:
            reasoning_content = content.split("<|start|>assistant<|channel|>final<|message|>")[0] + "<|start|>assistant<|channel|>final<|message|>"
            content = content.split("<|start|>assistant<|channel|>final<|message|>")[1]

        # Build usage dict with timing
        usage_dict = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "llm_call_time_sec": llm_duration,  # Add LLM call time
            "model": self.model_name,
            "api_base": api_base if api_base else "default",
        
        }

        # Log successful call to dedicated log file
        success_logger = logging.getLogger(f"{__name__}.success_calls")
        success_logger.info(f"Success call - {usage_dict}")


        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model=response.model,
            usage=usage_dict,
            raw_response=response,
            finish_reason=response.choices[0].finish_reason,
        )

    def trim_input(self, messages, max_tokens: Optional[int] = None,):
        # 1. 基础配置
        CONTEXT_LIMIT = self.context_limit
        SAFETY_MARGIN = 1000  # 稍微留宽一点，防止 token 计数器误差
        
        # 如果用户请求了特定的 max_tokens，我们要为输出预留出这些空间
        reserved_output_tokens = max_tokens if max_tokens is not None else 20480
        max_input_allowance = CONTEXT_LIMIT - reserved_output_tokens - SAFETY_MARGIN

        # 2. 实现滑动窗口截断逻辑
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        try:
            current_tokens = token_counter(model=self.model_name, messages=formatted_messages)
            
            if current_tokens > max_input_allowance:
                # 找到系统提示词 (通常是第一个)
                system_msg = [m for m in formatted_messages if m["role"] == "system"]
                other_msgs = [m for m in formatted_messages if m["role"] != "system"]
                
                # 从后往前保留消息，直到达到限制
                trimmed_msgs = []
                accumulated_tokens = token_counter(model=self.model_name, messages=system_msg)
                
                for msg in reversed(other_msgs):
                    msg_tokens = token_counter(model=self.model_name, messages=[msg])
                    if accumulated_tokens + msg_tokens < max_input_allowance:
                        trimmed_msgs.insert(0, msg)
                        accumulated_tokens += msg_tokens
                    else:
                        break
                
                formatted_messages = system_msg + trimmed_msgs
                logging.warning(f"检测到输入超限 ({current_tokens})，已滑动窗口截断至 {accumulated_tokens}")
        except Exception as e:
            logging.error(f"Token counting/trimming failed: {e}")
        return formatted_messages

    def prepare_request(self, messages, requested_max_tokens: Optional[int] = None):
        CONTEXT_LIMIT = self.context_limit
        SAFETY_MARGIN = 500 
        
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        try:
            input_tokens = token_counter(model=self.model_name, messages=formatted_messages)
        except Exception as e:
            logging.error(f"Token counting failed: {e}")
            input_tokens = 0 

        MIN_OUTPUT_RESERVE = 8192
        max_input_allowance = CONTEXT_LIMIT - (requested_max_tokens or MIN_OUTPUT_RESERVE) - SAFETY_MARGIN

        if input_tokens > max_input_allowance:
            system_msgs = [m for m in formatted_messages if m["role"] == "system"]
            other_msgs = [m for m in formatted_messages if m["role"] != "system"]
            
            system_tokens = token_counter(model=self.model_name, messages=system_msgs)

            if system_tokens >= max_input_allowance:
                logging.error("System prompt is too large for the context limit.")
                formatted_messages = system_msgs
                input_tokens = system_tokens
            else:
                trimmed_history = []
                current_accumulated = system_tokens
                
                for msg in reversed(other_msgs):
                    msg_tokens = token_counter(model=self.model_name, messages=[msg])
                    
                    if current_accumulated + msg_tokens <= max_input_allowance:
                        trimmed_history.insert(0, msg) 
                        current_accumulated += msg_tokens
                    else:
                        break
                
                formatted_messages = system_msgs + trimmed_history
                input_tokens = current_accumulated
                
                logging.warning(
                    f"Input exceeds limit. Trimmed from {input_tokens} to {current_accumulated}. "
                    f"Kept {len(trimmed_history)} historical messages."
                )

        available_for_output = CONTEXT_LIMIT - input_tokens - SAFETY_MARGIN
        
        if requested_max_tokens is None:
            # No explicit cap requested, use full available output budget.
            final_max_tokens = available_for_output
        else:
            if requested_max_tokens > available_for_output:
                logging.warning(f"Requested max_tokens {requested_max_tokens} exceeds limit. Trimming to {available_for_output}")
                final_max_tokens = available_for_output
            else:
                final_max_tokens = requested_max_tokens

        final_max_tokens = max(final_max_tokens, 1)

        return formatted_messages, final_max_tokens
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using LiteLLM."""
        # formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        # formatted_messages = self.trim_input(messages, max_tokens)
        requested_max_tokens = (
            max_tokens if max_tokens is not None else self.default_max_tokens
        )
        formatted_messages, max_tokens = self.prepare_request(
            messages, requested_max_tokens
        )

        call_kwargs = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": temperature,
            "timeout": 60*90,
            **kwargs,
        }

        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        else:
            # Calculate max_tokens as context_limit - input_tokens to avoid exceeding max model length
            # Leave a safety margin to account for token_counter inaccuracies and API formatting overhead
            CONTEXT_LIMIT = self.context_limit
            SAFETY_MARGIN = 500  # Safety margin for token_counter inaccuracies and API overhead
            try:
                input_tokens = token_counter(model=self.model_name, messages=formatted_messages)
                
                # Check if input exceeds limit (with safety margin)
                if input_tokens > (CONTEXT_LIMIT - SAFETY_MARGIN):
                    # print("input_tokens: ", input_tokens)
                    # print("input", formatted_messages)
                    raise ValueError(f"Input tokens ({input_tokens}) exceed safe limit ({CONTEXT_LIMIT - SAFETY_MARGIN})")
                
                # Calculate max_tokens with safety margin to account for:
                # 1. token_counter inaccuracies (may underestimate actual tokens)
                # 2. API formatting overhead (extra tokens added during API call)
                # 3. extra_body parameters (e.g., reasoning_effort) that may affect token count
                available_tokens = CONTEXT_LIMIT - input_tokens - SAFETY_MARGIN
                call_kwargs["max_tokens"] = max(1, available_tokens)  # Ensure at least 1 token
            except Exception:
                # If token counting fails, don't set max_tokens and let API use default
                pass

        if self.api_key:
            call_kwargs["api_key"] = self.api_key

        # Get API base (randomly selected if multiple are configured)
        api_base = self._get_api_base()
        if api_base:
            call_kwargs["api_base"] = api_base

        # Measure LLM call time
        llm_start_time = time.time()
        response = await acompletion(**call_kwargs)
        llm_end_time = time.time()
        llm_duration = llm_end_time - llm_start_time

        content = response.choices[0].message.content or ""
        reasoning_content = None

        if "qwen3" in self.model_name.lower() and "</think>" in content:
            reasoning_content = content.split("</think>")[0] + "</think>"
            content = content[content.rfind("</think>") + len("</think>"):]
        
        if "gpt-oss" in self.model_name.lower() and "<|start|>assistant<|channel|>final<|message|>" in content:
            reasoning_content = content.split("<|start|>assistant<|channel|>final<|message|>")[0] + "<|start|>assistant<|channel|>final<|message|>"
            content = content.split("<|start|>assistant<|channel|>final<|message|>")[1]

        # Build usage dict with timing
        usage_dict = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "llm_call_time_sec": llm_duration,  # Add LLM call time
            "model": self.model_name,
            "api_base": api_base if api_base else "default",
        
        }

        # Log successful call to dedicated log file
        success_logger = logging.getLogger(f"{__name__}.success_calls")
        success_logger.info(f"Success call - {usage_dict}")


        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model=response.model,
            usage=usage_dict,
            raw_response=response,
            finish_reason=response.choices[0].finish_reason,
        )

    async def generate_batch(
        self,
        messages_batch: List[List[Message]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[LLMResponse]:
        """Generate responses for a batch of message sequences."""
        tasks = [
            self.generate(messages, temperature, max_tokens, **kwargs)
            for messages in messages_batch
        ]
        return await asyncio.gather(*tasks)