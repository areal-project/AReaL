"""Raw vLLM HTTP client for /v1/completions endpoint.

Sends manually-constructed ChatML prompts to a vLLM server and handles
response parsing, including <think>...</think> block extraction.
"""

import asyncio
import re
import time
from typing import Dict, Optional, Tuple

import aiohttp


async def post_completion(
    session: aiohttp.ClientSession,
    api_base: str,
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    timeout: int = 600,
) -> Dict:
    """POST a raw prompt to /v1/completions and return the JSON response.

    Args:
        session: aiohttp client session
        api_base: Base URL (e.g., "http://127.0.0.1:8000/v1")
        model_name: Served model name
        prompt: Full ChatML-formatted prompt string
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response dict
    """
    url = f"{api_base}/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(
                f"vLLM request failed (status {resp.status}): {body[:500]}"
            )
        return await resp.json()


def extract_response_text(
    response_json: Dict,
) -> Tuple[str, Optional[str], Dict]:
    """Extract content, reasoning, and usage from a completions response.

    Handles Qwen3-style <think>...</think> blocks by splitting them out.

    Returns:
        (content, reasoning_content, usage_dict)
    """
    choices = response_json.get("choices", [])
    if not choices:
        return ("", None, {})

    raw_text = choices[0].get("text", "")
    usage = response_json.get("usage", {})

    # Handle <think>...</think> reasoning block
    reasoning_content = None
    content = raw_text

    if "</think>" in raw_text:
        think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip()
            # Content is everything after the last </think>
            content = raw_text.rsplit("</think>", 1)[-1].strip()

    return (content, reasoning_content, usage)


async def wait_for_server(api_base: str, timeout: int = 300) -> bool:
    """Poll the /models endpoint until the server is ready.

    Args:
        api_base: Base URL (e.g., "http://127.0.0.1:8000/v1")
        timeout: Maximum seconds to wait

    Returns:
        True if server became ready, False if timeout
    """
    url = f"{api_base}/models"
    start = time.time()

    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(2)

    return False


