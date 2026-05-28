import asyncio
import copy
import json
import os
import time

import aiohttp
import tiktoken
from openai import AsyncOpenAI
from prompt import EXTRACTOR_PROMPT
from qwen_agent.tools.base import BaseTool, register_tool

from areal.utils import logging

logger = logging.getLogger("VisitTool")

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


@register_tool("visit", allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = "visit"
    description = "Visit webpage(s) and return the summary of the content."
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s).",
            },
        },
        "required": ["url", "goal"],
    }

    # The `call` method is the main function of the tool.
    def __init__(self, cfg: dict | None = None, summary_client=None):
        super().__init__(cfg)
        self._llm_client = summary_client

    async def call(self, params: str | dict, **kwargs) -> str:  # type: ignore[override]
        try:
            url = params["url"]
            goal = params["goal"]
        except Exception:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()

        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        if isinstance(url, str):
            logger.debug(f"Visiting single URL: {url}")
            response = await self.readpage_jina(url, goal)
        else:
            response = []
            logger.debug(f"Visiting multiple URLs: {url}")
            assert isinstance(url, list)
            start_time = time.time()
            for u in url:
                if time.time() - start_time > 900:
                    cur_response = f"The useful information in {u} for user goal {goal} as follows: \n\n"
                    cur_response += (
                        "Evidence in page: \n"
                        + "The provided webpage content could not be accessed. Please check the URL or file format."
                        + "\n\n"
                    )
                    cur_response += (
                        "Summary: \n"
                        + "The webpage content could not be processed, and therefore, no information is available."
                        + "\n\n"
                    )
                else:
                    try:
                        cur_response = await self.readpage_jina(u, goal)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {e}"
                response.append(cur_response)
            response = "\n=======\n".join(response)

        logger.debug(f"Summary Length {len(response)}; Summary Content {response}")
        return response.strip()

    async def call_server(self, msgs, max_retries=2):
        if not self._llm_client:
            api_key = os.environ.get("API_KEY")
            url_llm = os.environ.get("API_BASE")
            model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=url_llm,
            )
            summary_args = dict(
                model=model_name,
                messages=msgs,
                temperature=0.7,
            )
        else:
            client = self._llm_client
            summary_args = dict(
                messages=msgs,
                temperature=0.7,
            )
        for attempt in range(max_retries):
            try:
                chat_response = await client.chat.completions.create(**summary_args)
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except Exception:
                        left = content.find("{")
                        right = content.rfind("}")
                        if left != -1 and right != -1 and left <= right:
                            content = content[left : right + 1]
                    return content
            except Exception:
                if attempt == (max_retries - 1):
                    return ""
                await asyncio.sleep(0.5)
                continue

    async def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.

        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page

        Returns:
            str: The webpage content or error message
        """
        max_retries = 3
        timeout = 50
        logger.debug(
            f"entering jina_readpage url: {url}, jina_api_keys={JINA_API_KEYS}"
        )
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                headers = {"Authorization": f"Bearer {JINA_API_KEYS}"}
                try:
                    async with session.get(  # type: ignore[union-attr]
                        f"https://r.jina.ai/{url}", headers=headers, timeout=timeout
                    ) as response:
                        if response.status == 200:
                            webpage_content = await response.text()
                            return webpage_content
                        else:
                            await response.text()
                            raise ValueError("jina readpage error")
                except Exception as e:
                    logger.debug(f"jina_readpage {url} failed with error {e}")
                    if attempt == max_retries - 1:
                        logger.debug(
                            f"jina_readpage {url} failed after {max_retries} attempts"
                        )
                        return "[visit] Failed to read page."
                    await asyncio.sleep(0.5)

        return "[visit] Failed to read page."

    async def html_readpage_jina(self, url: str) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            logger.debug(
                f"html_readpage_jina {url} attempt {attempt + 1}/{max_attempts}"
            )
            content = await self.jina_readpage(url)
            if (
                content
                and not content.startswith("[visit] Failed to read page.")
                and content != "[visit] Empty content."
                and not content.startswith("[document_parser]")
            ):
                return content
        return "[visit] Failed to read page."

    async def readpage_jina(self, url: str, goal: str) -> str:
        """Read and summarize a webpage using Jina + LLM extractor."""
        summary_page_func = self.call_server
        max_retries = int(os.getenv("VISIT_SERVER_MAX_RETRIES", 1))
        _content = await self.html_readpage_jina(url)
        content = copy.deepcopy(_content)
        valid_content = (
            content
            and not content.startswith("[visit] Failed to read page.")
            and content != "[visit] Empty content."
            and not content.startswith("[document_parser]")
        )
        logger.debug(
            f"readpage_jina content: {len(content)}, valid_content: {valid_content}"
        )
        if not valid_content:
            useful_information = (
                f"The useful information in {url} for user goal {goal} as follows: \n\n"
            )
            useful_information += (
                "Evidence in page: \n"
                + "The provided webpage content could not be accessed. Please check the URL or file format."
                + "\n\n"
            )
            useful_information += (
                "Summary: \n"
                + "The webpage content could not be processed, and therefore, no information is available."
                + "\n\n"
            )
            return useful_information

        content = truncate_to_tokens(content, max_tokens=95000)
        messages = [
            {
                "role": "user",
                "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal),
            }
        ]
        raw = await summary_page_func(messages, max_retries=max_retries)
        summary_retries = 3
        while isinstance(raw, str) and len(raw) < 10 and summary_retries >= 0:
            truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
            status_msg = (
                f"[visit] Summary url[{url}] attempt {3 - summary_retries + 1}/3, content length: {len(content)}, truncating to {truncate_length} chars"
                if summary_retries > 0
                else f"[visit] Summary url[{url}] failed after 3 attempts, final truncation to 25000 chars"
            )
            logger.debug(status_msg)
            content = content[:truncate_length]
            extraction_prompt = EXTRACTOR_PROMPT.format(
                webpage_content=content, goal=goal
            )
            messages = [{"role": "user", "content": extraction_prompt}]
            raw = await summary_page_func(messages, max_retries=max_retries)
            summary_retries -= 1

        if isinstance(raw, str):
            raw = raw.replace("```json", "").replace("```", "").strip()

        logger.debug("Final raw response:", len(raw))

        parse_retry_times = 0
        while parse_retry_times < 3:
            try:
                raw_obj = json.loads(raw) if isinstance(raw, str) else raw
                break
            except Exception:
                raw = await summary_page_func(messages, max_retries=max_retries)
                parse_retry_times += 1
        else:
            raw_obj = None

        if raw_obj is None:
            useful_information = f"The provided webpage content: {_content[:1000]}"
            return useful_information

        useful_information = (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
        )
        useful_information += (
            "Evidence in page: \n" + str(raw_obj.get("evidence", "")) + "\n\n"
        )
        useful_information += "Summary: \n" + str(raw_obj.get("summary", "")) + "\n\n"

        if len(useful_information) < 10 and summary_retries < 0:
            logger.debug(
                "[visit] Could not generate valid summary after maximum retries"
            )
            useful_information = "[visit] Failed to read page"
        logger.debug(f"Final useful_information: {useful_information}")
        return useful_information
