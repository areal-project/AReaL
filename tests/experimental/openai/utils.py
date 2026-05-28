import os

from openai import AsyncOpenAI


class SimpleAgent:
    async def run(self, data: dict, **extra_kwargs) -> float:
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        _ = await client.chat.completions.create(
            messages=data["messages"], model="default"
        )
        return 1.0
