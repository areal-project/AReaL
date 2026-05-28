"""
Real search and visit tools for the search scaffolding example.

Uses the Serper API (SERPER_KEY_ID env var) for web search and
basic aiohttp fetching for page visits.
"""

import asyncio
import json
import os
import re

import aiohttp

SERPER_KEY = os.environ.get("SERPER_KEY_ID", "")

# Maximum characters to keep from a fetched webpage.
_MAX_PAGE_CHARS = 8000


async def real_search(queries: list[str]) -> str:
    """Perform Google searches via the Serper API.

    Parameters
    ----------
    queries : list[str]
        List of search query strings.

    Returns
    -------
    str
        Formatted search results for each query, separated by ``=======``.
    """
    tasks = [_search_single(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return "\n=======\n".join(results)


async def _search_single(query: str) -> str:
    """Search a single query via Serper and return formatted results."""

    def _contains_chinese(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    if _contains_chinese(query):
        payload = {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
    else:
        payload = {"q": query, "location": "United States", "gl": "us", "hl": "en"}

    headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}

    last_exc: Exception | None = None
    async with aiohttp.ClientSession() as session:
        for _attempt in range(5):
            try:
                async with session.post(
                    "https://google.serper.dev/search",
                    json=payload,
                    headers=headers,
                ) as resp:
                    text = await resp.text()
                    try:
                        results = json.loads(text)
                    except Exception:
                        return f"[Search] Failed to parse response for '{query}'."

                    if "organic" not in results:
                        return (
                            f"No results found for query: '{query}'. "
                            "Use a less specific query."
                        )

                    web_snippets = []
                    for idx, page in enumerate(results.get("organic", []), start=1):
                        date_published = (
                            f"\nDate published: {page['date']}"
                            if page.get("date")
                            else ""
                        )
                        source = (
                            f"\nSource: {page['source']}" if page.get("source") else ""
                        )
                        snippet = f"\n{page['snippet']}" if page.get("snippet") else ""
                        entry = (
                            f"{idx}. [{page.get('title', '')}]"
                            f"({page.get('link', '')})"
                            f"{date_published}{source}\n{snippet}"
                        )
                        entry = entry.replace("Your browser can't play this video.", "")
                        web_snippets.append(entry)

                    return (
                        f"A Google search for '{query}' found "
                        f"{len(web_snippets)} results:\n\n## Web Results\n"
                        + "\n\n".join(web_snippets)
                    )
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.5)
                continue

    return (
        f"Google search Timeout or error ({last_exc}); "
        "return None, Please try again later."
    )


def _html_to_text(html: str) -> str:
    """Very basic HTML tag stripping."""
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.I)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def real_visit(urls: list[str], goal: str) -> str:
    """Fetch webpages and return truncated text content.

    Parameters
    ----------
    urls : list[str]
        List of URLs to visit.
    goal : str
        The information goal for visiting the webpages.

    Returns
    -------
    str
        Text content summaries for each URL, separated by ``=======``.
    """
    results = []
    for url in urls:
        content = await _fetch_page(url)
        summary = (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
            f"Evidence in page: \n{content}\n\n"
            f"Summary: \nContent fetched from {url} related to {goal}.\n\n"
        )
        results.append(summary)
    return "\n=======\n".join(results)


async def _fetch_page(url: str) -> str:
    """Fetch a URL and return plain-text content (truncated)."""
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return (
                        f"The provided webpage returned HTTP {resp.status}. "
                        "Please check the URL or try another source."
                    )
                html = await resp.text()
                text = _html_to_text(html)
                if len(text) > _MAX_PAGE_CHARS:
                    text = text[:_MAX_PAGE_CHARS] + "\n... [truncated]"
                return text if text else "The webpage returned empty content."
    except Exception as e:
        return f"Failed to fetch {url}: {e}"
