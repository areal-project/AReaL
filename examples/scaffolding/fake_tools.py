"""
Fake search and visit tools for the search scaffolding example.

These tools return canned results so the example can run without
external API keys (no SERPER_KEY or JINA_API_KEYS required).
"""


async def fake_search(queries: list[str]) -> str:
    """Return fake search results for each query.

    Parameters
    ----------
    queries : list[str]
        List of search query strings.

    Returns
    -------
    str
        Fake search results formatted like the real search tool output.
    """
    results = []
    for query in queries:
        snippet = (
            f"A Google search for '{query}' found 3 results:\n\n"
            f"## Web Results\n"
            f"1. [Wikipedia - {query}](https://en.wikipedia.org/wiki/{query.replace(' ', '_')})\n"
            f"Source: Wikipedia\n"
            f"This article provides an overview of {query}.\n\n"
            f"2. [Britannica - {query}](https://www.britannica.com/topic/{query.replace(' ', '-')})\n"
            f"Source: Britannica\n"
            f"A comprehensive reference on {query} with detailed analysis.\n\n"
            f"3. [Research Paper on {query}](https://arxiv.org/abs/2401.00001)\n"
            f"Source: arXiv\n"
            f"Recent academic research related to {query}."
        )
        results.append(snippet)
    return "\n=======\n".join(results)


async def fake_visit(urls: list[str], goal: str) -> str:
    """Return fake webpage content summaries.

    Parameters
    ----------
    urls : list[str]
        List of URLs to visit.
    goal : str
        The information goal for visiting the webpages.

    Returns
    -------
    str
        Fake webpage content summaries for each URL.
    """
    results = []
    for url in urls:
        summary = (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
            f"Evidence in page: \n"
            f"The webpage discusses topics related to {goal}. "
            f"Key findings include several relevant data points and references "
            f"that contribute to understanding the subject matter. "
            f"The content covers historical context, current developments, "
            f"and expert opinions on the topic.\n\n"
            f"Summary: \n"
            f"This source provides relevant background information about {goal}. "
            f"The main conclusions support a factual understanding of the topic "
            f"based on available evidence.\n\n"
        )
        results.append(summary)
    return "\n=======\n".join(results)
