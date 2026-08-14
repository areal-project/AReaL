"""WebShop prompt templates for MemRL.

Adopted from LaMer's WebShop prompt (admissible-actions style):
`/storage/openpsi/users/yl/LaMer/agent_system/environments/webshop/{prompt.py,env_manager.py}`.

Each step renders the legal action set inline; the LLM picks one and returns
`<action>...</action>`. No few-shot dialogues — the legal-action enumeration is
what makes WebShop tractable for an off-the-shelf LLM.
"""
from typing import Iterable, List, Tuple


WEBSHOP_SYSTEM_PROMPT = (
    "You are an expert autonomous agent operating in the WebShop e-commerce environment.\n"
    "Your job is to find and purchase a product matching the user's shopping goal.\n\n"
    "Strategy guidance:\n"
    "- Search with specific keywords from the goal (attributes, color, size, price).\n"
    "- On a results page, pick a candidate product whose title mentions the most goal attributes.\n"
    "- On a product page, select the required options (color/size/style) before buying.\n"
    "- Click 'Buy Now' once the product matches all goal criteria.\n"
    "- The search bar only exists on the home page. If you want to issue a NEW\n"
    "  search query while on a results or product page, you MUST first click\n"
    "  'Back to Search' to return to the home page; only then can you call\n"
    "  search[...] again.\n"
    "- Only choose actions that appear in the admissible-actions list shown\n"
    "  each step. Never invent action strings outside that list.\n\n"
    "Output format:\n"
    "- Reason briefly (3-5 short sentences max — avoid restating the whole product list).\n"
    "- End with exactly one admissible action wrapped in <action> </action> tags."
)


WEBSHOP_USER_TEMPLATE = (
    "Your task is to: {goal}.{memory_block}{trajectory_block}\n\n"
    "Your admissible actions of the current situation are:\n"
    "[\n"
    "{admissible_actions}\n"
    "].\n\n"
    "Now it's your turn to take one action for the current step.\n"
    "First reason briefly about the current situation in 3-5 short sentences. "
    "Then choose ONE admissible action and present it inside <action> </action> tags. "
    "Do not output anything after the closing </action> tag."
)


WEBSHOP_MEMORY_PREFIX_SUCCESS = "From previous successful shopping experiences:"
WEBSHOP_MEMORY_PREFIX_FAILURE = "Common mistakes to avoid (from past failures):"


def format_admissible_actions(avail: dict) -> str:
    """Render the action list block consumed by ``WEBSHOP_USER_TEMPLATE``.

    Mirrors LaMer's ``format_avail_actions``
    (``LaMer/agent_system/environments/webshop/env_manager.py:177-190``):
    ``has_search_bar`` becomes the leading ``'search[<your query>]'`` row, and
    each clickable text becomes ``'click[<text>]'``.
    """
    if not isinstance(avail, dict):
        return ""

    lines: List[str] = []
    if avail.get("has_search_bar"):
        lines.append("'search[<your query>]'")

    for txt in avail.get("clickables", []) or []:
        if not txt:
            continue
        lines.append(f"'click[{txt}]'")

    return ",\n".join(lines)


def format_webshop_history(
    history: Iterable[Tuple[str, str]],
    max_turns: int = 5,
    obs_chars: int = 400,
) -> str:
    """Render the recent action/observation tail used in the LaMer prompt."""
    history_list = list(history or [])
    if not history_list:
        return ""

    tail = history_list[-max_turns:]
    base_idx = len(history_list) - len(tail)
    rendered: List[str] = []
    for offset, (action, obs) in enumerate(tail):
        turn = base_idx + offset + 1
        rendered.append(f"Action {turn}: {action}")
        snippet = (obs or "")[:obs_chars]
        rendered.append(f"Observation {turn}: {snippet}")
    return "\n".join(rendered)


def format_webshop_memories(retrieved_memories: dict) -> str:
    """Compact memory rendering, untouched from the previous prompt."""
    if not retrieved_memories:
        return ""

    parts: List[str] = []
    success_mems = retrieved_memories.get("successed", [])
    failed_mems = retrieved_memories.get("failed", [])

    if success_mems:
        parts.append(WEBSHOP_MEMORY_PREFIX_SUCCESS)
        for i, mem in enumerate(success_mems, 1):
            content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
            if content:
                parts.append(f"  {i}. {content[:500]}")

    if failed_mems:
        parts.append(WEBSHOP_MEMORY_PREFIX_FAILURE)
        for i, mem in enumerate(failed_mems, 1):
            content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
            if content:
                parts.append(f"  {i}. {content[:500]}")

    return "\n".join(parts)
