from typing import Optional


POST_PAUSE_CONTINUATION_TOKEN_QUOTA = 4096


def get_post_pause_request_max_tokens(
    remaining_context_tokens: int, remaining_quota_tokens: Optional[int]
) -> int:
    """Compute max_tokens for a post-pause continuation request."""
    if remaining_context_tokens <= 0:
        return 0
    if remaining_quota_tokens is None:
        return remaining_context_tokens
    if remaining_quota_tokens <= 0:
        return 0
    return min(remaining_context_tokens, remaining_quota_tokens)
