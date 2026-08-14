"""
BigCodeBench (BCB) integration utilities.

This package contains the minimal glue needed to:
- load BigCodeBench JSONL datasets
- construct prompts
- decode with Memp's LLM + MemoryService
- evaluate using the official BigCodeBench checker (vendored under 3rdparty/)
"""

from .task_wrappers import load_bcb_data, split_dataset, get_prompt, write_samples
from .bcb_adapter import MempBCBDecoder, extract_code_from_response

__all__ = [
    "load_bcb_data",
    "split_dataset",
    "get_prompt",
    "write_samples",
    "MempBCBDecoder",
    "extract_code_from_response",
]

