import asyncio
import json
import sys
from pathlib import Path

import pytest

from areal.api.cli_args import GenerationHyperparameters

# Add the parent directory to the path so we can import TIR modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tir_workflow import TIRConfig, TIRWorkflow  # noqa: E402
from tool_manager import ToolCallStatus, ToolManager  # noqa: E402
from train_tir import math_reward_fn  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@pytest.mark.asyncio
async def test_tool_manager():
    """Test tool manager"""
    python_manager = ToolManager(timeout=10, enabled_tools="python", debug_mode=True)
    calc_manager = ToolManager(timeout=10, enabled_tools="calculator")

    try:
        # Test Python execution through the normal tool-call interface.
        python_result, python_status = await python_manager.aexecute_tool_call(
            "```python\nprint(2 + 3)\n```"
        )
        assert python_status == ToolCallStatus.SUCCESS
        assert python_result == "dummy python output"

        # Test calculator.
        calc_result, calc_status = await calc_manager.aexecute_tool_call(
            "<calculator>2 * 3 + 4</calculator>"
        )
        assert calc_status == ToolCallStatus.SUCCESS
        assert calc_result == "10"

        # Test malformed Python tool input.
        empty_result, empty_status = await python_manager.aexecute_tool_call(
            "```python\n\n```"
        )
        assert empty_status == ToolCallStatus.ERROR
        assert "Error" in empty_result
    finally:
        await python_manager.acleanup()
        await calc_manager.acleanup()


@pytest.mark.asyncio
async def test_tool_manager_error_handling():
    """Test tool manager error handling with real execution"""
    python_manager = ToolManager(timeout=10, enabled_tools="python", debug_mode=False)

    try:
        # Test ZeroDivisionError
        zero_result, zero_status = await python_manager.aexecute_tool_call(
            "```python\n1 / 0\n```"
        )
        print(f"Zero division result: {zero_result}, status: {zero_status}")
        assert zero_status == ToolCallStatus.ERROR
        assert (
            "division by zero" in zero_result.lower()
            or "zerodivisionerror" in zero_result.lower()
        )

    finally:
        await python_manager.acleanup()


@pytest.mark.asyncio
async def test_tir_workflow():
    """Test TIR workflow initialization."""
    tokenizer = FakeTokenizer()

    workflow = TIRWorkflow(
        reward_fn=math_reward_fn,
        gconfig=GenerationHyperparameters(max_new_tokens=32, max_tokens=256),
        tokenizer=tokenizer,
        tir_config=TIRConfig(max_turns=3, max_length=2000),
    )

    assert workflow.tool_manager is not None
    assert workflow.start_markers
    assert workflow.end_markers

    await workflow.tool_manager.acleanup()


def test_data_loading():
    """Test data loading"""
    data_file = Path(__file__).parent / "data" / "sample_math.jsonl"
    assert data_file.exists(), f"Data file not found: {data_file}"

    with open(data_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    assert len(data) > 0, "No data loaded"
    assert "messages" in data[0], "Missing 'messages' field"
    assert "answer" in data[0], "Missing 'answer' field"


async def main():
    """Run all tests"""
    try:
        await test_tool_manager()
        await test_tir_workflow()
        test_data_loading()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
