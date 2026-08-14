"""
Memory builders for different build strategies in the Memp system.

This module implements the Builder pattern for constructing procedural memories
according to different build strategies: Trajectory, Script, and Proceduralization.
"""

import logging
import ast
import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from .strategies import BuildStrategy
from ..providers.base import BaseLLM

logger = logging.getLogger(__name__)


class MemoryBuilder(ABC):
    """
    Abstract base class for memory builders.

    Each builder implements a specific build strategy for creating
    procedural memory content from task trajectories.
    """

    def __init__(self, llm_provider: Optional[BaseLLM] = None):
        """
        Initialize the memory builder.

        Args:
            llm_provider: LLM provider for script generation (required for some strategies)
        """
        self.llm_provider = llm_provider

    @abstractmethod
    def build(self, task_description: str, trajectory: str) -> str:
        """
        Build memory content from task description and trajectory.

        Args:
            task_description: Natural language description of the task
            trajectory: Detailed step-by-step trajectory of task execution

        Returns:
            Memory content string formatted according to the build strategy

        Raises:
            RuntimeError: If memory building fails
        """
        pass

    def build_batch(self, td2traj: dict[str, str]) -> dict[str, str]:
        """
        串行批处理版本：默认对输入字典逐条调用 build()，避免并发带来的竞争问题。

        参数：
            td2traj: {task_description: trajectory}
        返回：
            {task_description: memory_content}
        说明：
            - 保持向后兼容：不改变单条 build() 的签名与语义
            - 子类如需并行优化，可自行重写，但请注意底层存储的并发约束
        """
        results: dict[str, str] = {}
        for task_description, trajectory in td2traj.items():
            results[task_description] = self.build(task_description, trajectory)
        return results


    @property
    @abstractmethod
    def strategy(self) -> BuildStrategy:
        """Return the build strategy this builder implements."""
        pass


class TrajectoryBuilder(MemoryBuilder):
    """
    Builder for Trajectory strategy.

    This strategy stores the complete trajectory as-is without any processing.
    It's the simplest strategy and serves as a baseline for comparison.
    """

    @staticmethod
    def _compact(value: str, limit: int) -> str:
        value = re.sub(r"\s+", " ", (value or "").strip())
        return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"

    def build(self, task_description: str, trajectory: str) -> str:
        """Store trajectory memories as bounded action/observation summaries.

        The previous raw passthrough created very large prompt payloads and was
        brittle when historical values were not exact Python literals.  Keep the
        trajectory strategy, but materialize a deterministic compact form at
        write time so both new and converted checkpoint memories share a stable
        representation.
        """
        logger.debug(f"Building compact trajectory memory for task: {task_description[:50]}...")
        if not isinstance(trajectory, str):
            trajectory = str(trajectory)
        value = trajectory.strip()
        if value.startswith("COMPACT_TRAJECTORY_V1:"):
            return value
        payload = None
        for parser in (json.loads, ast.literal_eval):
            try:
                payload = parser(value)
                break
            except Exception:
                continue
        if isinstance(payload, dict):
            payload = payload.get("trajectory", payload.get("messages", payload.get("steps")))
        pairs = []
        pending_observation = ""
        if isinstance(payload, list):
            for message in payload:
                if not isinstance(message, dict):
                    continue
                content = message.get("content", "")
                if not isinstance(content, str):
                    continue
                if message.get("role") == "user":
                    if "Now, it's your turn" in content:
                        content = content.split("Now, it's your turn", 1)[-1]
                    pending_observation = self._compact(content, 420)
                elif message.get("role") == "assistant":
                    match = re.search(r"(?im)^\s*Action\s*:\s*(.+)$", content)
                    action = self._compact(match.group(1) if match else content, 240)
                    if action:
                        pairs.append((pending_observation, action))
                        pending_observation = ""
        if not pairs:
            return "COMPACT_TRAJECTORY_V1:\nTask goal: " + self._compact(task_description, 500)
        lines = ["COMPACT_TRAJECTORY_V1:", "Archived Action/Observation Summary:"]
        for idx, (observation, action) in enumerate(pairs[-24:], 1):
            if observation:
                lines.append(f"{idx}. Observation: {observation}")
            lines.append(f"   Action: {action}")
        return "\n".join(lines)

    @property
    def strategy(self) -> BuildStrategy:
        """Return BuildStrategy.TRAJECTORY."""
        return BuildStrategy.TRAJECTORY


class ScriptBuilder(MemoryBuilder):
    """
    Builder for Script strategy.

    This strategy uses an LLM to generate a high-level script from the trajectory,
    creating a more abstract representation of the task completion process.
    """

    def __init__(self, llm_provider: BaseLLM):
        """
        Initialize the script builder.

        Args:
            llm_provider: LLM provider for script generation (required)

        Raises:
            ValueError: If llm_provider is None
        """
        if llm_provider is None:
            raise ValueError("ScriptBuilder requires an LLM provider")
        super().__init__(llm_provider)

    def build(self, task_description: str, trajectory: str) -> str:
        """
        Build memory content using Script strategy.

        Args:
            task_description: Natural language description of the task
            trajectory: Detailed step-by-step trajectory of task execution

        Returns:
            High-level script generated by the LLM

        Raises:
            RuntimeError: If script generation fails
        """
        try:
            logger.debug(f"Generating script for task: {task_description[:50]}...")
            script = self.llm_provider.generate_script(trajectory)
            logger.debug(f"Generated script: {script[:100]}...")
            return script
        except Exception as e:
            logger.error(f"Failed to generate script: {e}")
            raise RuntimeError(f"Script generation failed: {e}")

    @property
    def strategy(self) -> BuildStrategy:
        """Return BuildStrategy.SCRIPT."""
        return BuildStrategy.SCRIPT


class ProceduralizationBuilder(MemoryBuilder):
    """
    Builder for Proceduralization strategy.

    This strategy combines both the high-level script and the detailed trajectory,
    providing both abstract and concrete representations of the task completion.
    """

    def __init__(self, llm_provider: BaseLLM, strip_thinking: bool = False, max_trajectory_len: int = 0):
        """
        Initialize the proceduralization builder.

        Args:
            llm_provider: LLM provider for script generation (required)
            strip_thinking: If True, strip <think> blocks before sending to LLM
            max_trajectory_len: If > 0, truncate trajectory to this length

        Raises:
            ValueError: If llm_provider is None
        """
        if llm_provider is None:
            raise ValueError("ProceduralizationBuilder requires an LLM provider")
        super().__init__(llm_provider)
        self.strip_thinking = strip_thinking
        self.max_trajectory_len = max_trajectory_len

    def build(self, task_description: str, trajectory: str) -> str:
        """
        Build memory content using Proceduralization strategy.

        Args:
            task_description: Natural language description of the task
            trajectory: Detailed step-by-step trajectory of task execution

        Returns:
            Combined content with both script and trajectory in the format:
            "SCRIPT:\n{script}\n\nTRAJECTORY:\n{trajectory}"

        Raises:
            RuntimeError: If script generation fails
        """
        try:
            logger.debug(f"Generating procedural memory for task: {task_description[:50]}...")
            script = self.llm_provider.generate_script(
                trajectory,
                strip_thinking=self.strip_thinking,
                max_trajectory_len=self.max_trajectory_len,
            )

            from memrl.utils.sanitize import sanitize_llm_output
            script = sanitize_llm_output(script)

            if script:
                return script

            # generate_script returned empty after sanitization — fall through
            logger.warning("generate_script returned empty after sanitization, using task_description fallback")
        except Exception as e:
            logger.warning(f"generate_script failed, using task_description fallback: {e}")

        # Fallback: return task_description as a minimal placeholder.
        # Do NOT store raw trajectory — it causes downstream parse failures
        # and pollutes the memory with noise.
        return task_description

    @property
    def strategy(self) -> BuildStrategy:
        """Return BuildStrategy.PROCEDURALIZATION."""
        return BuildStrategy.PROCEDURALIZATION


def get_builder(
    strategy: BuildStrategy,
    llm_provider: Optional[BaseLLM] = None,
    strip_thinking: bool = False,
    max_trajectory_len: int = 0,
) -> MemoryBuilder:
    """
    Factory method to create the appropriate memory builder for a given strategy.

    Args:
        strategy: The build strategy to use
        llm_provider: LLM provider (required for Script and Proceduralization strategies)
        strip_thinking: If True, strip <think> blocks in trajectory before LLM call
        max_trajectory_len: If > 0, truncate trajectory to this many chars

    Returns:
        Appropriate MemoryBuilder instance

    Raises:
        ValueError: If strategy is invalid or required LLM provider is missing
    """
    if strategy == BuildStrategy.TRAJECTORY:
        return TrajectoryBuilder(llm_provider)

    elif strategy == BuildStrategy.SCRIPT:
        if llm_provider is None:
            raise ValueError("Script strategy requires an LLM provider")
        return ScriptBuilder(llm_provider)

    elif strategy == BuildStrategy.PROCEDURALIZATION:
        if llm_provider is None:
            raise ValueError("Proceduralization strategy requires an LLM provider")
        return ProceduralizationBuilder(llm_provider, strip_thinking=strip_thinking, max_trajectory_len=max_trajectory_len)

    else:
        raise ValueError(f"Unknown build strategy: {strategy}")


# Export all builder classes and factory function
__all__ = [
    'MemoryBuilder',
    'TrajectoryBuilder',
    'ScriptBuilder',
    'ProceduralizationBuilder',
    'get_builder'
]
