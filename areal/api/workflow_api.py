# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Union


if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.experimental.openai.types import InteractionWithTokenLogpReward


class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        """Run a single episode of the workflow.

        Note
        ----
        Returning `None` implies that this trajectory is rejected and will not be used for training.

        See concrete example implementations under the `areal/workflow` directory.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to use for generating responses
        data : Dict[str, Any]
            Input data for the workflow episode

        Returns
        -------
        Dict[str, Any] | None | Dict[str, InteractionWithTokenLogpReward]
            The trajectory result, None if rejected, or a dictionary of completion results
        """
        raise NotImplementedError()


class _DeprecatedAgentWorkflowMeta(ABCMeta):
    """Metaclass that ensures deprecation warning triggers on any subclass instantiation.

    This approach guarantees the warning fires even if subclasses forget to call
    super().__init__(), since __call__ executes before any __init__ method.
    Inherits from ABCMeta to maintain compatibility with ABC.
    """

    def __call__(cls, *args, **kwargs):
        import warnings

        warnings.warn(
            f"{cls.__name__} inherits from deprecated AgentWorkflow. "
            "You no longer need to inherit from this class. "
            "Any class with a compatible async run() method will work.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__call__(*args, **kwargs)


class AgentWorkflow(ABC, metaclass=_DeprecatedAgentWorkflowMeta):
    """Base class for agent-based workflows (DEPRECATED).

    .. deprecated:: 1.0.0
        Inheriting from AgentWorkflow is no longer required. Any class with
        a compatible ``run()`` method will work. This class is kept for
        backward compatibility but may be removed in a future version.

    To use agent-based workflows, simply implement a class with::

        async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> dict[str, float] | float
    """

    @abstractmethod
    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> dict[str, float] | float:
        """Run an agent with any SDK, e.g., OpenAI SDK.

        `data` contains the input data for the agent, which may
        include any parameters or hyperparameters required.

        `extra_kwargs` includes parameters provided by AReaL:

        - base_url: str
            The base URL of the OpenAI-compatible proxy server
        - http_client: httpx.AsyncClient
            The HTTP client to use for making requests in AsyncOpenAI
        - api_key: str
            The session-scoped API key for authenticating with the proxy

        Parameters
        ----------
        data : dict[str, Any]
            Input data for the agent workflow

        Returns
        -------
        dict[str, float] | float
            The final reward or a dictionary of reward keyed by response ID
        """
        raise NotImplementedError()


# Type alias for workflow parameter across the stack.
# Accepts RolloutWorkflow instances/classes, string import paths, or any
# callable object with a compatible run() method.
WorkflowLike = Union[
    "RolloutWorkflow",
    type["RolloutWorkflow"],
    str,
    Any,  # Any object with async def run(data, **extra_kwargs) method
]
