# memp/envs/base.py
from abc import ABC, abstractmethod

class IEnv(ABC):
    """
    Abstract base environment interface.
    Defines a unified API for different environments.
    """

    @abstractmethod
    def reset(self, task=None):
        """
        Reset the environment.
        Args:
            task (optional): task configuration or identifier.
        Returns:
            state (dict): initial environment state.
        """
        pass

    @abstractmethod
    def step(self, action: str):
        """
        Perform one environment step.
        Args:
            action (str): action string to execute.
        Returns:
            state (dict): environment state after the action.
            reward (float): step reward.
            done (bool): whether the episode is finished.
            info (dict): extra information.
        """
        pass

    @abstractmethod
    def current_trace(self):
        """
        Return the current interaction trace.
        Typically includes states, actions, and rewards.
        """
        pass

    @abstractmethod
    def close(self):
        """Close the environment and free resources."""
        pass
