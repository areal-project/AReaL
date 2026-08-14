# memp/agent/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseAgent(ABC):
    """
    Abstract Base Class for an agent that can act in an environment.
    """
    @abstractmethod
    def reset(self, task_description: str) -> None:
        """
        Resets the agent's state for a new episode.
        This is where memory retrieval should happen.
        """
        pass

    @abstractmethod
    def act(self, observation: str) -> str:
        """
        Given an observation, decide on the next action to take.
        """
        pass

    @abstractmethod
    def get_trajectory(self) -> List[Dict[str, str]]:
        """
        Returns the full trajectory of the completed episode.
        """
        pass