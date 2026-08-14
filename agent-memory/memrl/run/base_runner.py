# memp/run/base_runner.py
from abc import ABC, abstractmethod

class BaseRunner(ABC):
    """
    Abstract Base Class for an experiment runner.

    The Runner is responsible for orchestrating the entire interaction between
    the agent, the environment, and any other services (like memory).
    """

    @abstractmethod
    def run(self, num_episodes: int):
        """
        The main entry point to start the experiment.

        Args:
            num_episodes (int): The total number of episodes to run.
        """
        pass