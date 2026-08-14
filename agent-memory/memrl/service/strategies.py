"""
Strategy definitions for the Memp procedural memory system.

This module defines the three types of strategies that can be combined
to create the 9 different strategy combinations of the Memp algorithm.
"""

from enum import Enum


class BuildStrategy(Enum):
    """
    Build strategies define how procedural memories are constructed from trajectories.
    
    The build phase determines what information is stored in memory and how
    it is represented for later retrieval and use.
    """
    
    TRAJECTORY = "trajectory"
    """Store complete task trajectories directly."""
    
    SCRIPT = "script" 
    """Store only LLM-generated high-level scripts."""
    
    PROCEDURALIZATION = "proceduralization"
    """Store both trajectories and scripts (recommended main method)."""


class RetrieveStrategy(Enum):
    """
    Retrieve strategies define how relevant memories are found for new tasks.
    
    The retrieval phase determines which existing memories are most relevant
    to assist with a new task.
    """
    
    RANDOM = "random"
    """Randomly sample memories (baseline for comparison)."""
    
    QUERY = "query"
    """Use original task description vector as retrieval key."""
    
    AVEFACT = "avefact"
    """Use average of keyword vectors as retrieval key (recommended main method)."""


class UpdateStrategy(Enum):
    """
    Update strategies define how the memory system learns from new experiences.
    
    The update phase determines when and how new experiences are added to
    memory, and whether existing memories should be modified.
    """
    
    VANILLA = "vanilla"
    """Add all new task trajectories to memory."""
    
    VALIDATION = "validation"
    """Only add successful trajectories, filter out failures."""
    
    ADJUSTMENT = "adjustment"
    """Reflect and adjust existing memories when tasks fail (recommended main method)."""


class StrategyConfiguration:
    """
    Container for a complete strategy configuration.
    
    Combines all three strategy types into a single configuration object
    that can be used to initialize a MemoryService.
    """
    
    def __init__(
        self, 
        build: BuildStrategy,
        retrieve: RetrieveStrategy,
        update: UpdateStrategy
    ):
        """
        Initialize strategy configuration.
        
        Args:
            build: Build strategy to use
            retrieve: Retrieve strategy to use  
            update: Update strategy to use
        """
        self.build = build
        self.retrieve = retrieve
        self.update = update
    
    @classmethod
    def main_combination(cls) -> "StrategyConfiguration":
        """
        Get the main recommended strategy combination.
        
        Default combination in this implementation:
        Proceduralization + Query + Adjustment
        
        Returns:
            StrategyConfiguration with the main combination
        """
        return cls(
            build=BuildStrategy.PROCEDURALIZATION,
            retrieve=RetrieveStrategy.QUERY,
            update=UpdateStrategy.ADJUSTMENT
        )
    
    @classmethod
    def baseline_combination(cls) -> "StrategyConfiguration":
        """
        Get a simple baseline strategy combination.
        
        Returns:
            StrategyConfiguration with baseline strategies
        """
        return cls(
            build=BuildStrategy.TRAJECTORY,
            retrieve=RetrieveStrategy.RANDOM,
            update=UpdateStrategy.VANILLA
        )
    
    @classmethod
    def from_strings(cls, build: str, retrieve: str, update: str) -> "StrategyConfiguration":
        """
        Create strategy configuration from string names.
        
        Args:
            build: Build strategy name
            retrieve: Retrieve strategy name
            update: Update strategy name
            
        Returns:
            StrategyConfiguration object
            
        Raises:
            ValueError: If any strategy name is invalid
        """
        try:
            build_strategy = BuildStrategy(build.lower())
            retrieve_strategy = RetrieveStrategy(retrieve.lower())
            update_strategy = UpdateStrategy(update.lower())
            
            return cls(build_strategy, retrieve_strategy, update_strategy)
            
        except ValueError as e:
            raise ValueError(f"Invalid strategy name: {e}")
    
    def __str__(self) -> str:
        """String representation of the strategy configuration."""
        return f"{self.build.value}+{self.retrieve.value}+{self.update.value}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"StrategyConfiguration("
                f"build={self.build.value}, "
                f"retrieve={self.retrieve.value}, "
                f"update={self.update.value})")
    
    def __eq__(self, other) -> bool:
        """Check equality with another StrategyConfiguration."""
        if not isinstance(other, StrategyConfiguration):
            return False
        return (self.build == other.build and 
                self.retrieve == other.retrieve and 
                self.update == other.update)
    
    def __hash__(self) -> int:
        """Hash function for use as dictionary keys."""
        return hash((self.build, self.retrieve, self.update))


# Predefined configurations for easy access
MAIN_STRATEGY = StrategyConfiguration.main_combination()
BASELINE_STRATEGY = StrategyConfiguration.baseline_combination()

# All possible strategy combinations (27 total)
ALL_STRATEGIES = [
    StrategyConfiguration(build, retrieve, update)
    for build in BuildStrategy
    for retrieve in RetrieveStrategy  
    for update in UpdateStrategy
]