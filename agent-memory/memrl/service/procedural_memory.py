"""
ProceduralMemory schema and data structures for the Memp system.

This module defines the core data structures for procedural memories,
extending MemOS's TextualMemoryItem with Memp-specific metadata and functionality.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata

from .strategies import BuildStrategy, RetrieveStrategy, UpdateStrategy


class MemoryType(Enum):
    """Types of procedural memories in the Memp system."""
    TRAJECTORY = "trajectory"
    SCRIPT = "script"
    PROCEDURE = "procedure"


@dataclass
class MempMetadata:
    """Extended metadata for Memp procedural memories."""
    
    # Core identification
    task_description: str
    memory_type: MemoryType
    
    # Strategy information
    build_strategy: BuildStrategy
    retrieve_strategy: RetrieveStrategy
    update_strategy: UpdateStrategy
    
    # Source and context
    source_benchmark: str = "unknown"  # "TravelPlanner" | "ALFWorld" | "unknown"
    source_episode_id: Optional[str] = None
    
    # Memory content details
    trajectory_content: Optional[str] = None  # Full trajectory for reference
    script_content: Optional[str] = None      # Generated script content
    
    # Retrieval and similarity
    avefact_keywords: Optional[List[str]] = None
    avefact_vector: Optional[List[float]] = None
    query_vector: Optional[List[float]] = None
    
    # Performance and quality metrics
    confidence_score: float = 100.0
    success_rate: Optional[float] = None
    retrieval_count: int = 0
    last_retrieved: Optional[str] = None
    
    # Versioning and updates
    version: int = 1
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        
        # Convert enums to strings
        result['memory_type'] = self.memory_type.value
        result['build_strategy'] = self.build_strategy.value
        result['retrieve_strategy'] = self.retrieve_strategy.value
        result['update_strategy'] = self.update_strategy.value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MempMetadata':
        """Create from dictionary (for deserialization)."""
        # Convert string enums back to enum objects
        if 'memory_type' in data:
            data['memory_type'] = MemoryType(data['memory_type'])
        if 'build_strategy' in data:
            data['build_strategy'] = BuildStrategy(data['build_strategy'])
        if 'retrieve_strategy' in data:
            data['retrieve_strategy'] = RetrieveStrategy(data['retrieve_strategy'])
        if 'update_strategy' in data:
            data['update_strategy'] = UpdateStrategy(data['update_strategy'])
            
        return cls(**data)


class ProceduralMemory:
    """
    Wrapper class for procedural memories in the Memp system.
    
    This class encapsulates a MemOS TextualMemoryItem with Memp-specific
    metadata and provides convenient methods for memory operations.
    """
    
    def __init__(
        self,
        task_description: str,
        memory_content: str,
        memp_metadata: MempMetadata,
        memory_id: Optional[str] = None
    ):
        """
        Initialize a procedural memory.
        
        Args:
            task_description: Natural language description of the task
            memory_content: The actual memory content (trajectory/script/combined)
            memp_metadata: Memp-specific metadata
            memory_id: Optional memory ID (generated if not provided)
        """
        self.memory_id = memory_id or str(uuid.uuid4())
        self.task_description = task_description
        self.memory_content = memory_content
        self.memp_metadata = memp_metadata
        
        # Create MemOS TextualMemoryItem
        self._create_textual_memory_item()
    
    def _create_textual_memory_item(self) -> None:
        """Create the underlying MemOS TextualMemoryItem."""
        # Prepare MemOS metadata
        memos_metadata = TextualMemoryMetadata(
            type=self.memp_metadata.memory_type.value,
            source="conversation",
            entities=[self.task_description],
            tags=self._generate_tags(),
            visibility="private",
            confidence=self.memp_metadata.confidence_score,
            memory_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=self.memp_metadata.updated_at
        )
        
        # Create TextualMemoryItem
        self.textual_memory_item = TextualMemoryItem(
            id=self.memory_id,
            memory=self.memory_content,
            metadata=memos_metadata
        )
    
    def _generate_tags(self) -> List[str]:
        """Generate tags for MemOS based on Memp metadata."""
        tags = [
            "memp",
            "procedural",
            self.memp_metadata.memory_type.value,
            self.memp_metadata.build_strategy.value,
            self.memp_metadata.source_benchmark.lower()
        ]
        
        # Add strategy-specific tags
        if self.memp_metadata.retrieve_strategy == RetrieveStrategy.AVEFACT:
            tags.append("avefact")
        elif self.memp_metadata.retrieve_strategy == RetrieveStrategy.QUERY:
            tags.append("query")
        elif self.memp_metadata.retrieve_strategy == RetrieveStrategy.RANDOM:
            tags.append("random")
            
        return tags
    
    def update_retrieval_stats(self) -> None:
        """Update retrieval statistics when memory is accessed."""
        self.memp_metadata.retrieval_count += 1
        self.memp_metadata.last_retrieved = datetime.now().isoformat()
        self.memp_metadata.updated_at = self.memp_metadata.last_retrieved
        
        # Update the underlying TextualMemoryItem
        self._create_textual_memory_item()
    
    def update_content(self, new_content: str, adjustment_reason: Optional[str] = None) -> None:
        """Update memory content (for Adjustment strategy)."""
        self.memory_content = new_content
        self.memp_metadata.version += 1
        self.memp_metadata.updated_at = datetime.now().isoformat()
        
        # Add adjustment information to metadata
        if adjustment_reason:
            if not hasattr(self.memp_metadata, 'adjustment_history'):
                self.memp_metadata.adjustment_history = []
            self.memp_metadata.adjustment_history.append({
                'timestamp': self.memp_metadata.updated_at,
                'reason': adjustment_reason,
                'version': self.memp_metadata.version
            })
        
        # Recreate TextualMemoryItem with updated content
        self._create_textual_memory_item()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'memory_id': self.memory_id,
            'task_description': self.task_description,
            'memory_content': self.memory_content,
            'memp_metadata': self.memp_metadata.to_dict(),
            'textual_memory_item': {
                'id': self.textual_memory_item.id,
                'memory': self.textual_memory_item.memory,
                'metadata': self.textual_memory_item.metadata.__dict__ if self.textual_memory_item.metadata else None
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProceduralMemory':
        """Create from dictionary (for deserialization)."""
        memp_metadata = MempMetadata.from_dict(data['memp_metadata'])
        
        return cls(
            task_description=data['task_description'],
            memory_content=data['memory_content'],
            memp_metadata=memp_metadata,
            memory_id=data['memory_id']
        )
    
    @classmethod
    def create_trajectory_memory(
        cls,
        task_description: str,
        trajectory: str,
        build_strategy: BuildStrategy,
        retrieve_strategy: RetrieveStrategy,
        update_strategy: UpdateStrategy,
        source_benchmark: str = "unknown",
        **kwargs
    ) -> 'ProceduralMemory':
        """Factory method for creating trajectory-based memories."""
        memp_metadata = MempMetadata(
            task_description=task_description,
            memory_type=MemoryType.TRAJECTORY,
            build_strategy=build_strategy,
            retrieve_strategy=retrieve_strategy,
            update_strategy=update_strategy,
            source_benchmark=source_benchmark,
            trajectory_content=trajectory,
            **kwargs
        )
        
        return cls(
            task_description=task_description,
            memory_content=trajectory,
            memp_metadata=memp_metadata
        )
    
    @classmethod
    def create_script_memory(
        cls,
        task_description: str,
        script: str,
        trajectory: str,
        build_strategy: BuildStrategy,
        retrieve_strategy: RetrieveStrategy,
        update_strategy: UpdateStrategy,
        source_benchmark: str = "unknown",
        **kwargs
    ) -> 'ProceduralMemory':
        """Factory method for creating script-based memories."""
        memp_metadata = MempMetadata(
            task_description=task_description,
            memory_type=MemoryType.SCRIPT,
            build_strategy=build_strategy,
            retrieve_strategy=retrieve_strategy,
            update_strategy=update_strategy,
            source_benchmark=source_benchmark,
            trajectory_content=trajectory,
            script_content=script,
            **kwargs
        )
        
        return cls(
            task_description=task_description,
            memory_content=script,
            memp_metadata=memp_metadata
        )
    
    @classmethod
    def create_procedural_memory(
        cls,
        task_description: str,
        script: str,
        trajectory: str,
        build_strategy: BuildStrategy,
        retrieve_strategy: RetrieveStrategy,
        update_strategy: UpdateStrategy,
        source_benchmark: str = "unknown",
        **kwargs
    ) -> 'ProceduralMemory':
        """Factory method for creating proceduralization memories (script + trajectory)."""
        combined_content = f"SCRIPT:\n{script}\n\nTRAJECTORY:\n{trajectory}"
        
        memp_metadata = MempMetadata(
            task_description=task_description,
            memory_type=MemoryType.PROCEDURE,
            build_strategy=build_strategy,
            retrieve_strategy=retrieve_strategy,
            update_strategy=update_strategy,
            source_benchmark=source_benchmark,
            trajectory_content=trajectory,
            script_content=script,
            **kwargs
        )
        
        return cls(
            task_description=task_description,
            memory_content=combined_content,
            memp_metadata=memp_metadata
        )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"ProceduralMemory(id={self.memory_id[:8]}..., "
                f"type={self.memp_metadata.memory_type.value}, "
                f"strategy={self.memp_metadata.build_strategy.value}/"
                f"{self.memp_metadata.retrieve_strategy.value}/"
                f"{self.memp_metadata.update_strategy.value})")
