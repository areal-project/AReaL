"""
AveFact key generation for the retrieval strategy.

This module implements the AveFact (Average of Facts) approach for generating
retrieval keys from task descriptions using keyword extraction and embedding averaging.
"""

from typing import List, Optional
import numpy as np

from ..providers.base import BaseLLM, BaseEmbedder
from ..providers.embedding import AverageEmbedder


class AveFactKeyer:
    """
    Generates AveFact retrieval keys from task descriptions.
    
    The AveFact strategy works by:
    1. Extracting keywords from the task description using an LLM
    2. Computing embeddings for each keyword
    3. Averaging the embeddings to create a retrieval key vector
    
    This approach often works better than using the full task description
    because it focuses on key concepts while filtering out noise.
    """
    
    def __init__(self, llm: BaseLLM, embedder: BaseEmbedder, max_keywords: int = 8):
        """
        Initialize AveFact key generator.
        
        Args:
            llm: LLM provider for keyword extraction
            embedder: Embedding provider for vectorization
            max_keywords: Maximum number of keywords to extract
        """
        self.llm = llm
        self.embedder = embedder
        self.max_keywords = max_keywords
        self.average_embedder = AverageEmbedder()
    
    def generate_key(self, task_description: str) -> List[float]:
        """
        Generate AveFact retrieval key for a task description.
        
        Args:
            task_description: Natural language task description
            
        Returns:
            Average embedding vector for the extracted keywords
            
        Raises:
            Exception: If key generation fails
        """
        # Step 1: Extract keywords using LLM
        keywords = self.llm.extract_keywords(task_description, self.max_keywords)
        
        if not keywords:
            # Fallback: use the full task description
            keywords = [task_description]
        
        # Step 2: Generate embeddings for keywords
        embeddings = self.embedder.embed(keywords)
        
        # Step 3: Compute average embedding
        avg_embedding = self.average_embedder.average_embeddings(embeddings)
        
        return avg_embedding
    
    def generate_keys_batch(self, task_descriptions: List[str]) -> List[List[float]]:
        """
        Generate AveFact keys for multiple task descriptions.
        
        Args:
            task_descriptions: List of task descriptions
            
        Returns:
            List of average embedding vectors
        """
        keys = []
        for task_desc in task_descriptions:
            key = self.generate_key(task_desc)
            keys.append(key)
        return keys
    
    def generate_weighted_key(
        self, 
        task_description: str, 
        keyword_weights: Optional[List[float]] = None
    ) -> List[float]:
        """
        Generate weighted AveFact key with custom keyword importance.
        
        Args:
            task_description: Natural language task description
            keyword_weights: Optional weights for each keyword (must match keyword count)
            
        Returns:
            Weighted average embedding vector
        """
        keywords = self.llm.extract_keywords(task_description, self.max_keywords)
        
        if not keywords:
            keywords = [task_description]
        
        embeddings = self.embedder.embed(keywords)
        
        if keyword_weights:
            if len(keyword_weights) != len(keywords):
                raise ValueError("Number of weights must match number of keywords")
            avg_embedding = self.average_embedder.weighted_average_embeddings(
                embeddings, keyword_weights
            )
        else:
            avg_embedding = self.average_embedder.average_embeddings(embeddings)
        
        return avg_embedding


class SimpleKeyer:
    """
    Simple key generator that uses full task description embeddings.
    
    This is used for the Query retrieval strategy, which directly embeds
    the full task description without keyword extraction.
    """
    
    def __init__(self, embedder: BaseEmbedder):
        """
        Initialize simple key generator.
        
        Args:
            embedder: Embedding provider for vectorization
        """
        self.embedder = embedder
    
    def generate_key(self, task_description: str) -> List[float]:
        """
        Generate simple retrieval key from full task description.
        
        Args:
            task_description: Natural language task description
            
        Returns:
            Embedding vector for the full task description
        """
        return self.embedder.embed_single(task_description)
    
    def generate_keys_batch(self, task_descriptions: List[str]) -> List[List[float]]:
        """
        Generate simple keys for multiple task descriptions.
        
        Args:
            task_descriptions: List of task descriptions
            
        Returns:
            List of embedding vectors
        """
        return self.embedder.embed(task_descriptions)


class RandomKeyer:
    """
    Random key generator for baseline comparisons.
    
    This generates random vectors and is used with the Random retrieval
    strategy as a baseline to show the importance of semantic similarity.
    """
    
    def __init__(self, embedding_dim: int = 384, seed: Optional[int] = None):
        """
        Initialize random key generator.

        Args:
            embedding_dim: Dimension of generated random vectors
            seed: Random seed for reproducible results
        """
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate_key(self, task_description: str) -> List[float]:
        """
        Generate random retrieval key.

        Args:
            task_description: Task description (ignored)

        Returns:
            Random vector
        """
        return self.rng.normal(0, 1, self.embedding_dim).tolist()
    
    def generate_keys_batch(self, task_descriptions: List[str]) -> List[List[float]]:
        """
        Generate random keys for multiple task descriptions.
        
        Args:
            task_descriptions: List of task descriptions
            
        Returns:
            List of random vectors
        """
        return [self.generate_key(desc) for desc in task_descriptions]