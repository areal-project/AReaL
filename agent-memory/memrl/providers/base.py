"""
Base classes for LLM and Embedding providers.

This module defines abstract base classes that all providers must implement,
ensuring consistent interfaces across different service providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model providers.
    
    All LLM providers must implement the methods defined in this class to ensure
    consistent behavior across different models and services.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LLM provider with configuration parameters."""
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Generate a response from the LLM given input messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response string
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod 
    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        """
        Extract key concepts/keywords from text for AveFact strategy.
        
        This method is crucial for the AveFact retrieval strategy, which uses
        keyword-based vector averaging for memory retrieval.
        
        Args:
            text: Input text to extract keywords from
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords (up to max_keywords)
            
        Raises:
            Exception: If keyword extraction fails
        """
        pass
    
    def generate_script(self, trajectory: str) -> str:
        """
        Generate a high-level script from a task trajectory.
        
        This is used by the Script and Proceduralization build strategies
        to create abstract representations of successful task completions.
        
        Args:
            trajectory: Detailed step-by-step trajectory
            
        Returns:
            High-level script representation
        """
        prompt = f"""
        Analyze the following task trajectory and generate a concise, high-level script 
        that captures the key steps and decision points:
        
        {trajectory}
        
        Generate a script with 3-5 high-level steps that could guide similar tasks.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate(messages, temperature=0)


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding providers.
    
    Embedding providers are used to convert text into vector representations
    for similarity search and retrieval operations.
    """
    
    def __init__(self, max_text_len: int = 8196, **kwargs: Any) -> None:
        """
        Initialize the embedding provider with configuration parameters.

        Args:
            max_text_len: Maximum characters allowed per query before chunking.
                Longer queries will be split into fixed-size chunks and the
                resulting embeddings will be averaged. Set to 0 or a negative
                value to disable chunking.
        """
        self.max_text_len = max_text_len
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per input text)
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector for the input text
        """
        return self.embed([text])[0]

    # ---- chunking helpers -------------------------------------------------
    def _chunk_texts(self, texts: List[str]) -> tuple[List[str], List[int]]:
        """
        Split texts into chunks according to max_text_len.

        Returns:
            A tuple of (chunked_texts, chunk_counts_per_text)
        """
        if self.max_text_len and self.max_text_len > 0:
            chunk_size = self.max_text_len
        else:
            chunk_size = None

        chunked: List[str] = []
        counts: List[int] = []
        for text in texts:
            if chunk_size and len(text) > chunk_size:
                pieces = [
                    text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
                ]
            else:
                pieces = [text]
            chunked.extend(pieces)
            counts.append(len(pieces))
        return chunked, counts

    @staticmethod
    def _merge_chunk_embeddings(
        chunk_embeddings: List[List[float]], counts: List[int]
    ) -> List[List[float]]:
        """
        Merge chunk-level embeddings back to per-text embeddings by averaging.
        """
        merged: List[List[float]] = []
        idx = 0
        for count in counts:
            group = chunk_embeddings[idx : idx + count]
            idx += count
            if not group:
                merged.append([])
                continue
            if count == 1:
                merged.append(group[0])
            else:
                merged.append(BaseEmbedder._average_vectors(group))
        return merged

    @staticmethod
    def _average_vectors(vectors: List[List[float]]) -> List[float]:
        """Compute element-wise average of vectors without extra dependencies."""
        if not vectors:
            return []
        length = len(vectors[0])
        sums = [0.0] * length
        for vec in vectors:
            for i in range(length):
                sums[i] += float(vec[i])
        return [s / len(vectors) for s in sums]


class ProviderError(Exception):
    """Base exception class for provider-related errors."""
    pass


class LLMError(ProviderError):
    """Exception raised for LLM-specific errors."""
    pass


class EmbedderError(ProviderError):
    """Exception raised for embedding-specific errors."""
    pass
