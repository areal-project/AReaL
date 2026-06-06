#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Chunked LLM Server Example

This script demonstrates how to serve an LLM with chunked rotary encoding support.
It provides a simple interface for generating text with chunk-aware positional encoding.

Usage:
    python serve_chunked_llm.py --model MODEL_NAME [--host HOST] [--port PORT]
"""

import argparse
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
except ImportError:
    print("Warning: vLLM not installed. This is a demonstration script.")
    print("Install vLLM with: pip install vllm")
    LLM = None
    SamplingParams = None
    RequestOutput = None


@dataclass
class ChunkedPromptConfig:
    """Configuration for chunked prompt processing."""
    chunk_start_marker: str = "<Chunk>"
    chunk_end_marker: str = "</Chunk>"
    validate_pairing: bool = True
    strip_markers_from_output: bool = False


class ChunkedLLMServer:
    """
    LLM server with support for chunked rotary positional encoding.
    
    This server handles prompts containing <Chunk>...</Chunk> markers and applies
    special positional encoding as defined in ChunkedRotaryEmbedding.
    
    Example:
        server = ChunkedLLMServer("meta-llama/Llama-2-7b-hf")
        
        prompt = '''
        Analyze these documents:
        <Chunk>Document 1: Information about quantum computing.</Chunk>
        <Chunk>Document 2: Information about classical computing.</Chunk>
        What are the key differences?
        '''
        
        results = server.generate([prompt])
        print(results[0])
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[ChunkedPromptConfig] = None,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        **llm_kwargs,
    ):
        """
        Initialize the chunked LLM server.
        
        Args:
            model_name: Name or path of the model to load
            config: Configuration for chunk processing
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type for model weights ("auto", "float16", "bfloat16")
            **llm_kwargs: Additional arguments passed to vLLM's LLM class
        """
        if LLM is None:
            raise ImportError("vLLM is required. Install with: pip install vllm")
        
        self.config = config or ChunkedPromptConfig()
        
        print(f"Loading model: {model_name}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"dtype: {dtype}")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,  # May be needed for custom models
            **llm_kwargs,
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        
        # Try to get chunk token IDs (they may not exist in all tokenizers)
        try:
            self.chunk_start_token_id = self.tokenizer.convert_tokens_to_ids(
                self.config.chunk_start_marker
            )
            self.chunk_end_token_id = self.tokenizer.convert_tokens_to_ids(
                self.config.chunk_end_marker
            )
            
            print(f"✓ Chunk start marker '{self.config.chunk_start_marker}' -> ID: {self.chunk_start_token_id}")
            print(f"✓ Chunk end marker '{self.config.chunk_end_marker}' -> ID: {self.chunk_end_token_id}")
            
            # Check if these are valid token IDs (not UNK token)
            unk_token_id = self.tokenizer.unk_token_id
            if self.chunk_start_token_id == unk_token_id or self.chunk_end_token_id == unk_token_id:
                print("⚠ WARNING: Chunk markers map to UNK token. They may not be in vocabulary.")
                print("  Consider adding them as special tokens to your tokenizer.")
        except Exception as e:
            print(f"⚠ WARNING: Could not find chunk markers in tokenizer: {e}")
            self.chunk_start_token_id = None
            self.chunk_end_token_id = None
    
    def validate_chunk_markers(self, text: str) -> bool:
        """
        Validate that chunk markers are properly paired in the text.
        
        Args:
            text: Input text to validate
            
        Returns:
            True if valid, False otherwise
        """
        start_count = text.count(self.config.chunk_start_marker)
        end_count = text.count(self.config.chunk_end_marker)
        
        if start_count != end_count:
            return False
        
        # Check proper nesting (simple stack-based validation)
        depth = 0
        pos = 0
        while pos < len(text):
            start_pos = text.find(self.config.chunk_start_marker, pos)
            end_pos = text.find(self.config.chunk_end_marker, pos)
            
            if start_pos == -1 and end_pos == -1:
                break
            
            if start_pos != -1 and (end_pos == -1 or start_pos < end_pos):
                depth += 1
                pos = start_pos + len(self.config.chunk_start_marker)
            else:
                depth -= 1
                if depth < 0:
                    return False
                pos = end_pos + len(self.config.chunk_end_marker)
        
        return depth == 0
    
    def preprocess_prompt(self, prompt: str) -> str:
        """
        Preprocess a prompt before sending to the model.
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            Preprocessed prompt
        """
        # Validate chunk markers if enabled
        if self.config.validate_pairing:
            if not self.validate_chunk_markers(prompt):
                raise ValueError(
                    f"Invalid chunk markers in prompt. "
                    f"Ensure all {self.config.chunk_start_marker} have matching {self.config.chunk_end_marker}"
                )
        
        return prompt
    
    def postprocess_output(self, output: str) -> str:
        """
        Postprocess generated output.
        
        Args:
            output: Raw model output
            
        Returns:
            Postprocessed output
        """
        if self.config.strip_markers_from_output:
            output = output.replace(self.config.chunk_start_marker, "")
            output = output.replace(self.config.chunk_end_marker, "")
        
        return output
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        return_full_output: bool = False,
    ) -> List[str]:
        """
        Generate completions for prompts with chunk support.
        
        Args:
            prompts: List of prompts (can include chunk markers)
            sampling_params: Sampling parameters (temperature, top_p, etc.)
            return_full_output: If True, return full RequestOutput objects
            
        Returns:
            List of generated texts (or RequestOutput objects if return_full_output=True)
        """
        # Default sampling parameters
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=512,
            )
        
        # Preprocess all prompts
        processed_prompts = [self.preprocess_prompt(p) for p in prompts]
        
        print(f"Generating for {len(processed_prompts)} prompts...")
        
        # Generate
        outputs = self.llm.generate(processed_prompts, sampling_params)
        
        if return_full_output:
            return outputs
        
        # Extract and postprocess text from outputs
        results = []
        for output in outputs:
            text = output.outputs[0].text
            text = self.postprocess_output(text)
            results.append(text)
        
        return results
    
    def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ):
        """
        Generate text with streaming (token-by-token output).
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            
        Yields:
            Generated tokens as they are produced
        """
        raise NotImplementedError("Streaming is not yet implemented for chunked encoding")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "model": self.llm.llm_engine.model_config.model,
            "tokenizer_size": len(self.tokenizer),
            "chunk_start_token_id": self.chunk_start_token_id,
            "chunk_end_token_id": self.chunk_end_token_id,
        }


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(
        description="Serve an LLM with chunked rotary encoding support"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Chunked LLM Server")
    print("=" * 80)
    
    # Initialize server
    config = ChunkedPromptConfig(
        chunk_start_marker="<Chunk>",
        chunk_end_marker="</Chunk>",
        validate_pairing=True,
    )
    
    server = ChunkedLLMServer(
        model_name=args.model,
        config=config,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
    print("\n" + "=" * 80)
    print("Server ready!")
    print("=" * 80)
    print(server.get_stats())
    print()
    
    if args.interactive:
        # Interactive mode
        print("\nInteractive Mode - Enter prompts (Ctrl+C to exit)")
        print("You can use <Chunk>...</Chunk> markers in your prompts")
        print("-" * 80)
        
        try:
            while True:
                print("\nEnter prompt (or 'quit' to exit):")
                prompt = input("> ")
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt.strip():
                    continue
                
                try:
                    results = server.generate([prompt], sampling_params)
                    print("\n" + "=" * 80)
                    print("Generated:")
                    print("=" * 80)
                    print(results[0])
                    print("=" * 80)
                except Exception as e:
                    print(f"Error: {e}")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
    
    else:
        # Example prompts
        example_prompts = [
            """Analyze these documents:
<Chunk>Document 1: Artificial Intelligence has revolutionized many industries.</Chunk>
<Chunk>Document 2: Machine Learning is a subset of AI focused on learning from data.</Chunk>
What is the relationship between the concepts in these documents?""",
            
            """Compare the following:
<Chunk>Classical computers use bits (0 or 1) for computation.</Chunk>
<Chunk>Quantum computers use qubits which can be in superposition.</Chunk>
<Chunk>Neuromorphic computers mimic biological neural networks.</Chunk>
Summarize the key differences.""",
        ]
        
        print("Running example prompts...\n")
        
        for i, prompt in enumerate(example_prompts, 1):
            print(f"\n{'=' * 80}")
            print(f"Example {i}")
            print(f"{'=' * 80}")
            print(f"Prompt:\n{prompt}")
            print(f"\n{'-' * 80}")
            
            results = server.generate([prompt], sampling_params)
            
            print(f"Generated:\n{results[0]}")
            print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
