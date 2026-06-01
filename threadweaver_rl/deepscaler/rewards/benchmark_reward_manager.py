#!/usr/bin/env python3
"""
Benchmark script for RewardManager performance testing.
Tests the reward manager with 1000 samples and measures execution time.
"""

import time
import torch
import numpy as np
from tensordict import TensorDict
from typing import List
import statistics
import sys
import os
from transformers import AutoTokenizer

# Add the project root to path
sys.path.insert(0, '.')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../verl')

from verl.protocol import DataProto
from verl.trainer.reward_manager import RewardManager
from deepscaler.rewards.reward_types import RewardConfig


class MockTokenizer:
    """Mock tokenizer for benchmarking purposes."""

    def __init__(self):
        self.pad_token_id = 0

    def decode(self, token_ids):
        """Mock decode function that returns a realistic math problem response."""
        # Simulate different types of responses for variety
        responses = [
            "What is 2+2? I need to add 2 and 2. 2 + 2 = 4. The answer is \\boxed{4}.",
            "Solve for x: 3x + 5 = 14. <think> I need to isolate x. 3x = 14 - 5 = 9. So x = 9/3 = 3. </think> The answer is \\boxed{3}.",
            "What is the area of a circle with radius 5? <think> Area = πr². With r=5, Area = π(5)² = 25π. </think> The answer is \\boxed{25\\pi}.",
            "Find the derivative of x². <think> The derivative of x² is 2x using the power rule. </think> The answer is \\boxed{2x}.",
            "What is 10! (10 factorial)? <think> 10! = 10×9×8×7×6×5×4×3×2×1 = 3628800. </think> The answer is \\boxed{3628800}."
        ]
        # Use hash of token_ids to get consistent but varied responses
        idx = hash(str(token_ids.tolist() if hasattr(token_ids, 'tolist') else token_ids)) % len(responses)

        return "Text " * 10000 + responses[idx]


def create_mock_data_proto(batch_size: int, seq_len: int=512, prompt_len: int=256) -> DataProto:
    """Create a mock DataProto with realistic data for benchmarking."""

    # Create mock tensor data
    response_len = seq_len - prompt_len

    tensors = {}
    non_tensors = {}

    # Create batch data with correct data types
    prompts = torch.randint(1, 1000, (batch_size, prompt_len), dtype=torch.long)
    responses = torch.randint(1000, 2000, (batch_size, response_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    tensors['prompts'] = prompts
    tensors['responses'] = responses
    tensors['attention_mask'] = attention_mask

    # Create non-tensor data (ground truth answers)
    ground_truths = ['4', '3', '25π', '2x', '3628800'] * (batch_size // 5 + 1)
    ground_truths = ground_truths[:batch_size]

    data_sources = ['gsm8k'] * batch_size

    # Structure the reward model data correctly
    reward_model_data = []
    for i in range(batch_size):
        reward_model_data.append({'ground_truth': ground_truths[i]})

    non_tensors['reward_model'] = np.array(reward_model_data, dtype=object)
    non_tensors['data_source'] = np.array(data_sources, dtype=object)

    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def benchmark_reward_manager(num_samples: int = 1000, num_examine: int = 5, batch_sizes: List[int] = None) -> dict:
    """
    Benchmark the RewardManager with different batch sizes.

    Args:
        num_samples: Total number of samples to process
        num_examine: Number of samples to print during processing
        batch_sizes: List of batch sizes to test

    Returns:
        Dictionary containing benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 200, 500, 1000]

    print(f"🚀 Starting RewardManager Benchmark")
    print(f"📊 Total samples: {num_samples}")
    print(f"🔍 Batch sizes to test: {batch_sizes}")
    print("=" * 60)

    # Initialize components
    # tokenizer = MockTokenizer()
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    config = RewardConfig(require_think_end=False)

    results = {
        'total_samples': num_samples,
        'batch_results': {},
        'summary': {}
    }

    for batch_size in batch_sizes:
        if batch_size > num_samples:
            continue

        print(f"\n🧪 Testing batch size: {batch_size}")

        # Calculate number of batches needed
        num_batches = (num_samples + batch_size - 1) // batch_size
        actual_samples = min(num_samples, batch_size * num_batches)

        print(f"   Batches: {num_batches}, Actual samples: {actual_samples}")

        # Initialize reward manager
        reward_manager = RewardManager(tokenizer=tokenizer, num_examine=num_examine, config=config)

        batch_times = []
        total_start_time = time.time()

        processed_samples = 0

        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - processed_samples)
            if current_batch_size <= 0:
                break

            # Create mock data for this batch
            data = create_mock_data_proto(current_batch_size, prompt_len=1024, seq_len=16384)

            # Time the reward manager execution
            batch_start_time = time.time()

            try:
                # Call the reward manager
                rewards = reward_manager(data, return_dict=True)

                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)

                processed_samples += current_batch_size

                # Validate output
                assert 'reward_tensor' in rewards
                assert rewards['reward_tensor'].shape[0] == current_batch_size

                print(f"   ✅ Batch {batch_idx + 1}/{num_batches}: {batch_time:.3f}s ({current_batch_size} samples)")

            except Exception as e:
                print(f"   ❌ Batch {batch_idx + 1} failed: {e}")
                batch_times.append(float('inf'))
                break

        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        # Calculate statistics
        valid_times = [t for t in batch_times if t != float('inf')]

        if valid_times:
            avg_batch_time = statistics.mean(valid_times)
            median_batch_time = statistics.median(valid_times)
            min_batch_time = min(valid_times)
            max_batch_time = max(valid_times)
            samples_per_second = processed_samples / total_time
            time_per_sample = total_time / processed_samples if processed_samples > 0 else float('inf')
        else:
            avg_batch_time = median_batch_time = min_batch_time = max_batch_time = float('inf')
            samples_per_second = 0
            time_per_sample = float('inf')

        # Store results
        batch_result = {
            'batch_size': batch_size,
            'num_batches': len(valid_times),
            'processed_samples': processed_samples,
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'median_batch_time': median_batch_time,
            'min_batch_time': min_batch_time,
            'max_batch_time': max_batch_time,
            'samples_per_second': samples_per_second,
            'time_per_sample': time_per_sample,
            'success_rate': len(valid_times) / len(batch_times) if batch_times else 0
        }

        results['batch_results'][batch_size] = batch_result

        # Print summary for this batch size
        print(f"   📈 Results:")
        print(f"      Total time: {total_time:.3f}s")
        print(f"      Samples/sec: {samples_per_second:.2f}")
        print(f"      Time/sample: {time_per_sample:.4f}s")
        print(f"      Avg batch time: {avg_batch_time:.3f}s")
        print(f"      Success rate: {batch_result['success_rate']:.1%}")

    # Generate summary
    print("\n" + "=" * 60)
    print("📋 BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"{'Batch Size':<12} {'Samples/sec':<12} {'Time/sample':<14} {'Total Time':<12} {'Success Rate':<12}")
    print("-" * 70)

    best_throughput = 0
    best_batch_size = None

    for batch_size in sorted(results['batch_results'].keys()):
        result = results['batch_results'][batch_size]
        if result['samples_per_second'] > best_throughput:
            best_throughput = result['samples_per_second']
            best_batch_size = batch_size

        print(f"{batch_size:<12} {result['samples_per_second']:<12.2f} {result['time_per_sample']:<14.4f} {result['total_time']:<12.3f} {result['success_rate']:<12.1%}")

    results['summary'] = {
        'best_batch_size': best_batch_size,
        'best_throughput': best_throughput,
        'total_time_all_tests': sum(r['total_time'] for r in results['batch_results'].values())
    }

    print(f"\n🏆 Best performance: Batch size {best_batch_size} with {best_throughput:.2f} samples/sec")
    print(f"⏱️  Total benchmark time: {results['summary']['total_time_all_tests']:.2f}s")

    return results


def main():
    """Main function to run the benchmark."""
    print("RewardManager Performance Benchmark")
    print("===================================")

    # Run the benchmark
    try:
        # n_samples = 10
        # n_samples = 128
        n_samples = 2048

        results = benchmark_reward_manager(
            num_samples=n_samples,
            num_examine=0,  # Reduce output verbosity
            batch_sizes=[n_samples],
        )

        print("\n✅ Benchmark completed successfully!")

        # Save results to file
        # import json
        # results_file = "reward_manager_benchmark_results.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # results_json = convert_numpy_types(results)

        # with open(results_file, 'w') as f:
        #     json.dump(results_json, f, indent=2)

        # print(f"📄 Results saved to: {results_file}")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
