"""Test script for code verify service."""

import asyncio
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from evaluation.remote_lcb_pro_evaluator import RemoteLCBProEvaluator


async def test_remote_evaluator():
    """Test remote evaluator."""
    # Example service URL (replace with actual service URL)
    service_url = "http://localhost:8000"
    
    # Sample C++ code
    sample_code = """
#include <iostream>
using namespace std;
int main() {
    int a, b;
    cin >> a >> b;
    cout << a + b << endl;
    return 0;
}
"""
    
    problem_id = "1983A"  # Replace with actual problem ID
    
    print(f"Testing remote evaluator with service: {service_url}")
    print(f"Problem ID: {problem_id}")
    
    async with RemoteLCBProEvaluator(
        service_url=service_url,
        timeout=300,
    ) as evaluator:
        result = await evaluator.evaluate(
            problem="Add two numbers",
            solution=sample_code,
            problem_id=problem_id,
            language="cpp",
        )
        
        print(f"\nResults:")
        print(f"  Correct: {result.is_correct}")
        print(f"  Score: {result.score}")
        print(f"  Feedback: {result.feedback}")
        if result.details:
            print(f"  Passed: {result.details.get('passed', 'N/A')}/{result.details.get('total', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(test_remote_evaluator())
