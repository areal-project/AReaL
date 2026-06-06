# LLM Test-time Scaling

This project investigates best practices for LLM test-time scaling. In LLM test-time scaling, there are two fundamental operations: reflection & aggregation. Reflection enables the LLM to improve its existing solution with self-evaluation or external feedbacks. Aggregation aims to select the best solution given a population of solutions.

## Project Structure

```
llm_test_time_scale/
├── src/
│   ├── llm_service/      # LLM service layer for model communication
│   ├── prompts/          # Prompt templates and system prompts
│   ├── evaluation/       # Evaluation pipeline (judges and executors)
│   ├── scaling/          # Test-time scaling functions (reflection & aggregation)
│   ├── benchmarks/       # Benchmark data loaders
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── config/               # Configuration files
├── data/                 # Benchmark data
└── examples/             # Example scripts
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

**Note**: The code executor uses the `functioncall` framework. Ensure the `functioncall` directory is in the `LLM_Test-time_Scaling` directory:

```
LLM_Test-time_Scaling/
├── functioncall/
├── src/
└── ...
```

## Benchmarks

The project evaluates methods on:
- **IMOBench-AnswerBench**: Math reasoning tasks
- **LiveCodeBench Pro**: Coding tasks
- **GPQA Diamond**: Scientific QA

## Run Experiments

After LLM services are launched, run experiments through the following scripts:
```
// reflection
python scripts/run_imobench_experiment.py
python scripts/run_lcb_pro_experiment.py

// aggregation. more under aggregation_scripts directory
python scripts/run_aggregation_experiment.py
python scripts/run_lcb_pro_aggregation_experiment.py
```

## Methods

### Reflection
- No feedback
- Self-evaluation
- Ground-truth evaluation
- Sample case execution (for coding)

### Aggregation
- LLM generate 1 given n solutions
- LLM select 1-out-of-n solutions
- LLM scoring
- Scoring with ground-truth evaluation
- Voting (for math)
- Pairwise comparison

### Architectural Patterns
- Reflection first, then Aggregation
- Aggregation at each Reflection Turn
