# Synthetic Multiplication Dataset Generator

## Usage

Generate synthetic chain-of-thought multiplication examples:

```bash
# Save JSON format (for SFT, since trl does not allow a "prompt" field)
python generate_math.py -n 10000 \
  --dataset_dir mult-10k-par \
  --task mult \
  --create_val \
  --seed 42 \
  --val_seed 100 \
  --val_num_examples 1000 \
  --min_value 0 \
  --max_value 1000 \
  --min_len 5 \
  --max_len 8 \
  --save_format json \
  --parallel \
  --overwrite

# Save Parquet format (for RL, since verl requires a "prompt" field)
python generate_math.py -n 10000 \
  --dataset_dir mult-10k-par_pq \
  --task mult \
  --create_val \
  --seed 42 \
  --val_seed 100 \
  --val_num_examples 1000 \
  --min_value 0 \
  --max_value 1000 \
  --min_len 5 \
  --max_len 8 \
  --save_format parquet \
  --parallel \
  --overwrite
```

## Key Parameters

- `-n`: Number of training examples
- `--dataset_dir`: Output directory for datasets
- `--task`: Task type (use `mult` for multiplication)
- `--create_val`: Generate validation set
- `--val_num_examples`: Number of validation examples
- `--min_value/--max_value`: Range for integer values
- `--min_len/--max_len`: Number of integers to be multiplied
- `--save_format`: Output format (`json` or `parquet`)
- `--parallel`: Enable parallel CoT generation
- `--overwrite`: Overwrite existing files
