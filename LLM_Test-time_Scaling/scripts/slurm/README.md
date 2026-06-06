# SLURM Service Management for SGlang

This directory contains scripts for launching and managing SGlang model services on a SLURM cluster with automatic service discovery.

## Overview

The service management system provides:
- **Scalable deployment**: Launch multiple model services across cluster nodes
- **Automatic registration**: Services register their IP/port to a central registry
- **Load balancing**: Automatically distribute requests across all services
- **Service monitoring**: Health checks and status tracking
- **Easy integration**: Export service URLs for experiments

## Quick Start

### 1. Launch Multiple Services

```bash
# Launch 4 services on different nodes (ports 8000-8003)
bash scripts/slurm/launch_multiple_services.sh 4 gpt-oss-120b 8000

# Launch 8 services
bash scripts/slurm/launch_multiple_services.sh 8 gpt-oss-120b 9000
```

### 2. Check Service Status

```bash
# List all services
python scripts/slurm/service_manager.py list

# Check service health
python scripts/slurm/service_manager.py health

# View service URLs
python scripts/slurm/service_manager.py urls
```

### 3. Use Services in Experiments

```bash
# Export environment variables
python scripts/slurm/service_manager.py export -o env_vars.sh
source env_vars.sh

# Now run your experiment
python scripts/run_imobench_experiment.py
```

## Detailed Usage

### Launching Services

#### Single Service
```bash
# Submit a single service job
sbatch scripts/slurm/launch_sglang_service.sh \
    gpt-oss-120b \
    8000 \
    services.txt
```

#### Multiple Services
```bash
# Launch N services with automatic port assignment
bash scripts/slurm/launch_multiple_services.sh <NUM_NODES> <MODEL_PATH> <START_PORT>

# Examples:
bash scripts/slurm/launch_multiple_services.sh 16 gpt-oss-120b 8000
bash scripts/slurm/launch_multiple_services.sh 1 gpt-oss-20b 9000
```

### Service Manager Commands

#### List Services
```bash
# Show all services (running and stopped)
python scripts/slurm/service_manager.py list
```

#### Get Service URLs
```bash
# Print all running service URLs
python scripts/slurm/service_manager.py urls

# Example output:
# http://10.0.1.5:8000/v1
# http://10.0.1.6:8001/v1
# http://10.0.1.7:8002/v1
# http://10.0.1.8:8003/v1
```

#### Export Environment Variables
```bash
# Print export commands
python scripts/slurm/service_manager.py export

# Save to file and source
python scripts/slurm/service_manager.py export -o env_vars.sh
source env_vars.sh

```

#### Generate Config
```bash
# Get configuration snippet for experiments
python scripts/slurm/service_manager.py config
```

#### Check Health
```bash
# Check if services are responding
python scripts/slurm/service_manager.py health

# With custom timeout
python scripts/slurm/service_manager.py health --timeout 10
```

#### Cancel Services
```bash
# Cancel all running services (with confirmation)
python scripts/slurm/service_manager.py cancel

# Force cancel without confirmation
python scripts/slurm/service_manager.py cancel --force
```

## Integration with Experiments

### Method 1: Environment Variables (Recommended)
```bash
# Export service URLs to environment
python scripts/slurm/service_manager.py export -o env_vars.sh
source env_vars.sh

# Run experiment (automatically picks up OPENAI_API_BASE)
python scripts/run_imobench_experiment.py
```

### Method 2: Modify Experiment Script
```python
# In run_imobench_experiment.py
from scripts.slurm.service_manager import ServiceRegistry

# Load service URLs
registry = ServiceRegistry("services.txt")
API_BASE = registry.get_api_bases()

# Use in main()
async def main():
    MODEL = "openai/gpt-oss-120b"
    OUTPUT_DIR = "results/imobench_rollouts"
    API_KEY = os.getenv("OPENAI_API_KEY")
    # API_BASE is now set from registry
```

### Method 3: Direct Configuration
```bash
# Get the comma-separated URLs
URLS=$(python scripts/slurm/service_manager.py urls | tr '\n' ',')

# Set in script
export OPENAI_API_BASE="$URLS"
python scripts/run_imobench_experiment.py
```

## Service Registry Format

The `services.txt` file contains:
```
# Service Registry - 2024-01-15 10:30:00
# Format: JOB_ID|HOSTNAME|HOST_IP|PORT|MODEL_PATH|START_TIME|STATUS|END_TIME|EXIT_CODE
# =====================================
123456|node01|10.0.1.5|8000|gpt-oss-120b|2024-01-15 10:30:00|running
123457|node02|10.0.1.6|8001|gpt-oss-120b|2024-01-15 10:30:05|running
123458|node03|10.0.1.7|8002|gpt-oss-120b|2024-01-15 10:30:10|stopped|2024-01-15 12:00:00|0
```

## Monitoring

### SLURM Job Monitoring
```bash
# Check job status
squeue -u $USER

# View job output
tail -f logs/sglang_<JOB_ID>_<NODE>.out

# Check job errors
tail -f logs/sglang_<JOB_ID>_<NODE>.err
```

### Service Health Monitoring
```bash
# Continuous health check
watch -n 30 "python scripts/slurm/service_manager.py health"

# Test a specific service
curl http://<HOST_IP>:<PORT>/health
curl http://<HOST_IP>:<PORT>/v1/models
```

## Load Balancing

The system uses **random selection** load balancing:
- Each API request randomly selects one of the available service endpoints
- Configured in `litellm_service.py`
- Works automatically when multiple URLs are provided via comma separation

Example with 4 services:
```bash
export OPENAI_API_BASE="service_addr1,service_addr2,service_addr3,service_addr4"
```

Each request will randomly pick one endpoint, providing automatic load distribution.

## Troubleshooting

### Services Not Starting
```bash
# Check SLURM job logs
cat logs/sglang_<JOB_ID>_<NODE>.err

# Verify node has GPUs
srun --gres=gpu:8 nvidia-smi

# Test singularity container
singularity exec --nv /storage/openpsi/images/areal-latest.sif python3 --version
```

### IP Address Issues
```bash
# Check node network configuration
srun --nodes=1 --gres=gpu:8 hostname -I
srun --nodes=1 --gres=gpu:8 ip addr show

# Test connectivity between nodes
ping <HOST_IP>
```

### Service Not Responding
```bash
# Check if port is open
nc -zv <HOST_IP> <PORT>

# Test service endpoint
curl http://<HOST_IP>:<PORT>/health

# Check if process is running
srun --jobid=<JOB_ID> ps aux | grep sglang
```

### Port Already in Use
```bash
# Check for port conflicts
netstat -tulpn | grep <PORT>

# Use different starting port
bash scripts/slurm/launch_multiple_services.sh 4 <MODEL> 9000
```

## Best Practices

1. **Start with a few nodes**: Test with 2-4 nodes before scaling to many
2. **Monitor health regularly**: Use health checks to detect issues early
3. **Clean up stopped services**: Periodically remove old entries from registry
4. **Use unique ports**: Avoid port conflicts by incrementing start port
5. **Check logs**: Always review logs when services fail to start
6. **Test connectivity**: Verify network access between compute nodes
7. **Resource limits**: Ensure cluster has enough GPUs for all services

## Advanced Usage

### Custom Model Paths
```bash
# Launch different models on different nodes
sbatch scripts/slurm/launch_sglang_service.sh gpt-oss-120b 8000
sbatch scripts/slurm/launch_sglang_service.sh gpt-oss-20b 8001
```

### Multiple Registries
```bash
# Use separate registries for different experiments
bash scripts/slurm/launch_multiple_services.sh 4 <MODEL> 8000 services_exp1.txt
bash scripts/slurm/launch_multiple_services.sh 4 <MODEL> 9000 services_exp2.txt

# List from specific registry
python scripts/slurm/service_manager.py --registry services_exp1.txt list
```

### Cleanup
```bash
# Cancel all jobs
python scripts/slurm/service_manager.py cancel --force

# Remove old log files
rm logs/sglang_*.{out,err}

# Reset service registry
rm services.txt
```

## Files

- `launch_sglang_service.sh`: SLURM batch script for single service
- `launch_multiple_services.sh`: Helper to launch multiple services
- `service_manager.py`: Python utility for managing services
- `services.txt`: Service registry (auto-generated)
- `logs/`: SLURM job output and error logs
- `slurm_jobs/`: Job ID tracking files

## Requirements

- SLURM cluster with Singularity
- Compute nodes with GPUs
- SGlang installed in singularity container
- Python 3.7+ for service manager
- Network connectivity between compute nodes
