#!/bin/bash
# Kill stalled HLE experiment and restart
echo "[$(date)] Killing stalled processes..."
kill -9 472017 472020 2>/dev/null
sleep 2

# Clean up Qdrant lock files
echo "[$(date)] Cleaning Qdrant locks..."
find results/hle/exp_hle_memrl_gemini3_20260424-224217/snapshot/ -name ".lock" -delete 2>/dev/null

# Verify processes are dead
remaining=$(ps aux | grep "srun.*yl-memrl-hle" | grep -v grep | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo "[$(date)] WARNING: $remaining processes still alive, force killing all..."
    ps aux | grep "srun.*yl-memrl-hle" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
    sleep 2
fi

# Restart
echo "[$(date)] Restarting experiment..."
cd /storage/openpsi/users/yl/agent-memory/MemRL
nohup bash scripts/run_memrl_srun.sh hle > logs/hle_gemini_resume_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "[$(date)] Restarted with PID $!"
