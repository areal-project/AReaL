#!/bin/bash
# vLLM watchdog: monitors health endpoint and restarts if dead
# Usage: source this file, then call run_with_watchdog <main_cmd>
# Requires: LLM_PORT, LLM_CMD, LLM_PID_FILE to be set

start_llm() {
    if [ -f "$LLM_PID_FILE" ]; then
        local old_pid=$(cat "$LLM_PID_FILE")
        kill "$old_pid" 2>/dev/null || true
        sleep 2
    fi
    eval $LLM_CMD &
    echo $! > "$LLM_PID_FILE"
    echo "[WATCHDOG] Started vLLM pid=$(cat $LLM_PID_FILE) at $(date)"
}

_watchdog_loop() {
    local main_pid=$1
    local fails=0
    while kill -0 "$main_pid" 2>/dev/null; do
        if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
            fails=0
        else
            fails=$((fails+1))
            echo "[WATCHDOG] health check failed ($fails/3) at $(date)"
            if [ "$fails" -ge 3 ]; then
                echo "[WATCHDOG] restarting vLLM at $(date)"
                start_llm
                local restarted=0
                for w in $(seq 1 300); do
                    if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
                        echo "[WATCHDOG] vLLM restarted successfully at $(date)"
                        restarted=1
                        break
                    fi
                    sleep 1
                done
                if [ "$restarted" -eq 1 ]; then
                    fails=0
                else
                    echo "[WATCHDOG] vLLM restart FAILED after 300s at $(date)"
                fi
            fi
        fi
        sleep 60
    done
    echo "[WATCHDOG] main process finished, exiting watchdog"
}

cleanup_vllm() {
    if [ -f "$LLM_PID_FILE" ]; then
        kill $(cat "$LLM_PID_FILE") 2>/dev/null || true
        rm -f "$LLM_PID_FILE"
    fi
}

run_with_watchdog() {
    # $@ is the main experiment command
    "$@" &
    local main_pid=$!

    _watchdog_loop "$main_pid" &
    local wd_pid=$!

    wait "$main_pid"
    local exit_code=$?
    echo "[INFO] Main process exited with code: $exit_code"

    kill "$wd_pid" 2>/dev/null || true
    wait "$wd_pid" 2>/dev/null || true
    cleanup_vllm

    return $exit_code
}
