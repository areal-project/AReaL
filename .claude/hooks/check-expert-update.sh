#!/bin/bash
# Hook script to remind updating expert agents when related code changes
# Called by Claude Code PostToolUse hook

# Check if jq is available
if ! command -v jq &> /dev/null; then
    exit 0
fi

# Read JSON input from stdin
INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Define mappings: code path pattern -> expert agent file
check_expert_update() {
    local file="$1"
    local reminder_file=""
    local reminder_desc=""

    # Archon Engine / MoE related
    if [[ "$file" == *"areal/experimental/models/archon/"* ]] || \
       [[ "$file" == *"areal/experimental/engine/archon"* ]]; then
        reminder_file="archon-engine-expert.md"
        reminder_desc="Archon/MoE"
    fi

    # FSDP Engine related
    if [[ "$file" == *"areal/engine/fsdp_engine"* ]] || \
       [[ "$file" == *"areal/engine/fsdp_utils/"* ]]; then
        reminder_file="fsdp-engine-expert.md"
        reminder_desc="FSDP"
    fi

    # Megatron Engine related
    if [[ "$file" == *"areal/engine/megatron"* ]]; then
        reminder_file="megatron-engine-expert.md"
        reminder_desc="Megatron/PP"
    fi

    # Algorithm related (PPO, GRPO, workflows)
    if [[ "$file" == *"areal/trainer/ppo/"* ]] || \
       [[ "$file" == *"areal/workflow/"* ]] || \
       [[ "$file" == *"areal/reward/"* ]]; then
        reminder_file="algorithm-expert.md"
        reminder_desc="Algorithm/Workflow/Reward"
    fi

    # Output reminder if matched
    if [ -n "$reminder_file" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📝 Expert Update Reminder"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Modified: $file"
        echo "Consider updating: .claude/agents/$reminder_file ($reminder_desc)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
}

check_expert_update "$FILE_PATH"
