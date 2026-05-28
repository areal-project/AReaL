# Agent Service — Claude Agent SDK

## Overview

This example demonstrates AReaL's Agent Service running the **Claude Agent SDK**
(`claude-agent-sdk`) as a scalable HTTP micro-service. It turns Claude's autonomous
agent capabilities — multi-turn conversations, tool use, file editing, web search — into
a production-deployable service with session management, load balancing, and dynamic
scaling.

**Why this matters**: Projects like
[claude-agent-acp](https://github.com/agentclientprotocol/claude-agent-acp) expose
Claude Agent SDK via custom protocols (ACP) for editor integration. AReaL takes a
different approach — it wraps Claude Agent SDK into standard HTTP micro-services with
session-affine routing, so you can **scale, orchestrate, and train** Claude agents using
AReaL's RL infrastructure.

```
Client → Gateway (HTTP) → Router → DataProxy (session state) → Worker (ClaudeSDKClient)
```

## Prerequisites

```bash
uv pip install claude-agent-sdk
export ANTHROPIC_API_KEY=sk-...
```

## Quick Start

```bash
python examples/agent_service/run_agent_service.py
```

The script creates a `LocalScheduler`, launches Guard workers, then forks Router →
Worker+DataProxy → Gateway. An interactive prompt lets you chat with the Claude agent.

### Options

### Send requests directly

```bash
curl -X POST http://localhost:<gateway-port>/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer areal-agent-admin" \
  -d '{
    "input": [{"type": "message", "content": "Explain RLHF in simple terms"}],
    "model": "claude-agent",
    "user": "my-session"
  }'
```

## Configuration

Claude Agent SDK settings are controlled via environment variables:

| Variable               | Default             | Description                 |
| ---------------------- | ------------------- | --------------------------- |
| `ANTHROPIC_API_KEY`    | (required)          | Anthropic API key           |
| `CLAUDE_MODEL`         | `claude-sonnet-4-6` | Model to use                |
| `CLAUDE_SYSTEM_PROMPT` | (none)              | Optional system prompt      |
| `CLAUDE_MAX_TURNS`     | `20`                | Max agentic turns per query |

## Architecture

The Worker maintains a **session-persistent `ClaudeSDKClient`** per session key. Unlike
stateless wrappers, the SDK's internal session retains the full conversation transcript
— no need to re-send history on each turn.

```
Turn 1: Client → Gateway → Router → DataProxy → Worker
         Worker: creates ClaudeSDKClient for session "abc"
         Claude Agent SDK runs autonomously (tool calls, file ops, etc.)
         Response streams back through the chain

Turn 2: Client → Gateway → Router (same DataProxy) → DataProxy → Worker
         Worker: reuses ClaudeSDKClient for session "abc"
         SDK remembers full context from Turn 1
```

## Programmatic Usage

```python
import os

from areal.api.cli_args import AgentConfig, SchedulingSpec
from areal.experimental.agent_service.controller import (
    AgentController,
)
from areal.infra.scheduler.local import LocalScheduler

scheduler = LocalScheduler(experiment_name="demo", trial_name="run0", gpu_devices=[])
ctrl = AgentController(
    config=AgentConfig(
        agent_cls_path="examples.agent_service.agent.ClaudeAgent",
        scheduling_spec=(
            SchedulingSpec(
                env_vars={"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]},
            ),
        ),
    ),
    scheduler=scheduler,
)
ctrl.initialize()
# ctrl.gateway_addr → "http://10.0.0.1:9005"
# ctrl.scale_up(2)   → add 2 more pairs
# ctrl.scale_down(1) → remove 1 pair (with graceful drain)
ctrl.destroy()
```

Use `AgentConfig.scheduling_spec[0].env_vars` to pass environment variables to all
forked agent-service child processes.

## Files

| File                   | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `agent.py`             | `ClaudeAgent` — session-persistent Claude Agent SDK wrapper |
| `run_agent_service.py` | Controller-based launcher + interactive conversation        |
