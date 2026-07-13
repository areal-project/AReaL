# SPDX-License-Identifier: Apache-2.0

"""AReaL Agent Service — agent-level inference tier.

Exposes complete agent sessions (autonomous multi-step reasoning, tool use,
memory) via independent HTTP microservices: Gateway, Router, DataProxy,
and Worker.

Submodules
----------
- ``controller`` — :class:`AgentController` orchestrator
- ``gateway`` — public HTTP/WebSocket entry point
- ``router`` — session-affine routing
- ``data_proxy`` — stateful session proxy
- ``worker`` — stateless agent execution
- ``protocol`` — WebSocket frame types and helpers
"""

from .memory import (
    AsyncMemoryAgentCoordinator,
    MemoryAgentCoordinatorClosedError,
    MemoryAgentCoordinatorError,
    MemoryAgentSessionConflictError,
    MemoryAgentSessionPinV1,
    MemoryAgentTurnConflictError,
    MemoryAgentTurnV1,
)
from .memory_transport import (
    AREAL_MEMORY_METADATA_KEY,
    MEMORY_ASSIGNMENT_PIN_FIELD,
    MemoryAgentMetadataWireV1,
    MemoryAssignmentPinWireV1,
    MemoryPinTransportError,
    MemoryPinWireFormatError,
    MemorySessionPinCache,
    MemorySessionPinConflictError,
    ReservedMemoryMetadataError,
    parse_memory_assignment_pin_metadata,
)
from .types import (
    AgentRequest,
    AgentResponse,
    AgentRunnable,
    EventEmitter,
)

__all__ = [
    "AREAL_MEMORY_METADATA_KEY",
    "AgentRequest",
    "AgentResponse",
    "AgentRunnable",
    "AsyncMemoryAgentCoordinator",
    "EventEmitter",
    "MEMORY_ASSIGNMENT_PIN_FIELD",
    "MemoryAgentMetadataWireV1",
    "MemoryAgentCoordinatorClosedError",
    "MemoryAgentCoordinatorError",
    "MemoryAgentSessionConflictError",
    "MemoryAgentSessionPinV1",
    "MemoryAgentTurnConflictError",
    "MemoryAgentTurnV1",
    "MemoryAssignmentPinWireV1",
    "MemoryPinTransportError",
    "MemoryPinWireFormatError",
    "MemorySessionPinCache",
    "MemorySessionPinConflictError",
    "ReservedMemoryMetadataError",
    "parse_memory_assignment_pin_metadata",
]
