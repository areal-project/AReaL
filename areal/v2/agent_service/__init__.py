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
- ``memory_authorization`` — default-deny principal/session assignment grants
- ``memory_broker`` — host-owned grant admission and incarnation lifecycle
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
from .memory_authorization import (
    MemoryAssignmentGrantTargetV1,
    MemoryPrincipalV1,
    MemoryScopeActionV1,
    MemoryScopeAuthorizationConflictError,
    MemoryScopeAuthorizationDeniedError,
    MemoryScopeAuthorizationDisabledError,
    MemoryScopeAuthorizationError,
    MemoryScopeGrantAuthorizer,
    MemoryScopeGrantRequestV1,
    MemoryScopeGrantResolver,
    MemoryScopeGrantV1,
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)
from .memory_broker import (
    AuthorizedMemoryAgentBroker,
    AuthorizedMemorySessionV1,
    AuthorizedMemoryTurnV1,
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
    MemoryTurnCapability,
    MemoryTurnResultV1,
)

__all__ = [
    "AREAL_MEMORY_METADATA_KEY",
    "AgentRequest",
    "AgentResponse",
    "AgentRunnable",
    "AuthorizedMemoryAgentBroker",
    "AuthorizedMemorySessionV1",
    "AuthorizedMemoryTurnV1",
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
    "MemoryAssignmentGrantTargetV1",
    "MemoryAssignmentPinWireV1",
    "MemoryPinTransportError",
    "MemoryPinWireFormatError",
    "MemorySessionPinCache",
    "MemorySessionPinConflictError",
    "MemoryPrincipalV1",
    "MemoryScopeActionV1",
    "MemoryScopeAuthorizationConflictError",
    "MemoryScopeAuthorizationDeniedError",
    "MemoryScopeAuthorizationDisabledError",
    "MemoryScopeAuthorizationError",
    "MemoryScopeGrantAuthorizer",
    "MemoryScopeGrantRequestV1",
    "MemoryScopeGrantResolver",
    "MemoryScopeGrantV1",
    "MemorySessionIncarnationV1",
    "MemoryTurnCapability",
    "MemoryTurnResultV1",
    "MemoryWorkerAudienceV1",
    "ReservedMemoryMetadataError",
    "parse_memory_assignment_pin_metadata",
]
