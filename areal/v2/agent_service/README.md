# Agent Service

## Overview

The Agent Service provides **agent-level** capabilities on top of AReaL's model-level
proxy. It exposes complete agent sessions — multi-turn conversations with tool use,
memory, and pluggable agent frameworks — via independent HTTP microservices.

## Architecture

The Agent Service consists of four independent HTTP services that communicate via REST:

```
Client (HTTP/WS)
    │
    ▼
┌──────────┐  POST /route   ┌──────────┐
│ Gateway  │ ──────────────▶ │ Router   │
│          │ ◀────────────── │          │
└──────────┘  DataProxy addr └──────────┘
    │
    │ POST /session/{key}/turn
    ▼
┌──────────┐
│ DataProxy│
│ (history)│  POST /run   ┌──────────┐
│          │ ────────────▶│ Worker   │
└──────────┘              │ (agent)  │
                          └──────────┘
```

### Components

**Gateway** — Public entry point. Accepts WebSocket connections (Gateway protocol) and
HTTP requests via two bridges: the OpenResponses bridge (`POST /v1/responses`) and the
OpenAI chat-completions bridge (`POST /v1/chat/completions`). Routes to the appropriate
DataProxy via the Router.

**Router** — Session-affine routing service. DataProxy instances register at startup.
The Router assigns new sessions round-robin and maintains session → DataProxy affinity.

**DataProxy** — Stateful session proxy, paired 1:1 with a Worker. Manages per-session
conversation history. On each turn: reads history → constructs `AgentRequest` (with
history) → forwards to Worker → appends messages to history → returns response.

The first admitted turn also fixes the DataProxy session's security mode. A session is
either ordinary (no assignment pin) or Memory-bound (one immutable pin); it cannot be
upgraded, downgraded, or rebound in place. In particular, a later pin is rejected before
it can mutate the pin cache, so DataProxy history and routing state cannot cross the
Memory authorization boundary in place. The caller must drain and close the session
before opening another mode.

This mode is process-local defense in depth, not proof that Worker/plugin state was
erased. A stateful Agent's optional `close_session` hook must cooperate, and a future
authorized Worker runtime must independently bind its principal, pin, and broker-issued
incarnation. DataProxy restart or successful close alone is not a security assertion
about arbitrary plugin state.

**Worker** — Stateless agent execution server. Loads an `AgentRunnable` implementation
at startup. Each `POST /run` request is a single turn — the agent receives the full
conversation history in the request and returns a response. The Worker has no session
state.

### Internal hop authentication

Controller-managed deployments use a different random credential for every
DataProxy→Worker pair. The DataProxy creates a fresh Bearer header for both `/run` and
`/sessions/close`; it never forwards the Gateway request's Authorization header. The
pair credential is distinct from both the external admin key and the Gateway→DataProxy
Memory-control key, and it is not distributed to Gateway or Router. At startup, a
configured DataProxy calls the authenticated `/internal/auth-check` route and requires
its exact typed receipt; a wrong key or a standalone Worker therefore fails startup
instead of silently weakening the hop. `/health` remains unauthenticated for ordinary
readiness probes.

An empty Worker-hop key preserves standalone compatibility. Memory-control transport
cannot be enabled on a DataProxy in that mode. Internal-hop authentication verifies
possession of the pair key; it is **not** a principal/session grant for any
`MemoryScope`.

When a DataProxy has a Memory-control key, its session-history route and both its fixed
and deprecated session-close routes require that key before session-key validation or
state lookup. A history read discloses conversation content, while close destroys the
incarnation's history and Memory pin; accepting either operation from an unauthenticated
caller would expose or reset a privileged lifecycle. The public Gateway close route
still authenticates the external admin key, then replaces it with the dedicated
Memory-control credential for Gateway→DataProxy; it never forwards the external
credential. A configured `DataProxyClient` automatically carries the dedicated key on
history and both close routes. DataProxies with Memory control disabled retain anonymous
history and close for standalone compatibility. During a rolling upgrade, upgrade these
callers before enabling state authentication on the DataProxy; an old caller receives
`401` from a new Memory-capable DataProxy rather than silently weakening the boundary.

The Controller currently supplies pair keys through child-process arguments. This
protects against unauthenticated network callers and accidental cross-pair routing, not
arbitrary code running under the same OS identity: such a process may inspect sibling
process arguments. A hostile-plugin deployment needs separate OS identities or process
namespaces plus a protected secret channel (for example UDS permissions or mTLS).

### Canonical session identity

A session key is an identifier, not arbitrary display text. Because turn and history
APIs currently place it in one URL path segment, every public and internal boundary
accepts only 1–256 ASCII characters from `A-Z`, `a-z`, `0-9`, `.`, `_`, `~`, `:`, and
`-`; the complete keys `.` and `..` are also rejected. Validation never trims,
normalizes, percent-decodes, or silently rewrites a key. In particular, `/`, `\`, `?`,
`#`, `%`, whitespace, control characters, and Unicode cannot enter Router affinity,
DataProxy state, or Worker execution.

OpenAI `model` and `user` fields are business data and may still contain paths or
Unicode. The bridges preserve the readable `agent:model:user` / `chat:model:user` form
when its components are independently safe and unambiguous; otherwise they derive a
deterministic, domain-separated SHA-256 key. Thus common model IDs such as `org/model`
remain supported without turning their slash into routing syntax.

Session close uses the fixed `POST /sessions/close` endpoint with `session_key` in a
JSON body at Gateway→DataProxy and DataProxy→Worker hops. The Worker returns the exact
key digest in its close receipt, and the DataProxy clears history and Memory pin state
only when that receipt matches. This prevents repeated URL decoding from making a turn
and its close target different agent sessions. The old path-shaped close endpoints are
deprecated compatibility shims. New callers use the fixed endpoint first and fall back
to a legacy path only on `404`/`405`, after validating the key, so rolling upgrades do
not strand a closing session. Authentication or lifecycle failures never trigger that
downgrade. A legacy Worker's status-only close receipt is accepted only on this explicit
compatibility path; same-version fixed endpoints always require the exact key digest.

### Memory authorization contract

Internal hop identity is deliberately separate from Memory data authority. The
server-side `memory_authorization` contract binds an authenticated principal to all of
the following for one action:

- a fresh session-incarnation ID, so closing and reopening the same text session key
  cannot replay an old decision;
- a fresh Worker-audience ID, so a decision for one pair or process incarnation cannot
  be reused by another;
- the full assignment pin, not only its `MemoryScope`; and
- exactly `pin_assignment` or `expose_memory`, never a wildcard.

Both random IDs are non-secret replay domains generated by the host. They must not be
copied from request JSON, derived from network addresses, or derived from hop secrets. A
grant is a canonical audit record, not a bearer credential or cached lease: the trusted
resolver must return an active exact grant for every authorization operation. A
successful resolution admits only that operation. Later revocation blocks new
admissions; it does not roll back a consumer side effect that was already admitted.

`InMemoryMemoryScopeGrantStore` is the process-local reference resolver and control
store. It linearizes create, revoke, and resolve under one lock and publishes an
irreversible, content-addressed revocation tombstone. One canonical request may create
only one grant for the lifetime of that store: expiry or revocation cannot be bypassed
with a fresh idempotency key. A deliberately new authorization lifetime therefore needs
a fresh host-minted session incarnation or Worker audience. Future renewable grants must
add an explicit generation/supersession lineage instead of silently reusing the same
request identity.

The reference store scopes control addresses and idempotency keys by `MemoryScope`,
returns detached audit records, and treats IDs and hashes as non-authoritative pointers.
A new grant must already be active at its commit point, and its later half-open validity
interval is `valid_from <= now < valid_until`; a backwards clock observation fails
closed. Scheduled grants are intentionally deferred until explicit generation semantics
exist.

The store is non-durable and has no HTTP/admin surface, so process restart loses all
grants rather than restoring authority. A durable backend must atomically restore both
grants and their tombstones while preserving exact-request and linearization semantics;
an incomplete recovery must fail closed.

The `AuthorizedMemoryAgentBroker` is the default-off host seam that enforces this
contract around the async coordinator. It mints its own Worker audience and session
incarnations, checks `pin_assignment` before the release-control lookup, and checks
`expose_memory` before **every** capability invocation—including retries that the
coordinator could otherwise answer from a completed operation. Synchronous resolver work
runs through the coordinator's bounded executor rather than on the Worker event loop.

There are three distinct lifecycle points:

```text
grant resolution pending → admitted coordinator operation → consumer side effect
```

Cancellation or close during the first stage cannot start a later coordinator action.
Close waits for the second stage, and revocation after admission does not attempt to
undo the third. Closing and reopening the same textual session creates a new
incarnation, while replacing the broker creates a new Worker audience; old handles and
grant decisions cannot cross either boundary. Resolver request objects and coordinator
pins/turns use detached snapshots, while public handles are checked against private
scalar snapshots, so mutation of one alias cannot silently rewrite another authority
dimension.

The authorizer is default-disabled when no resolver is configured. Current HTTP ingress
does not establish an end-user principal, and the Worker app still does not turn a pin
envelope into a Memory capability. The broker is constructed and called only by trusted
host code, so adding it enables no HTTP Memory access by itself. In particular, the
shared admin key, either internal hop key, caller-chosen session key, pin scope, and
inference `sk-sess-*` handle are not principal proof.

`AuthorizedMemoryWorkerRuntime` adds the next, still default-off ownership boundary. It
takes exclusive lifecycle ownership of one broker and reserves each textual session key
for exactly one trusted `MemoryPrincipalV1`. Its public descriptor contains only the
broker-minted Worker audience and session incarnation; it contains neither the principal
nor the private broker handle, and an equal reconstructed descriptor has no local
authority. A cancelled reservation cannot release the broker's principal binding, while
close must finish before the same key can receive a fresh incarnation. This is only
process-local Worker state. A trusted host may explicitly bind a complete
`MemoryAgentSessionPinV1` and an exact runtime-issued reservation to an `AgentRequest`;
request metadata is ignored as authority. The returned `MemoryWorkerTurnLease` owns only
the exact request's capability lifetime and exposes neither the request, private broker
session, nor authorized turn. Its single-use `run_agent(...)` closes a structured turn
before returning, closes on Agent failure or cancellation, and transfers a lazy
`StreamResponse` to a cleanup iterator that keeps Memory valid through body EOF,
failure, cancellation, or explicit close. The source iterator closes before the
capability, so its `finally` block may still use the turn's Memory. The lease snapshots
the capability-bound `session_key` and `run_id` and rejects mutation before or during
the Agent call. Repeating a `run_id` reuses the broker trajectory but creates a fresh
independent capability lifetime.

Session retirement also has a process-local exact result. A reservation exposes a
detached `MemoryWorkerSessionIdentityV1` containing the broker-minted session
incarnation and Worker audience; this description is not a bearer capability and cannot
pin, run, or expose Memory. Closing the exact runtime-issued reservation returns a
`MemoryWorkerSessionCloseReceiptV1`. The runtime retains recent private retirement
tombstones in a bounded LRU, so a lost-response retry of A returns the same `closed`
result even after the text key has been reopened as B. It never resolves that retry
through the reusable key or repeats A's destructive cleanup. After an old tombstone is
evicted, a conditional identity retry safely degrades to `not_current`; it still cannot
touch B. The evicted process-local object handle is no longer replayable and is rejected
as stale. The capacity defaults to 4096 and is configurable at runtime construction. An
authenticated future host adapter may use `close_session_if_current(identity)`: an
unknown or mismatched identity returns `not_current` without revealing or modifying the
current incarnation, while an exact current or known-retired identity returns `closed`.
A damaged private state fails closed rather than being reported as absent. If a broker
ever repeats a still-retained incarnation identity, the runtime quarantines that text
key until shutdown instead of publishing an ambiguous successor.

A `MemoryWorkerSessionCloseReceiptV1` with outcome `closed` means only that the
Worker-owned **Memory runtime** completed its broker cleanup; `not_current` makes no
cleanup claim. It does not prove that an Agent plugin's key-indexed state was cleaned.

`AuthorizedMemoryAgentWorkerHost` adds that complete, still default-off ownership
boundary. Trusted host code transfers one Agent and one broker to it. The Agent must
provide `close_session(session_key)`; the host does not weaken `AgentRunnable` for
ordinary Workers. For every admitted turn, the host keeps an outer lease until a
structured response returns or a streaming body reaches EOF, fails, or is explicitly
closed. When an exact session close is requested, the host first waits for all such
outer leases to drain. It then closes the exact Memory reservation, invokes the Agent
hook, and only after both complete atomically publishes a distinct
`MemoryAgentWorkerSessionCloseReceiptV1` tombstone with outcome `closed`. The reusable
text key cannot be reopened between these stages, so A's delayed key-only Agent hook
cannot delete a new B. Shutdown also closes abandoned response bodies and cancels
in-flight Agent calls before it drains sessions.

The outer host owns its close task and recent full-host retirement LRU. Caller
cancellation cannot cancel publication, a lost-response retry replays A's full receipt,
and an evicted or mismatched identity returns `not_current` without touching B. If the
legacy Agent hook fails after it starts, the host cannot know whether its side effects
happened. It therefore quarantines the key for the rest of that host lifetime and
replays the same failure instead of repeating the hook. Shutdown is the only remaining
containment path: it drains the Memory runtime and calls the optional
`close_all_sessions()` hook when available, but does not retry the uncertain per-key
hook. Any shutdown error deliberately keeps the process-wide Agent ownership claim, so
uncertain plugin state cannot be attached to another exact host. A fully successful
shutdown releases the claim and permits deliberate reuse.

Both retirement ledgers are process-local and non-durable: LRU eviction discards only
old replay detail, while successful shutdown discards the caches together with the fresh
Worker audience. Neither receipt is currently consumed or produced by an HTTP route.

This remains a **default-off host adapter**, not Worker HTTP activation. No Worker route
constructs either ownership layer, no DataProxy envelope carries a reservation, and no
HTTP field is accepted as a principal. The lower runtime still requires its caller to
close an abandoned stream explicitly; the outer host additionally tracks published
bodies so its own shutdown can close them. HTTP integration still requires an atomic,
versioned open/run/close identity protocol and a trusted principal resolver. Passing a
broker to the runtime, or a broker and Agent to the outer host, is an exclusive
ownership transfer: trusted code must not retain a second execution or cleanup path.

`session_lifecycle_transport` defines the next default-off boundary: strict V1 JSON
values for a future DataProxy→Worker protocol. The `exact_session_lifecycle_v1`
capability is deliberately indivisible. Advertising it commits one authenticated Worker
pair to an idempotent open handshake, exact identity on every stateful run, and exact
close; it must never mean “exact close plus legacy run.” An exact client must not fall
back to a key-only route after selecting the capability.

Open sends a canonical reusable key, a DataProxy-owned `aopen_*` idempotency key, and
the expected Worker audience from capability negotiation. A future adapter must compare
that audience before reserving any state. For one Worker audience, the open ledger key
is `(worker_audience_id, open_request_id)` and its immutable binding is
`(trusted principal, session_key, expected_worker_audience_id)`. The same logical retry
must reuse the same request id. Reusing it with any changed binding is a conflict that
must not reveal the old identity; a new logical open needs a new id.

Before reserve, the adapter must capacity-check an unseen id and publish a pending
entry. Concurrent calls with the same binding join one owned, cancellation-shielded
task. Reserve and publication of its receipt form one cancellation-safe commit:
completed or unknown-effect entries remain replayable for the complete Worker-audience
lifetime and must not be LRU-evicted. A full ledger rejects only unseen ids before
reserve; an existing retry still replays while full. These rules ensure a lost response
replays A instead of creating B. The receipt echoes the request id and returns the
Worker-minted tuple of key, session incarnation, and audience. Run carries that complete
identity beside an immutable snapshot of the turn; the key is not duplicated inside the
turn. Close carries the same identity and returns either `closed` or `not_current`
through explicitly named full Agent-Worker receipt types. Only a matching `closed`
receipt proves full-host retirement. `not_current` is not cleanup success, must not
clear the DataProxy tombstone, and reveals no successor identity.

Capability, identity, open, run, and close values use exact field sets and integer
schema version 1; booleans, unknown fields, unsafe keys, malformed identity prefixes,
duplicate capability tokens, non-finite JSON, and non-JSON history or metadata are
rejected. Valid history and metadata within explicit depth, node, UTF-8 string-byte, and
16 MiB encoded payload budgets are canonicalized into private immutable snapshots before
later reconstruction. The paired canonical encoder applies the final 16 MiB limit to the
whole body after JSON escaping, so every accepted run request can pass the same module's
raw decoder. The raw decoder limits already-collected bytes before JSON materialization
and rejects invalid UTF-8, excessive parser nesting, non-finite numbers, malformed JSON,
and duplicate object names. Because it accepts `bytes`, it cannot by itself bound an
HTTP framework's earlier body read: a future adapter must stream at most `limit + 1`
bytes and reject overflow before assembling the body, rather than first calling an
unbounded `request.body()`.

These wire identities remain non-secret descriptions and grant no authority. The module
registers no HTTP route and performs no principal resolution. A future adapter must
authenticate the pair hop before raw decoding, bind open to a trusted principal, verify
that every returned request id, audience, and identity exactly matches its cached
expectation, use the paired canonical encoder for every outgoing lifecycle body, and
activate open/run/close together rather than exposing any partial protocol.

## Agent Protocol

Any class that satisfies the `AgentRunnable` protocol can run on the Worker:

```python
@runtime_checkable
class AgentRunnable(Protocol):
    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse: ...
```

### AgentRequest

```python
@dataclass
class AgentRequest:
    message: str                              # Current user message
    session_key: str                          # Session identifier
    run_id: str                               # Unique run identifier
    history: list[dict[str, str]]             # Prior conversation turns
    queue_mode: QueueMode = QueueMode.COLLECT
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def memory(self) -> MemoryTurnCapability | None: ...
```

`memory` is deliberately narrower than a store or retriever handle. It is already bound
to one scope, assignment, session, and turn; the agent can submit an idempotent
operation with query/history bytes but cannot select another scope, write an
acknowledgement, or manufacture an exposure. A successful call returns the configured
trusted consumer's output plus an exposure ID/hash that evaluation code must join back
to the Memory ledger. The output remains opaque: the exposure proves that the registered
consumer acknowledged exact delivery bytes, not that arbitrary output content is
correct, that an agent used it, or that it improved a later answer.

`memory` is a process-local, read-only property rather than a dataclass field. The
Worker host binds it after constructing the request, while `asdict(request)` and other
field-based wire serializers retain the pre-Memory schema. Runtime capability identity
must not be inferred from `AgentRequest` equality or used as an authorization/cache key.

The capability contract is currently an in-process, disabled-by-default seam. Trusted
host code can use `bind_authorized_memory_turn_capability(...)` only with a turn issued
by `AuthorizedMemoryAgentBroker`; each public `expose_memory(...)` call resolves a new
exact grant before it may reach the coordinator. The older bare-coordinator constructor
remains available for compatibility and tests, but is not the production authorization
path.

The Worker HTTP app does **not** yet turn `metadata["areal_memory"]` into a capability.
Wire activation still requires a trusted HTTP principal source in addition to the
authenticated DataProxy→Worker hop; an assignment pin alone is not authorization.
Because agent plugins share the Worker Python process, this interface prevents
accidental authority leakage rather than sandboxing malicious code. Adversarial plugins
require an out-of-process Memory broker.

Before wire activation, the Worker integration must also define host-controlled
query/history derivation (or explicitly audit agent-selected values), per-turn operation
and byte quotas, and a strict DTO for each registered consumer's output. Causal
evaluation must join **all** exposures recorded for the turn; an agent-returned pointer
is convenient indexing, not the source of truth.

### AgentResponse

```python
@dataclass
class AgentResponse:
    summary: str = ""                         # Agent reply text
    metadata: dict[str, Any] = field(default_factory=dict)
```

### EventEmitter

```python
class EventEmitter(Protocol):
    async def emit_delta(self, text: str) -> None: ...
    async def emit_tool_call(self, name: str, args: str) -> None: ...
    async def emit_tool_result(self, name: str, result: str) -> None: ...
```

## HTTP APIs

### Router

| Endpoint          | Method | Description                 |
| ----------------- | ------ | --------------------------- |
| `/health`         | GET    | Health check                |
| `/register`       | POST   | Register a DataProxy        |
| `/unregister`     | POST   | Unregister a DataProxy      |
| `/route`          | POST   | Get DataProxy for a session |
| `/remove_session` | POST   | Remove session affinity     |

### DataProxy

| Endpoint                 | Method | Description                                 |
| ------------------------ | ------ | ------------------------------------------- |
| `/health`                | GET    | Health check                                |
| `/session/{key}/turn`    | POST   | Send a message (turn)                       |
| `/sessions/close`        | POST   | Close session (JSON key)                    |
| `/session/{key}/close`   | POST   | Deprecated close shim                       |
| `/session/{key}/history` | GET    | Get history (internal-auth when configured) |

### Worker

| Endpoint               | Method | Description                    |
| ---------------------- | ------ | ------------------------------ |
| `/health`              | GET    | Health check                   |
| `/run`                 | POST   | Execute one agent turn         |
| `/sessions/close`      | POST   | Close session (JSON key)       |
| `/session/{key}/close` | POST   | Deprecated close shim          |
| `/internal/auth-check` | GET    | Verify pair-hop authentication |

### Gateway

| Endpoint               | Method | Description                    |
| ---------------------- | ------ | ------------------------------ |
| `/health`              | GET    | Health check                   |
| `/ws`                  | WS     | Gateway WebSocket protocol     |
| `/sessions/close`      | POST   | Close a session                |
| `/v1/responses`        | POST   | OpenResponses HTTP bridge      |
| `/v1/chat/completions` | POST   | OpenAI chat-completions bridge |

## Multi-turn Conversation Flow

```
Turn 1:
  Client → Gateway → Router (route session) → DataProxy
    DataProxy: history = []
    DataProxy → Worker: POST /run {message, history: []}
    Worker → Agent: run(request) → AgentResponse
    DataProxy: history = [user_msg, assistant_msg]
    DataProxy → Gateway → Client

Turn 2:
  Client → Gateway → Router (same DataProxy) → DataProxy
    DataProxy: history = [user_msg_1, assistant_msg_1]
    DataProxy → Worker: POST /run {message, history: [user_msg_1, assistant_msg_1]}
    Worker → Agent: run(request) → AgentResponse
    DataProxy: history = [..., user_msg_2, assistant_msg_2]
    DataProxy → Gateway → Client
```

## Code Organization

```
areal/v2/agent_service/
├── __init__.py          # Public exports (AgentRequest, AgentResponse, etc.)
├── README.md            # This document
├── auth.py              # Admin key auth helpers (hmac-safe comparison)
├── memory_authorization.py       # Exact principal/session/action grant contract
├── memory_authorization_store.py # Revocable in-memory grant control store
├── memory_broker.py     # Exact-grant host broker and session/turn incarnations
├── protocol.py          # Gateway protocol frame types
├── types.py             # AgentRequest, AgentResponse, EventEmitter, AgentRunnable
├── controller/
│   ├── __init__.py      # AgentController export
│   └── controller.py    # AgentController orchestrator
├── guard/
│   ├── __init__.py      # Module docstring
│   ├── __main__.py      # python -m areal.v2.agent_service.guard
│   └── app.py           # Guard Flask app (pass-through to areal.infra.rpc.guard)
├── gateway/
│   ├── __init__.py      # Public exports
│   ├── __main__.py      # python -m areal.v2.agent_service.gateway
│   ├── app.py           # create_gateway_app()
│   ├── bridge.py        # OpenResponsesBridge, mount_bridge()
│   └── config.py        # GatewayConfig dataclass
├── router/
│   ├── __init__.py      # Public exports
│   ├── __main__.py      # python -m areal.v2.agent_service.router
│   ├── app.py           # create_router_app()
│   ├── client.py        # RouterClient
│   └── config.py        # RouterConfig dataclass
├── data_proxy/
│   ├── __init__.py      # Public exports
│   ├── __main__.py      # python -m areal.v2.agent_service.data_proxy
│   ├── app.py           # create_data_proxy_app()
│   ├── client.py        # DataProxyClient
│   └── config.py        # DataProxyConfig dataclass
└── worker/
    ├── __init__.py      # Public exports
    ├── __main__.py      # python -m areal.v2.agent_service.worker
    ├── app.py           # create_worker_app()
    └── memory.py        # Disabled-by-default turn capability seam

examples/agent_service/
├── agent.py                  # ClaudeAgent (Claude Agent SDK)
├── run_agent_service.py      # Controller-based launcher + interactive demo
└── README.md                 # Example documentation
```
