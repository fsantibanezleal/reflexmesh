# Scoped MCP tool integration

`reflexmesh.mcp` implements MCP **2026-07-28 Streamable HTTP** discovery and
tool calls. It uses the revision's stateless, per-request protocol metadata;
there is no legacy `initialize` handshake or implicit version downgrade.
Install `reflexmesh[mcp]` for JSON Schema and bounded regular-expression support.

The supported surface is `server/discover`, paginated `tools/list`, and
`tools/call`, with JSON or SSE responses. Every request includes protocol
version, client identity and an empty client capability object. The transport
sends the required `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`, and schema
parameter headers. Unicode and control-bearing parameter values use the
specified base64 sentinel encoding. JSON-RPC IDs and response envelopes are
checked. Notifications are bounded observations, never instructions.

An endpoint and explicit tool allowlist come from trusted local configuration.
HTTPS uses the Python TLS defaults. Plain HTTP is available only for explicitly
enabled loopback endpoints. Redirects and embedded URL credentials are rejected.
Authentication headers can be supplied explicitly; the client does not search
environment variables, browsers, credential stores, or files for credentials.
OAuth discovery/authorization, stdio, legacy revisions, Tasks, sampling,
elicitation, resources and prompts are outside this adapter's advertised
capabilities. Unadvertised server requests fail; they do not launch an agent,
open a URL, or retry the tool automatically.

## Local authority

Discovery never creates execution authority. Register an `MCPBinding` through
an existing `Runtime`. The binding independently declares required capabilities,
canonical resource identifiers, read/write accesses, actual-state fingerprints,
local argument validation, and an outcome verifier. Tool descriptions,
`readOnlyHint`, other remote annotations, and response text cannot supply those
declarations. The returned local tool name is `mcp.<server_id>.<remote_tool>`.

```python
from reflexmesh.mcp import MCPClient, MCPServer, MCPBinding
from reflexmesh.runtime import EffectReceipt

client = MCPClient(MCPServer(
    server_id="inventory", url="https://inventory.example/mcp",
    allowed_tools=("set_stock",),
))
# These trusted functions bind an application object to its canonical resource
# and obtain fresh state from an independent source, not the tool's assertion.
binding = MCPBinding(
    tool="set_stock", capabilities=("inventory.write",),
    resources=registered_stock_resources,
    fingerprint=read_actual_stock_fingerprint,
    validate=validate_stock_arguments,
    verifier=verify_actual_stock_receipt,
)
tool_name = client.register(runtime, binding)
candidate = runtime.candidate(tool_name, {"sku": "local-approved-sku", "count": 3})
receipt = runtime.execute_candidate(candidate, idempotency_key="stock-change-42")
```

The named host functions in this integration sketch are application-specific;
the runnable disk-backed example is `tests/test_native_mcp.py`. A verifier
returns `EffectReceipt` with independently checked evidence and only the
declared changed resources. A mutating binding without such a verifier returns
`effect_unknown`, including when a server claims success. A trusted read-only
binding can verify that a bounded response arrived, but that does not establish
the truth of its contents. Remote read-only hints cannot create this binding.

Native admission checks capabilities, scope, revisions, leases, and duplicate
intent state immediately before the HTTP request. Schema descriptors are pinned
at registration: a subsequent explicit discovery refresh that changes/removes
the descriptor prevents dispatch under that registration. No automatic schema
refresh occurs inside a critical section. A fresh registration/version should
be selected by the trusted host after review.

## Bounds and uncertain effects

Defaults are a 30-second request deadline, 64 KiB request, 1 MiB response,
256 allowed tools, 16 discovery pages and 256 stream messages. Deadline and
cancellation monitoring disconnect active response sockets. DNS/connect and
the OS scheduler can add delay; this is not a hard real-time guarantee. Stream
closure requests cancellation under the protocol but cannot undo a remote
write. Transport failure, invalid output schemas, or unverified outcomes remain
unknown and hold native leases until reconciliation.

Schemas use a restricted JSON Schema 2020-12 profile: external references,
`patternProperties`, overly deep/large schemas, and invalid parameter-header
annotations are rejected. Local references are supported. Regex patterns have
length and execution limits; validation has keyword-count and elapsed checks.
These checks are practical rejection bounds, not a formal CPU/memory sandbox:
third-party validation keywords can do work between checks. Response content
is bounded untrusted data, with no automatic fetch of resource links or media.
The trusted host and registered verifier remain part of the security boundary.

The protocol tests use a real local HTTP server, SSE, real file writes, and the
compiled Rust broker. They cover allowlists, capability denial before tool
network traffic, schema/header attacks, descriptor changes, malformed envelopes,
redirects, pagination cycles, bounded responses, cancellation after a write,
and unsupported capability/version paths. They are interoperability evidence
for this surface, not MCP certification or a claim about every server.

Primary specification: [basic messages and metadata](https://modelcontextprotocol.io/specification/2026-07-28/basic),
[discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover),
[tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools),
[Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http).
