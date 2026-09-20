# Integration

Start with [runtime and controller contracts](../runtime.md) for the Python API. Register a ToolSpec with exact argument validation, concrete resource declarations, current fingerprints and independently verifiable effect receipts. File, fixed-process, explicit HTTP and MCP adapters supply concrete implementations. Keep application registration outside model-produced content.

Use [workspace workflows](../workspace-workflows.md) for the CLI and authenticated local workbench. A recipe is a dependency graph of registered actions with independent SHA256 postconditions. The browser previews exact file preconditions, runs through the native broker, inspects receipts, cancels and performs reviewed recovery. Hosted static pages have no host-tool authority.

For multiple concurrent goals, use Controller's `policy_factory` with independent mutable policy instances. Episode-scoped planner queues, budgets and effect residuals must not leak between goals. A shared positional policy is caller-owned and suitable only when stateless or explicitly goal-aware. Factories and effect verifiers remain trusted application code.

OllamaPlanner accepts an explicit local model; the compatible HTTP planner adapter requires an explicit HTTPS endpoint and caller-supplied credential. Structured planner output selects a registered candidate and is checked before dispatch. AsyncDeliberator binds responses to the complete originating observation and expires stale proposals. A provider's timeout or cancelled client future does not prove that remote computation stopped.

Operate the local service on loopback with its bearer token. Do not expose the local API through the public static nginx host. Exported traces may contain selected file content or process output; inspect and sanitize user traces before sharing. Recovery evidence and operator attestations remain in the durable journal, and generic process termination never claims rollback.
