# Architecture

The native broker owns exact capabilities, registered action versions, resource revisions, leases, cancellation and journal state. Python adapters bind actual files, processes and explicit service endpoints to those contracts. Models receive a candidate set and return proposals; they cannot extend their own authority.

Read [native broker](native-broker.md) for the reducer, receipt states and lifetime journal ownership, then [runtime contracts](../runtime.md) for the actual adapter sequence and controller lifecycle. Unknown effects remain unresolved across restart until independent evidence supports reconciliation. No replay repeats an effect.

[MCP integration](native-mcp.md) specifies the supported transport, schema checks, endpoint registration and ambiguous-write boundary. [Distribution packaging](native-packaging.md) records native build/source provenance and installed-wheel checks. The JSON audit files alongside those pages describe the exact runs named in them; earlier audits are not silently attributed to later source revisions.

[External transfer](native-transfer-benchmarks.md) describes pinned upstream environments, task eligibility and restricted comparability. A native admission count is a runtime observation, not a benchmark reward or proof of policy generalization.
