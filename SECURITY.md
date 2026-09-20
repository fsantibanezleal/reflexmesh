# Security and authority model

Report a suspected vulnerability through the repository's private security reporting mechanism. Do not attach credentials, private traces or exploit payloads to public issues.

The native broker controls cooperating registered tools. It is not a kernel boundary, VM or container. Run untrusted executable code inside an independently configured sandbox. File validation does not eliminate races against hostile external writers. A model score or confidence never grants capabilities. Tool specifications and resource adapters belong to the trusted application.

The local server binds loopback, checks the Host and Origin, limits request size and requires a session bearer token for executable endpoints. Never expose it through a public proxy without a separately designed authentication, tenancy, isolation and quota layer. Static exported artifacts contain no local execution authority.

Cancellation requests do not guarantee rollback. HTTP write acknowledgments and process termination require application-specific postcondition verification. Unresolved outcomes retain resource conflicts until reconciled. Journal checksums detect accidental or unsynchronized modification; an attacker with write access to both source and journal is outside this process-level trust boundary.
