# local-controller requirements

Date: 2026-09-23. Migration of the authorized existing implementation.

```text
R-001 WHEN a recipe is received, THE service SHALL validate all steps before executing any.
Gate: tests/test_workflows.py
R-002 WHEN concurrent goals require mutable policies, THE controller SHALL reject shared factory instances.
Gate: tests/test_controller.py
R-003 WHEN remote tool arguments violate registration, THE MCP bridge SHALL reject them.
Gate: tests/test_native_mcp.py
```
