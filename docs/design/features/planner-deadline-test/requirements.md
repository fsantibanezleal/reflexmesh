# Deadline regression reliability

```text
R-001 WHEN a pending planner reaches its TTL, THE regression SHALL verify the fast fallback with a controlled clock and pending future independent of host scheduling.
Gate: tests/test_learning_metacontrol.py::test_changed_binding_discards_proposal_and_ttl_bounds_affected_hold
```
