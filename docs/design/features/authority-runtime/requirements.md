# authority-runtime requirements

Date: 2026-09-23. Migration of the authorized existing implementation.

```text
R-001 WHEN an effect is proposed, THE runtime SHALL enforce revision, lease and durable intent ordering.
Gate: tests/test_runtime.py
R-002 IF an effect is unresolved after a crash, THE runtime SHALL require independent recovery evidence.
Gate: tests/test_workspace_recovery.py
R-003 WHEN a second process opens an owned journal, THE runtime SHALL reject it.
Gate: tests/test_runtime_ownership.py
```
