# learning-metacontrol requirements

Date: 2026-09-23. Migration of the authorized existing implementation.

```text
R-001 WHEN fitting candidate policies, THE pipeline SHALL keep seed/group splits disjoint.
Gate: tests/test_learning_integrity.py
R-002 WHEN allocation labels are constructed, THE paired branches SHALL have equal starting candidate features and causal histories.
Gate: tests/test_learning_pairing.py
R-003 WHEN a paired cost does not justify delegation, THE gate selection SHALL permit never-defer.
Gate: tests/test_learning_metacontrol.py
```
