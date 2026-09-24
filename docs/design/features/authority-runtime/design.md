# authority-runtime design

Rust owns the admission reducer; Python owns adapters and SQLite persistence. Persist intent before an effect, then persist the verified result. Restart converts incomplete work to unknown without repeating it. Lifetime OS locks prevent two runtime owners. Recovery compares immutable registration and the current record digest; it never trusts an imported success label.

The product SDD defines the scope and deploy boundary. Requirements refer to executable gates and are not inferred from successful rendering or method names.
