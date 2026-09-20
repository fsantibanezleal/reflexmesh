# Contributing

Use a scoped task branch and pull request to `develop`, followed by a separate promotion to `main`. Keep unrelated work intact. Contributions must include provenance for scientific claims and independent effect verification for new tools.

Install Python 3.11 or later, a stable Rust compiler and platform C/C++ build tools. Create a virtual environment and run `python -m pip install -e '.[train,server,dev]'`. Run `cargo test --locked`, `cargo clippy --all-targets -- -D warnings`, and `python -m pytest`. Heavy experiments are explicit pipeline commands and never run as a side effect of package import or serving.

New policies need genuine training/inference behavior, checkpoint/version contracts and evaluation against held-out outcome truth. Do not use case IDs, hidden evaluator targets or future outcomes as observation features. Report missing outcomes separately. Do not claim generality, calibrated safety or scientific novelty from demonstration agreement.

Registered effectors must validate all arguments and resource handles at dispatch, declare effect uncertainty, document cancellation semantics and never execute text returned by a model as arbitrary code. Native reducers must preserve deterministic replay and bounded state. External model/data/code licenses remain separate from this repository's Apache-2.0 license.
