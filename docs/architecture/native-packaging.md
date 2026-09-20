# Native distribution checks

The package uses the setuptools backend with `setuptools-rust` and PyO3. Source
distributions include `Cargo.toml`, `Cargo.lock`, the Rust modules, Python source,
license and notices. Building from that source distribution creates the actual
native extension; there is no Python broker fallback.

Although Cargo enables PyO3's Python 3.11 limited API, the current setuptools
configuration emits **per-interpreter wheels** such as
`cp312-cp312-win_amd64`, not a universally installable `abi3` wheel. Use the wheel
matching the interpreter and platform. Do not rename its tags manually. The
release workflow builds Python 3.11, 3.12 and 3.13 wheels through cibuildwheel;
Linux release wheels require its manylinux repair step. Ordinary Linux CI build
wheels demonstrate installation on that runner but do not by themselves meet
PyPI's portable Linux wheel requirements.

The initial Engine CI run at commit `c3f5f28` completed all nine distribution
build/install jobs across Windows, Linux and macOS and Python 3.11-3.13. This
checks those actual runner architectures, not every architecture named by an OS.
Its runtime/scientific jobs separately exposed an import-path bug in the crash
test fixture; distribution success did not imply the full test suite passed.
The fixture was corrected to import the same installed package as the parent.
[CI evidence](https://github.com/fsantibanezleal/reflexmesh/actions/runs/35541915147).

Local final-wheel audit procedure:

```powershell
python -m build --outdir artifacts/package-audit-final
python -m twine check artifacts/package-audit-final/*
python -m venv .external/package-base-only
.external/package-base-only/Scripts/python -m pip install artifacts/package-audit-final/reflexmesh-0.1.0-cp312-cp312-win_amd64.whl
.external/package-base-only/Scripts/python -I -m reflexmesh.benchmarks.package_audit --checkpoints artifacts/checkpoints --require-base-only --output docs/architecture/native-package-base-audit.json
.external/package-base-only/Scripts/python -m pip check
```

The audit asserts an installed `site-packages` import, actual file write and
native replay, and actual C01 document routing with M01, M02 and cached native
M03/M04. It checks that Torch, sklearn, XGBoost, Gymnasium, SB3 and ONNX Runtime
are absent in the base-only environment and are not imported by these serving
paths. M03/M04 load the supplied fitted JSON checkpoints; the Python wheel does
not silently fabricate or train missing models. Native inference and broker
admission remain separate operations.

The full local evaluation environment is separate from both that pristine base
installation and active training. Its installed wheel has its own package/native
files. Reused scientific dependencies are exposed through a plain `.pth` path
entry, which does not recursively process the training environment's editable
package hooks. Verify the actual imported package and native binary before
starting long evaluations. This local reuse arrangement is not a published
dependency lock; each scientific run must retain its dependency provenance.

`native-package-base-audit.json` and the final distribution audit record portable
artifact names and hashes. The checks establish packaging, numerical execution
and a small actual-workflow smoke result. The frozen scientific matrix,
external subsets, CI suite and release publication are separate evidence gates.
No audit command here publishes to PyPI.
