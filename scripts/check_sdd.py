#!/usr/bin/env python3
"""The SDD gate: a design document exists, and every requirement names a gate that is real.

Three checks, in increasing strength:

1. A repo that contains implementation code has ``docs/design/SDD.md``.
2. Every requirement in it carries a ``Gate:`` line.
3. **The named gate exists.** The file is on disk and, when the gate names a test, that test name
   appears in it.

Check 3 is the one that matters. A requirement can name ``tests/test_nothing.py::test_imaginary``
and satisfy check 2 while verifying nothing at all, which is precisely the failure the rule was
written for: not a missing gate, but a gate that is believed and measures nothing. A guard that
stopped at check 2 would itself be an instance of it.

Standard library only, so this stays a cheap CI check.

Usage: ``python scripts/check_sdd.py [repo_root]``. Exits non-zero on any finding.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SDD_PATH = Path("docs/design/SDD.md")
FEATURE_GLOB = "docs/design/features/*/requirements.md"

#: A requirement opens with an identifier like ``R-001`` at the start of a line.
REQUIREMENT = re.compile(r"^(R-\d+)\b(.*)$")
GATE = re.compile(r"^\s*Gate:\s*(?P<target>\S+)\s*$")

#: EARS keywords. A requirement that uses none of them is prose wearing an identifier.
EARS_KEYWORDS = ("SHALL",)

CODE_DIRECTORIES = ("src", "app", "data-pipeline", "frontend/src", "python", "rust")

#: Gates that name something other than a file, and what makes each acceptable.
NON_FILE_GATE_PREFIXES = ("manual:", "review:")


def find_code(root: Path) -> list[Path]:
    found: list[Path] = []
    for directory in CODE_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            continue
        found.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
        found.extend(base.rglob("*.ts"))
        found.extend(base.rglob("*.tsx"))
    return found


def requirement_blocks(text: str) -> list[str]:
    """The fenced blocks that hold requirements.

    Requirements live inside fenced blocks, and only there. Scanning the whole document would treat
    a prose cross-reference to a requirement as a second declaration of it, which is how this guard
    first failed against the document it was written for.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


def parse_requirements(text: str) -> list[tuple[str, str, str | None]]:
    """Return ``(identifier, statement, gate)`` for every requirement found."""
    out: list[tuple[str, str, str | None]] = []
    lines: list[str] = []
    for block in requirement_blocks(text):
        lines.extend(block.splitlines())
    for index, line in enumerate(lines):
        match = REQUIREMENT.match(line.strip())
        if not match:
            continue
        identifier = match.group(1)
        statement_parts = [match.group(2).strip()]
        gate: str | None = None
        for following in lines[index + 1 :]:
            gate_match = GATE.match(following)
            if gate_match:
                gate = gate_match.group("target")
                break
            if REQUIREMENT.match(following.strip()) or not following.strip():
                if not following.strip():
                    continue
                break
            statement_parts.append(following.strip())
        out.append((identifier, " ".join(p for p in statement_parts if p), gate))
    return out


def gate_exists(root: Path, gate: str) -> tuple[bool, str]:
    """Check the named gate is real. Returns ``(ok, explanation)``."""
    if gate.startswith(NON_FILE_GATE_PREFIXES):
        return False, (
            f"{gate!r} is not a mechanical gate; a requirement is verified by something that "
            "fails on its own, not by a person remembering to look"
        )

    path_part, _, test_part = gate.partition("::")
    path = root / path_part
    if not path.is_file():
        return False, f"names {path_part!r}, which does not exist"

    if test_part:
        content = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix in {".ts", ".mjs", ".js"}:
            # A node:test name is a sentence, and a gate target cannot hold spaces, so the gate
            # spells it with underscores and must match the start of one `test("...")` title.
            spoken = test_part.replace("_", " ")
            if f'test("{spoken}' not in content:
                return False, f"names {test_part!r}, and no test in {path_part} begins {spoken!r}"
        elif f"def {test_part}" not in content:
            return False, f"names {test_part!r}, which is not defined in {path_part}"
    return True, ""


def check_document(root: Path, document: Path) -> list[str]:
    problems: list[str] = []
    text = document.read_text(encoding="utf-8", errors="replace")
    requirements = parse_requirements(text)

    if not requirements:
        problems.append(f"{document}: contains no requirements (expected lines starting 'R-NNN')")
        return problems

    seen: set[str] = set()
    for identifier, statement, gate in requirements:
        label = f"{document}:{identifier}"
        if identifier in seen:
            problems.append(f"{label}: identifier used more than once")
        seen.add(identifier)

        if not any(keyword in statement for keyword in EARS_KEYWORDS):
            problems.append(
                f"{label}: is not in EARS form (no SHALL); see conventions/spec-driven-development.md"
            )

        if gate is None:
            problems.append(
                f"{label}: has no 'Gate:' line. A requirement with no named verification is a wish"
            )
            continue

        ok, why = gate_exists(root, gate)
        if not ok:
            problems.append(f"{label}: {why}")

    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    problems: list[str] = []

    sdd = root / SDD_PATH
    code = find_code(root)

    if not sdd.is_file():
        if code:
            problems.append(
                f"{SDD_PATH} is missing, but this repo has {len(code)} implementation file(s). "
                "The design document comes before the code"
            )
        else:
            print("no implementation code and no SDD: nothing to check")
            return 0
    else:
        problems += check_document(root, sdd)

    for feature in sorted(root.glob(FEATURE_GLOB)):
        problems += check_document(root, feature.relative_to(root))

    if problems:
        print("SDD gate failed:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"SDD gate passed: {sdd.relative_to(root)}, every requirement names a gate that exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
