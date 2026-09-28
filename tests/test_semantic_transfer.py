"""Executable truth and stimulus diversity for the separate semantic lane."""

import hashlib
import json

import pytest
from reflexmesh.environments.semantic_transfer import (
    PROGRAMS,
    SemanticEnvironment,
    semantic_matrix,
)


@pytest.mark.parametrize("spec", semantic_matrix(), ids=lambda spec: spec.episode_id)
def test_semantic_choices_have_independent_file_truth(spec):
    assert "stage-1.json" in spec.goal
    with SemanticEnvironment(spec) as environment:
        observation = environment.observe()
        actual = [c for c in observation.admissible if c.tool.startswith("sandbox.cap_")]
        assert len(actual) == 9
        assert all(c.features["known"] == 0 for c in actual)
        assert all(c.arguments["operation"].startswith("cap_") for c in actual)
        assert all(operation.replace("_", " ") not in spec.goal for operation in environment.program)
        assert not environment.verify()
        # A plausible wrong first-stage tool performs a real, wrong file write.
        decoy = next(
            key for key, (_, _, _, stage) in environment.actions.items()
            if stage == 0 and key != environment.correct_ids[0]
        )
        environment.step("a-" + decoy)
        assert (environment.root / "stage-1.json").exists()
        assert not environment.verify()
        # Repairing that output and completing the other actual effects is
        # required for success. A receipt alone is not the task oracle.
        for key in environment.correct_ids:
            environment.step("a-" + key)
        assert environment.verify()
        assert len(environment.runtime.snapshot()["state"]["intents"]) == 4


@pytest.mark.parametrize("family", PROGRAMS)
def test_seed_changes_stimulus_and_two_program_families_change_required_action(family):
    signatures = set()
    programs = set()
    for spec in (s for s in semantic_matrix() if s.family == family and s.variant == "nominal"):
        with SemanticEnvironment(spec) as environment:
            input_hash = hashlib.sha256((environment.root / "input.json").read_bytes()).hexdigest()
            signatures.add((input_hash, tuple(environment.actions)))
            programs.add(environment.program)
            assert json.loads((environment.root / "input.json").read_text()) == environment.inputs
    assert len(signatures) == 8
    assert len(programs) == (2 if family in {"D07", "D08"} else 1)
