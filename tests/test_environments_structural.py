import pytest
from reflexmesh.environments.structural import StructuralEnvironment, structural_matrix


@pytest.mark.parametrize("spec", structural_matrix(), ids=lambda spec: spec.episode_id)
def test_frozen_structural_programs_execute_independent_output_truth(spec):
    with StructuralEnvironment(spec) as env:
        observation = env.observe()
        assert all(c.features.get("known") == 0 for c in observation.admissible)
        assert observation.state["uncertain"]
        for operation in list(env.actions)[:3]:
            env.step("a-" + operation)
        assert env.verify()
        assert len(env.runtime.snapshot()["state"]["intents"]) == 3
