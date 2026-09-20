"""Cached compiled scorer validation and concurrency, with no model service calls."""

import importlib.machinery
import importlib.util
import json
import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native():
    extension = os.environ.get("REFLEXMESH_NATIVE_TEST_EXTENSION")
    if extension:
        loader = importlib.machinery.ExtensionFileLoader("_native", str(Path(extension).resolve()))
        spec = importlib.util.spec_from_loader("_native", loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module
    from reflexmesh import _native

    return _native


@pytest.fixture
def linear_export():
    return json.dumps(
        {
            "schema_version": 1,
            "feature_names": ["x", "y"],
            "coef": [2, -1],
            "intercept": 0.5,
            "calibration": {"kind": "platt", "coef": 1.2, "intercept": -0.3},
        }
    )


@pytest.fixture
def tree_export():
    return json.dumps(
        {
            "schema_version": 1,
            "feature_names": ["x", "y"],
            "base_margin": 0,
            "trees": [
                {
                    "nodes": [
                        {
                            "node_id": 0,
                            "feature_index": 0,
                            "threshold": 1,
                            "left": 1,
                            "right": 2,
                            "missing": 1,
                        },
                        {"node_id": 1, "leaf": -2},
                        {"node_id": 2, "leaf": 2},
                    ]
                }
            ],
            "calibration": {"kind": "platt", "coef": 1, "intercept": 0},
        }
    )


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_cached_packed_and_json_paths_agree_across_threads(
    native, linear_export, tree_export, kind
):
    cls, model = (
        (native.LinearScorer, linear_export)
        if kind == "linear"
        else (native.TreeScorer, tree_export)
    )
    scorer = cls(model)
    rows = [[0.9999, 0], [1, 0], [1.0001, 0]]
    packed = struct.pack("<6f", *(value for row in rows for value in row))
    reference = scorer.score_f32(packed, 3)
    # Float32 packed values are authoritative for the tree threshold path.
    unpacked = struct.unpack("<6f", packed)
    assert reference == json.loads(
        scorer.score(json.dumps([unpacked[i : i + 2] for i in (0, 2, 4)]))
    )
    with ThreadPoolExecutor(max_workers=12) as executor:
        assert all(
            result == reference
            for result in executor.map(lambda _: scorer.score_f32(packed, 3), range(120))
        )
    names = scorer.feature_names
    names.append("not a model feature")
    assert scorer.feature_names == ["x", "y"]
    if kind == "tree":
        assert reference[0] == pytest.approx(1 / (1 + math.exp(2)))
        assert reference[1] == reference[2] == pytest.approx(1 / (1 + math.exp(-2)))


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_cached_rejects_dimension_nonfinite_and_payload_mismatch(
    native, linear_export, tree_export, kind
):
    cls, model = (
        (native.LinearScorer, linear_export)
        if kind == "linear"
        else (native.TreeScorer, tree_export)
    )
    scorer = cls(model)
    for payload, count in (
        (b"", 1),
        (struct.pack("<2f", 1, 2), 2),
        (b"", 4097),
        (struct.pack("<2f", float("nan"), 1), 1),
        (struct.pack("<2f", 1, float("inf")), 1),
    ):
        with pytest.raises(RuntimeError):
            scorer.score_f32(payload, count)
    assert scorer.score_f32(b"", 0) == []
    with pytest.raises(RuntimeError):
        scorer.score("[[1]]")


def test_invalid_models_fail_at_construction(native, linear_export, tree_export):
    invalid = json.loads(tree_export)
    invalid["trees"][0]["nodes"][0]["left"] = 0
    with pytest.raises(RuntimeError, match="cyclic"):
        native.TreeScorer(json.dumps(invalid))
    invalid = json.loads(linear_export)
    invalid["coef"].append(0)
    with pytest.raises(RuntimeError, match="dimension"):
        native.LinearScorer(json.dumps(invalid))
