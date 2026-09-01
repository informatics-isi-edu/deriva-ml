"""Golden byte-equivalence captures of ``DerivaML.lookup_lineage``.

These tests lock the CURRENT behavior of ``lookup_lineage`` byte-for-byte
so that later refactoring tasks (extracting the walk into its own engine
module) can be checked against an unambiguous ground truth. Every scenario
here builds a ``_FakeML`` harness (imported, not copied, from
``test_lookup_lineage_unit``), runs ``lookup_lineage``, and canonically
serializes the result to JSON. The serialized bytes are compared against a
committed golden file in ``tests/execution/goldens/``.

Regenerating goldens (pre-refactor only — regenerating after a behavior
change defeats the whole point of a regression gate)::

    UPDATE_LINEAGE_GOLDENS=1 DERIVA_ML_ALLOW_DIRTY=true uv run pytest \\
        tests/execution/test_lineage_goldens.py -q -p no:randomly

Then re-run WITHOUT the env var to confirm the freshly written goldens
byte-match a second independent run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.execution.test_lookup_lineage_unit import (
    _FakeML,
    _StubDataset,
    _StubWorkflow,
)

GOLDEN_DIR = Path(__file__).parent / "goldens"


def _canonical(dump: dict) -> bytes:
    return (json.dumps(dump, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _rid(n: int, prefix: str = "1") -> str:
    """Generate a deterministic, catalog-shaped RID for a scenario.

    Never a human-written literal — every RID used by a scenario builder
    below is produced by this helper (or a workflow/execution-prefixed
    sibling call), matching the codebase-wide RID discipline.
    """
    return f"{prefix}-{n:04X}"


# ---------------------------------------------------------------------------
# Scenario builders. Each returns (ml, root_rid, lookup_lineage_kwargs).
# ---------------------------------------------------------------------------


def _simple_chain():
    """DS-1 <- EXE-A <- DS-0 (no producer). One producing execution."""
    ml = _FakeML()
    ds0 = _rid(1)
    exe_a = _rid(1, prefix="2")
    ds1 = _rid(2)
    ml.add_dataset(ds0, description="root data", producer=None)
    ml.add_execution(
        exe_a,
        description="train",
        workflow=_StubWorkflow(workflow_rid=_rid(1, prefix="3"), name="trainer"),
        input_datasets=[_StubDataset(ds0, "root data", "0.1.0")],
    )
    ml.add_dataset(ds1, description="trained set", producer=exe_a)
    return ml, ds1, {}


def _two_level():
    """DS-2 <- EXE-B <- DS-1 <- EXE-A <- DS-0 (no producer)."""
    ml = _FakeML()
    ds0 = _rid(10)
    exe_a = _rid(10, prefix="2")
    ds1 = _rid(11)
    exe_b = _rid(11, prefix="2")
    ds2 = _rid(12)
    ml.add_dataset(ds0, description="raw", producer=None)
    ml.add_execution(
        exe_a,
        description="stage one",
        workflow=_StubWorkflow(workflow_rid=_rid(10, prefix="3"), name="stage-one"),
        input_datasets=[_StubDataset(ds0, "raw", "0.1.0")],
    )
    ml.add_dataset(ds1, description="intermediate", producer=exe_a)
    ml.add_execution(
        exe_b,
        description="stage two",
        workflow=_StubWorkflow(workflow_rid=_rid(11, prefix="3"), name="stage-two"),
        input_datasets=[_StubDataset(ds1, "intermediate", "0.1.0")],
    )
    ml.add_dataset(ds2, description="final", producer=exe_b)
    return ml, ds2, {}


def _diamond():
    """EXE-C consumed two datasets (DS-A, DS-B), both produced by EXE-X.

    One producer (EXE-X) reached via two distinct input datasets — the
    walk must dedupe to a single expanded node plus an already-shown
    marker is NOT produced here (dedup happens via the parent-RID set,
    so EXE-X is expanded exactly once and appears once in ``parents``).
    """
    ml = _FakeML()
    ds_root = _rid(20)
    exe_x = _rid(20, prefix="2")
    ds_a = _rid(21)
    ds_b = _rid(22)
    exe_c = _rid(21, prefix="2")
    ds_out = _rid(23)
    ml.add_dataset(ds_root, description="shared source", producer=None)
    ml.add_execution(
        exe_x,
        description="shared producer",
        workflow=_StubWorkflow(workflow_rid=_rid(20, prefix="3"), name="splitter"),
        input_datasets=[_StubDataset(ds_root, "shared source", "0.1.0")],
    )
    ml.add_dataset(ds_a, description="branch a", producer=exe_x)
    ml.add_dataset(ds_b, description="branch b", producer=exe_x)
    ml.add_execution(
        exe_c,
        description="merge",
        workflow=_StubWorkflow(workflow_rid=_rid(21, prefix="3"), name="merger"),
        input_datasets=[
            _StubDataset(ds_a, "branch a", "0.1.0"),
            _StubDataset(ds_b, "branch b", "0.1.0"),
        ],
    )
    ml.add_dataset(ds_out, description="merged", producer=exe_c)
    return ml, ds_out, {}


def _cycle():
    """A -> B -> A: EXE-B consumes what EXE-A produced, and EXE-A (via a
    scripted producer edge) is also recorded as consuming what EXE-B
    produced, creating an active-path cycle the walk must detect rather
    than recurse forever.
    """
    ml = _FakeML()
    ds_a = _rid(30)
    exe_a = _rid(30, prefix="2")
    ds_b = _rid(31)
    exe_b = _rid(31, prefix="2")
    # DS-A is produced by EXE-B (closing the loop); EXE-A consumes DS-A.
    ml.add_dataset(ds_a, description="cyclic a", producer=exe_b)
    ml.add_execution(
        exe_a,
        description="cyclic exec a",
        input_datasets=[_StubDataset(ds_a, "cyclic a", "0.1.0")],
    )
    # DS-B is produced by EXE-A; EXE-B consumes DS-B, closing the cycle.
    ml.add_dataset(ds_b, description="cyclic b", producer=exe_a)
    ml.add_execution(
        exe_b,
        description="cyclic exec b",
        input_datasets=[_StubDataset(ds_b, "cyclic b", "0.1.0")],
    )
    # Root: a dataset produced by EXE-A so the walk starts on the active path.
    ds_root = _rid(32)
    ml.add_dataset(ds_root, description="cycle root", producer=exe_a)
    return ml, ds_root, {}


def _depth_cap():
    """Same shape as two_level, but depth=0 -> only the immediate producer."""
    ml, root, _ = _two_level()
    return ml, root, {"depth": 0}


def _member_fallback():
    """Dataset with no recorded (version) producer, but member producers set.

    The walk must seed from the member producer even though
    ``root.producing_execution`` stays None (origin not recorded).
    """
    ml = _FakeML()
    ds_src = _rid(40)
    exe_up = _rid(40, prefix="2")
    ds_im = _rid(41)
    ml.add_dataset(ds_src, description="source", producer=None)
    ml.add_execution(
        exe_up,
        description="upload",
        workflow=_StubWorkflow(workflow_rid=_rid(40, prefix="3"), name="uploader"),
        input_datasets=[_StubDataset(ds_src, "source", "0.1.0")],
    )
    ml.add_dataset(ds_im, description="image set", producer=None)
    ml.set_member_producers(ds_im, {exe_up})
    return ml, ds_im, {}


def _cap_truncation():
    """6-chain, max_executions=3 -> partial walk, walked_complete False."""
    ml = _FakeML()
    ds = [_rid(50 + i) for i in range(7)]
    exes = [_rid(50 + i, prefix="2") for i in range(6)]
    ml.add_dataset(ds[0], description="link 0", producer=None)
    for i in range(6):
        ml.add_execution(
            exes[i],
            description=f"link {i}",
            workflow=_StubWorkflow(workflow_rid=_rid(50 + i, prefix="3"), name=f"stage-{i}"),
            input_datasets=[_StubDataset(ds[i], f"link {i}", "0.1.0")],
        )
        ml.add_dataset(ds[i + 1], description=f"link {i + 1}", producer=exes[i])
    return ml, ds[6], {"max_executions": 3}


def _failed_lookup():
    """A mid-chain execution's producer RID is never registered via
    ``add_execution`` — ``lookup_execution`` raises for it, and
    ``_walk_node`` treats the parent as unreachable (defensive path:
    returns ``None`` for that node rather than failing the whole walk,
    silently dropping it from ``parents``).

    Shape: DS-OUT <- EXE-TOP <- DS-MID (producer = an execution RID that
    was never registered with ``add_execution``, so ``lookup_execution``
    raises DerivaMLException when the walk tries to expand it as a
    parent).
    """
    ml = _FakeML()
    unresolvable_exec = _rid(60, prefix="2")  # never added via add_execution
    ds_mid = _rid(61)
    exe_top = _rid(61, prefix="2")
    ds_out = _rid(62)
    ml.add_dataset(ds_mid, description="mid dataset", producer=unresolvable_exec)
    ml.add_execution(
        exe_top,
        description="top",
        workflow=_StubWorkflow(workflow_rid=_rid(61, prefix="3"), name="top-stage"),
        input_datasets=[_StubDataset(ds_mid, "mid dataset", "0.1.0")],
    )
    ml.add_dataset(ds_out, description="out dataset", producer=exe_top)
    return ml, ds_out, {}


SCENARIOS = {
    "simple_chain": _simple_chain,
    "two_level": _two_level,
    "diamond": _diamond,
    "cycle": _cycle,
    "depth_cap": _depth_cap,
    "member_fallback": _member_fallback,
    "cap_truncation": _cap_truncation,
    "failed_lookup": _failed_lookup,
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_lookup_lineage_matches_golden(name: str):
    ml, root, kwargs = SCENARIOS[name]()
    got = _canonical(ml.lookup_lineage(root, **kwargs).model_dump(mode="json"))
    golden_path = GOLDEN_DIR / f"{name}.json"
    if os.environ.get("UPDATE_LINEAGE_GOLDENS"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_bytes(got)
    assert golden_path.exists(), "golden missing — UPDATE_LINEAGE_GOLDENS=1 (pre-refactor only)"
    assert got == golden_path.read_bytes(), f"lookup_lineage BYTES diverged from golden '{name}'"
