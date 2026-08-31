"""Unit tests for the LineageResult HTML renderer (issue #378).

Offline by design: the renderer consumes a ``LineageResult`` (or its
``model_dump()`` dict) and never touches a catalog. Results are built
programmatically from the public lineage models; RIDs are generated, and
assertions check the rendered page's content, escaping, and
self-containment.
"""

from __future__ import annotations

import json

from deriva_ml.execution.lineage import (
    DatasetSummary,
    ExecutionSummary,
    LineageNode,
    LineageResult,
    RootDescriptor,
    VersionAttribution,
    WorkflowSummary,
)
from deriva_ml.execution.lineage_html import lineage_result_to_html, main


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def _exec_summary(n: int, *, name="trainer", url=None, version=None) -> ExecutionSummary:
    wf = WorkflowSummary(rid=_rid(n + 100), name=name, url=url, version=version)
    return ExecutionSummary(rid=_rid(n), description=f"execution {n}", workflow=wf, status="Completed")


def _dataset_result() -> LineageResult:
    origin = _exec_summary(10, name="splitter", url="https://github.com/org/repo", version="1.0.0")
    toucher = _exec_summary(11, name="migration")
    walk = LineageNode(
        execution=origin,
        consumed_datasets=[DatasetSummary(rid=_rid(30), description="cohort", version="4.11.0")],
        parents=[LineageNode(execution=_exec_summary(12, name="loader"))],
    )
    root = RootDescriptor(
        rid=_rid(1),
        type="Dataset",
        description="training split <b>bold</b>",
        version="0.6.0",
        producing_execution=origin,
        origin_recorded=True,
        version_history=[
            VersionAttribution(version="0.1.0", execution_rid=origin.rid, execution=origin, description="created"),
            VersionAttribution(version="0.2.0", execution_rid=toucher.rid, execution=toucher, description="migrated"),
        ],
    )
    return LineageResult(root=root, lineage=walk, executions_visited=2, walked_complete=True)


def test_root_card_carries_identity_version_and_origin_badge():
    page = lineage_result_to_html(_dataset_result())
    assert _rid(1) in page
    assert "0.6.0" in page  # root current version
    assert "origin recorded" in page.lower()
    assert "splitter" in page  # origin workflow name


def test_version_history_earliest_first_with_origin_marker():
    page = lineage_result_to_html(_dataset_result())
    assert page.index("0.1.0") < page.index("0.2.0")
    assert "ORIGIN" in page  # first row marked
    assert "migration" in page  # toucher present, attributed to its version


def test_origin_unrecorded_and_not_applicable_badges():
    unrecorded = LineageResult(
        root=RootDescriptor(rid=_rid(2), type="Dataset", description=None, origin_recorded=False)
    )
    page = lineage_result_to_html(unrecorded)
    assert "origin unrecorded" in page.lower()

    asset = LineageResult(root=RootDescriptor(rid=_rid(3), type="Asset", description=None))
    asset_page = lineage_result_to_html(asset)
    assert "origin unrecorded" not in asset_page.lower()
    assert "origin recorded" not in asset_page.lower()


def test_walk_tree_nests_parents_and_pins_dataset_versions():
    page = lineage_result_to_html(_dataset_result())
    assert "loader" in page  # nested parent rendered
    assert f"{_rid(30)}" in page and "4.11.0" in page  # consumed dataset @ pinned version


def test_workflow_url_is_link_only_when_web_url():
    page = lineage_result_to_html(_dataset_result())
    assert 'href="https://github.com/org/repo"' in page
    no_url = LineageResult(
        root=RootDescriptor(rid=_rid(4), type="Execution", description=None),
        lineage=LineageNode(execution=_exec_summary(13, url="file:///local/checkout")),
    )
    page2 = lineage_result_to_html(no_url)
    assert 'href="file' not in page2  # non-web URL shown as text, never a link


def test_catalog_text_is_escaped():
    hostile = LineageResult(root=RootDescriptor(rid=_rid(5), type="Dataset", description='<script>alert("x")</script>'))
    page = lineage_result_to_html(hostile)
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_page_is_self_contained():
    page = lineage_result_to_html(_dataset_result())
    assert page.lstrip().startswith("<!DOCTYPE html>")
    assert "<script" not in page  # no JS at all
    assert '<link rel="stylesheet"' not in page  # CSS inline only
    assert "src=" not in page  # no external images/frames


def test_dict_input_renders_identically():
    result = _dataset_result()
    assert lineage_result_to_html(result) == lineage_result_to_html(result.model_dump())


def test_walk_flags_surface_truncation():
    capped = LineageResult(
        root=RootDescriptor(rid=_rid(6), type="Execution", description=None),
        lineage=LineageNode(execution=_exec_summary(14)),
        walked_complete=False,
        depth_capped=True,
        executions_visited=7,
    )
    page = lineage_result_to_html(capped)
    assert "partial" in page.lower() or "truncat" in page.lower()


def test_cli_offline_render(tmp_path):
    src = tmp_path / "lineage.json"
    out = tmp_path / "lineage.html"
    src.write_text(json.dumps(_dataset_result().model_dump()))
    rc = main(["--input", str(src), "--output", str(out)])
    assert rc == 0
    text = out.read_text()
    assert _rid(1) in text and text.lstrip().startswith("<!DOCTYPE html>")


def test_deep_chain_renders_without_recursion_error():
    """A walk near the 500-execution cap as one deep chain must render.

    Pins the codex P2: recursive rendering (frame + generator frame per
    level) exceeded Python's recursion limit on walks lookup_lineage
    itself completed successfully.
    """
    # Built as dicts: pydantic's own model_dump() has a serializer depth
    # ceiling on chains this deep (a separate, pre-existing limitation of
    # the models — the renderer's documented dict input is the path a
    # deep saved walk actually takes).
    node: dict = {"execution": _exec_summary(9000).model_dump(), "parents": []}
    for i in range(2999):
        node = {"execution": _exec_summary(8000 + i).model_dump(), "parents": [node]}
    data = {
        "root": RootDescriptor(rid=_rid(7), type="Execution", description=None).model_dump(),
        "lineage": node,
        "executions_visited": 3000,
        "walked_complete": True,
        "cycle_detected": False,
        "depth_capped": False,
    }
    page = lineage_result_to_html(data)
    assert page.count('<div class="exec">') == 3000


def test_cli_round_trips_non_latin_text(tmp_path):
    """Unicode catalog text survives write/read regardless of locale (codex P2)."""
    result = LineageResult(root=RootDescriptor(rid=_rid(8), type="Dataset", description="眼底画像 �команда ✓"))
    src = tmp_path / "l.json"
    out = tmp_path / "l.html"
    src.write_text(json.dumps(result.model_dump()), encoding="utf-8")
    assert main(["--input", str(src), "--output", str(out)]) == 0
    assert "眼底画像" in out.read_text(encoding="utf-8")
