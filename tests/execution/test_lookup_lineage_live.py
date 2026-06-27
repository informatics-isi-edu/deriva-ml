"""Live-catalog smoke test for ``DerivaML.lookup_lineage``.

Builds a two-execution data-flow chain against a real catalog and
walks the lineage of the leaf dataset back to the root execution.
Gated on ``DERIVA_HOST`` like the other live-smoke tests in this
directory.
"""

from __future__ import annotations

import os

import pytest

from deriva_ml import MLVocab as vc
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion, VersionPart
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.lineage import LineageResult


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_lookup_lineage_two_execution_chain(test_ml):
    """exe1 -> ds1 -> exe2 -> ds2; walk from ds2 back to exe1."""
    test_ml.add_term(vc.dataset_type, "LineageTest", description="Lineage smoke")
    test_ml.add_term(vc.workflow_type, "Lineage Test", description="Lineage smoke")

    wf = test_ml.create_workflow(
        name="Lineage smoke workflow",
        workflow_type="Lineage Test",
        description="Lineage smoke test workflow",
    )

    # Execution 1 produces ds1.
    exe1 = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="lineage exe1"),
    )
    ds1 = exe1.create_dataset(
        dataset_types="LineageTest",
        description="lineage smoke ds1",
        version=DatasetVersion(1, 0, 0),
    )

    # Execution 2 consumes ds1 as input and produces ds2.
    exe2 = test_ml.create_execution(
        ExecutionConfiguration(
            workflow=wf,
            description="lineage exe2",
            datasets=[DatasetSpec(rid=ds1.dataset_rid, version=ds1.current_version)],
        ),
    )
    ds2 = exe2.create_dataset(
        dataset_types="LineageTest",
        description="lineage smoke ds2",
        version=DatasetVersion(1, 0, 0),
    )

    # Walk lineage from ds2.
    result = test_ml.lookup_lineage(ds2.dataset_rid)

    assert isinstance(result, LineageResult)
    assert result.walked_complete is True
    assert result.cycle_detected is False
    assert result.root.type == "Dataset"
    assert result.root.rid == ds2.dataset_rid

    # Producing execution of ds2 is exe2.
    assert result.root.producing_execution is not None
    assert result.root.producing_execution.rid == exe2.execution_rid
    assert result.lineage is not None
    assert result.lineage.execution.rid == exe2.execution_rid

    # exe2 consumed ds1.
    consumed_rids = {ds.rid for ds in result.lineage.consumed_datasets}
    assert ds1.dataset_rid in consumed_rids

    # exe1 should appear as a parent of exe2.
    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert exe1.execution_rid in parent_rids
    assert result.executions_visited >= 2


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_lookup_lineage_descends_into_member_asset_producers(test_ml, tmp_path):
    """tk-018: lookup_lineage(dataset) reaches the source the dataset's MEMBER
    assets were produced from, even when a different execution versioned the
    dataset.

    Shape built on the catalog:
        exec_src  -> DS_SRC (source dataset)
        exec_up   consumes DS_SRC, produces Image asset members (member producer)
        DS_IM     version-produced by exec_ds, members = exec_up's Image assets

    Assert: lookup_lineage(DS_IM) surfaces exec_up as a parent (reached via the
    member assets), and exec_up's consumed DS_SRC appears under it.

    The domain-schema ``Image`` asset table is used because it lives in the same
    schema as the ``Dataset_Image`` element-type association table (already
    registered at catalog creation), avoiding cross-schema lookup issues that
    arise with ``Execution_Asset`` (ML-schema table vs domain-schema association).

    Args:
        test_ml: Clean DerivaML instance backed by a real catalog, provided by the
            conftest session fixture.
        tmp_path: Pytest-provided temporary directory for staging asset files.
    """
    # --- Vocabulary setup ---
    test_ml.add_term(vc.dataset_type, "TK018Source", description="TK-018 source dataset type")
    test_ml.add_term(vc.dataset_type, "TK018Image", description="TK-018 image-like dataset type")
    test_ml.add_term(vc.workflow_type, "TK018 Lineage Test", description="TK-018 lineage smoke")

    wf = test_ml.create_workflow(
        name="TK018 member-asset lineage workflow",
        workflow_type="TK018 Lineage Test",
        description="TK-018 member-asset traversal smoke test",
    )

    # --- exec_src: produces DS_SRC ---
    exec_src = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="tk018 exec_src"),
    )
    with exec_src.execute():
        ds_src = exec_src.create_dataset(
            dataset_types="TK018Source",
            description="tk018 source dataset",
            version=DatasetVersion(1, 0, 0),
        )
    exec_src.commit_output_assets()
    ds_src_rid = ds_src.dataset_rid

    # --- Create a Subject row so Image assets can satisfy its FK ---
    # Image in the test domain schema has a Subject FK; insert one directly.
    domain_path = test_ml._domain_path()
    subject_rows = list(domain_path.tables["Subject"].insert([{"Name": "tk018-subject"}]))
    subject_rid = subject_rows[0]["RID"]

    # --- exec_up: consumes DS_SRC, produces an Image asset (member producer) ---
    exec_up = test_ml.create_execution(
        ExecutionConfiguration(
            workflow=wf,
            description="tk018 exec_up (image asset producer)",
            datasets=[DatasetSpec(rid=ds_src_rid, version=ds_src.current_version)],
        ),
    )
    image_file = tmp_path / "tk018_image.txt"
    image_file.write_text("tk018 image asset content")

    with exec_up.execute():
        exec_up.asset_file_path(
            "Image",
            image_file,
            Subject=subject_rid,
        )
    exec_up.commit_output_assets()
    exec_up_rid = exec_up.execution_rid

    # Retrieve the Image RID created by exec_up (used as DS_IM member).
    uploaded_assets = exec_up.uploaded_assets
    image_entries = uploaded_assets.get(f"{test_ml.default_schema}/Image", [])
    assert image_entries, (
        f"exec_up did not produce any Image rows in {test_ml.default_schema}/Image; "
        f"uploaded keys: {list(uploaded_assets.keys())}"
    )
    asset_rid = image_entries[0].asset_rid

    # --- Ensure Image is registered as a dataset element type (idempotent) ---
    # Dataset_Image is only created by create_demo_datasets (ensure_datasets);
    # on a clean test_ml catalog it may not exist yet.
    test_ml.add_dataset_element_type("Image")

    # --- exec_ds: creates DS_IM (version-producer), adds the Image as member ---
    # The essential invariant: DS_IM's Dataset_Version.Execution = exec_ds,
    # while the Image's Image_Execution Output producer = exec_up.
    exec_ds = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="tk018 exec_ds (dataset assembler)"),
    )
    with exec_ds.execute():
        ds_im = exec_ds.create_dataset(
            dataset_types="TK018Image",
            description="tk018 image-like dataset versioned by exec_ds",
            version=DatasetVersion(1, 0, 0),
        )
        ds_im.add_dataset_members(
            {"Image": [asset_rid]},
            description="tk018: add exec_up's Image asset as member",
        )
    exec_ds.commit_output_assets()
    ds_im_rid = ds_im.dataset_rid
    exec_ds_rid = exec_ds.execution_rid

    # Sanity: version-producer must be exec_ds, NOT exec_up.
    assert exec_ds_rid != exec_up_rid, "exec_ds and exec_up must be distinct executions for tk-018 invariant"

    # --- Walk lineage from DS_IM ---
    result = test_ml.lookup_lineage(ds_im_rid)

    assert isinstance(result, LineageResult)
    assert result.root.type == "Dataset"
    assert result.root.rid == ds_im_rid

    # The member-producer (exec_up) is reachable as a parent somewhere in the
    # tree (directly under the root version-producer node).
    assert result.lineage is not None
    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert exec_up_rid in parent_rids, f"member-producer {exec_up_rid} not surfaced; parents were {parent_rids}"

    # And from exec_up the walk reaches the source dataset DS_SRC it consumed.
    up_node = next(p for p in result.lineage.parents if p.execution.rid == exec_up_rid)
    consumed_src = {d.rid for d in up_node.consumed_datasets}
    assert ds_src_rid in consumed_src, (
        f"source dataset {ds_src_rid} not reached via member-producer; consumed were {consumed_src}"
    )


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_lookup_lineage_dataset_with_no_producer(test_ml):
    """A Dataset with no Dataset_Version row at all has no producer."""
    test_ml.add_term(vc.dataset_type, "LineageOrphan", description="Orphan smoke")

    # Use the Dataset module directly to skip the version-creation path.
    from deriva_ml.dataset.dataset import Dataset

    rid = (
        Dataset._create_dataset_record(  # type: ignore[attr-defined]
            ml_instance=test_ml,
            dataset_types=["LineageOrphan"],
            description="lineage orphan smoke",
            execution_rid=None,
        )
        if hasattr(Dataset, "_create_dataset_record")
        else None
    )

    if rid is None:
        pytest.skip(
            "Cannot create a dataset without a version row in this build; no public API exposes the producer-less path."
        )

    result = test_ml.lookup_lineage(rid)
    assert result.root.type == "Dataset"
    assert result.root.producing_execution is None
    assert result.lineage is None
    assert result.walked_complete is True


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_member_producer_query_scales_to_many_members(test_ml, tmp_path):
    """tk-023: the member-producer query must not 404 on a dataset with hundreds
    of members. Builds a dataset whose 250 asset members are produced by one
    upstream execution, then asserts lookup_lineage surfaces that producer.

    The OLD client-side ``.in_()`` over >=200 member RIDs blew the Apache URL
    limit (~8 KB) and 404'd. The rewritten server-side membership join carries
    only the dataset RID in the URL, so it is safe at any member count. This
    test is the live gate that proves the new join works on a real catalog.

    Shape built on the catalog::

        exec_src  → DS_SRC (source dataset)
        exec_up   consumes DS_SRC; produces N_MEMBERS=250 Image assets
        DS_BIG    versioned by exec_ds; members = all 250 exec_up Image assets

    Assert: ``lookup_lineage(DS_BIG)`` does NOT raise (no 404) AND surfaces
    ``exec_up`` as a parent in the lineage tree (reached via the member assets).

    Args:
        test_ml: Clean DerivaML instance backed by a real catalog, provided by
            the conftest session fixture.
        tmp_path: Pytest-provided temporary directory for staging asset files.
    """
    N_MEMBERS = 250  # > the ~hundreds threshold where the old URL blew up

    # --- Vocabulary setup ---
    test_ml.add_term(vc.dataset_type, "TK023Source", description="TK-023 source dataset type")
    test_ml.add_term(vc.dataset_type, "TK023Big", description="TK-023 large member dataset type")
    test_ml.add_term(vc.workflow_type, "TK023 Scale Test", description="TK-023 member-producer scale smoke")

    wf = test_ml.create_workflow(
        name="TK023 member-producer scale workflow",
        workflow_type="TK023 Scale Test",
        description="TK-023: member-producer query must not 404 at >=200 members",
    )

    # --- exec_src: produces DS_SRC (source dataset) ---
    exec_src = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="tk023 exec_src"),
    )
    with exec_src.execute():
        ds_src = exec_src.create_dataset(
            dataset_types="TK023Source",
            description="tk023 source dataset",
            version=DatasetVersion(1, 0, 0),
        )
    exec_src.commit_output_assets()
    ds_src_rid = ds_src.dataset_rid

    # --- Create a Subject row so Image assets can satisfy its FK ---
    domain_path = test_ml._domain_path()
    subject_rows = list(domain_path.tables["Subject"].insert([{"Name": "tk023-subject"}]))
    subject_rid = subject_rows[0]["RID"]

    # --- exec_up: consumes DS_SRC; produces N_MEMBERS Image assets ---
    exec_up = test_ml.create_execution(
        ExecutionConfiguration(
            workflow=wf,
            description=f"tk023 exec_up ({N_MEMBERS}-image asset producer)",
            datasets=[DatasetSpec(rid=ds_src_rid, version=ds_src.current_version)],
        ),
    )

    with exec_up.execute():
        for i in range(N_MEMBERS):
            asset_file = tmp_path / f"tk023_image_{i:04d}.txt"
            asset_file.write_text(f"tk023 image asset {i}")
            exec_up.asset_file_path(
                "Image",
                asset_file,
                Subject=subject_rid,
            )
    exec_up.commit_output_assets()
    exec_up_rid = exec_up.execution_rid

    # Confirm exec_up produced the expected number of Image assets.
    uploaded_assets = exec_up.uploaded_assets
    image_entries = uploaded_assets.get(f"{test_ml.default_schema}/Image", [])
    assert len(image_entries) == N_MEMBERS, (
        f"exec_up produced {len(image_entries)} Image assets, expected {N_MEMBERS}; "
        f"uploaded keys: {list(uploaded_assets.keys())}"
    )
    asset_rids = [entry.asset_rid for entry in image_entries]

    # --- Ensure Image is registered as a dataset element type (idempotent) ---
    test_ml.add_dataset_element_type("Image")

    # --- exec_ds: creates DS_BIG (version-producer) and adds all N_MEMBERS members ---
    # DS_BIG's Dataset_Version.Execution = exec_ds (version-producer);
    # the Image assets' Image_Execution Output producer = exec_up (member-producer).
    exec_ds = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="tk023 exec_ds (dataset assembler)"),
    )
    with exec_ds.execute():
        ds_big = exec_ds.create_dataset(
            dataset_types="TK023Big",
            description=f"tk023 large dataset with {N_MEMBERS} Image members",
            version=DatasetVersion(1, 0, 0),
        )
        ds_big.add_dataset_members(
            {"Image": asset_rids},
            description=f"tk023: add all {N_MEMBERS} exec_up Image assets as members",
        )
    exec_ds.commit_output_assets()
    ds_big_rid = ds_big.dataset_rid

    # Sanity: version-producer must be exec_ds, NOT exec_up.
    assert exec_ds.execution_rid != exec_up_rid, "exec_ds and exec_up must be distinct executions for tk-023 invariant"

    # --- Walk lineage from DS_BIG ---
    # This must NOT raise (the old .in_() code 404'd at this member count).
    result = test_ml.lookup_lineage(ds_big_rid)

    seen: set[str] = set()

    def _collect(node):
        if node is None:
            return
        seen.add(node.execution.rid)
        for p in node.parents:
            _collect(p)

    _collect(result.lineage)

    assert exec_up_rid in seen, (
        f"member-producer {exec_up_rid} not surfaced for a {N_MEMBERS}-member dataset; saw {seen}"
    )
    assert result.cycle_detected is False


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_lookup_lineage_reflects_consumed_version_not_latest(test_ml):
    """A dataset consumed at v1 must contribute v1's producers to lineage even
    after it is mutated to v2 by a different execution (tk-020 Gap 1).

    Shape built on the catalog:
        exec_v1  produces D@1.0.0
        exec_mid consumes D@1.0.0, produces DS_OUT
        exec_v2  later releases D to v2 (2.0.0) via mark_dev + release

    Assert: lookup_lineage(DS_OUT) surfaces exec_v1 (the consumed-version
    producer of D, i.e. the execution stamped on ``Dataset_Version`` for
    D@1.0.0), and does NOT surface exec_v2 (the later/latest producer
    stamped on D@2.0.0).

    Args:
        test_ml: Clean DerivaML instance backed by a real catalog, provided by
            the conftest session fixture.
    """
    # --- Vocabulary + workflow setup ---
    test_ml.add_term(vc.dataset_type, "TK020Consumed", description="TK-020 consumed dataset type")
    test_ml.add_term(vc.dataset_type, "TK020Output", description="TK-020 output dataset type")
    test_ml.add_term(vc.workflow_type, "TK020 Lineage Test", description="TK-020 consumed-version smoke")

    wf = test_ml.create_workflow(
        name="TK020 consumed-version lineage workflow",
        workflow_type="TK020 Lineage Test",
        description="TK-020 consumed-version regression test",
    )

    # --- exec_v1: produces D@1.0.0 ---
    # Dataset_Version.Execution = exec_v1 for version 1.0.0.
    exec_v1 = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="tk020 exec_v1 (D@1.0.0 producer)"),
    )
    d = exec_v1.create_dataset(
        dataset_types="TK020Consumed",
        description="tk020 dataset D, v1",
        version=DatasetVersion(1, 0, 0),
    )
    d_rid = d.dataset_rid
    exec_v1_rid = exec_v1.execution_rid
    v1 = d.current_version  # DatasetVersion(1, 0, 0)

    # --- exec_mid: consumes D@v1, produces DS_OUT ---
    # The DatasetSpec pins the CONSUMED version so Dataset_Execution.Dataset_Version
    # points at the 1.0.0 Dataset_Version row (exec_v1's row).
    exec_mid = test_ml.create_execution(
        ExecutionConfiguration(
            workflow=wf,
            description="tk020 exec_mid (consumer of D@v1, producer of DS_OUT)",
            datasets=[DatasetSpec(rid=d_rid, version=v1)],
        ),
    )
    ds_out = exec_mid.create_dataset(
        dataset_types="TK020Output",
        description="tk020 DS_OUT produced by exec_mid",
        version=DatasetVersion(1, 0, 0),
    )
    ds_out_rid = ds_out.dataset_rid

    # --- exec_v2: mutates D to v2 (2.0.0) ---
    # mark_dev opens a dev period; release promotes it and stamps
    # Dataset_Version.Execution = exec_v2 for the new version 2.0.0.
    # exec_v2 is a distinct execution (different RID from exec_v1 / exec_mid).
    exec_v2 = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="tk020 exec_v2 (D mutator, v2 producer)"),
    )
    exec_v2_rid = exec_v2.execution_rid

    d.mark_dev(description="tk020: exec_v2 opens dev period on D", execution=exec_v2)
    d.release(
        bump=VersionPart.major,
        description="tk020: exec_v2 releases D@2.0.0",
        execution=exec_v2,
    )

    # Sanity: D now has two released versions; the latest is produced by exec_v2.
    assert exec_v1_rid != exec_v2_rid, "exec_v1 and exec_v2 must be distinct for tk-020 invariant"

    # --- Walk lineage from DS_OUT ---
    result = test_ml.lookup_lineage(ds_out_rid)

    # Collect every execution RID anywhere in the lineage tree.
    seen: set[str] = set()

    def _collect(node):
        if node is None:
            return
        seen.add(node.execution.rid)
        for p in node.parents:
            _collect(p)

    _collect(result.lineage)

    assert exec_v1_rid in seen, f"consumed-version producer {exec_v1_rid} missing; saw {seen}"
    assert exec_v2_rid not in seen, f"latest-version producer {exec_v2_rid} leaked into lineage; saw {seen}"
    assert result.cycle_detected is False
