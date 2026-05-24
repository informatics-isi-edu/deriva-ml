"""End-to-end test of the directional Asset_Role contract.

The public-API contract (see ``docs/user-guide/executions.md`` —
"How execution-asset roles work"):

    Every asset associated with an execution carries either an
    ``Input_File`` or ``Output_File`` directional ``Asset_Type``
    tag AND its ``{Asset}_Execution`` row has the matching
    ``Asset_Role`` ("Input" or "Output").

    deriva-ml assigns the role and tag; callers don't pass them.

Unit-level coverage in ``test_asset_role_auto_tag.py`` pins the
inner helpers in isolation. This file exercises the **full
public API** against a live catalog:

- ``download_asset`` → role "Input" + ``Input_File`` tag.
- ``asset_file_path`` + ``commit_output_assets`` → role
  "Output" + ``Output_File`` tag.

The motivation is a regression that lived undetected for many
months: ``bag_commit._add_asset_rows_to_bag`` wrote
``Asset_Role="Output"`` correctly but **never** auto-added the
``Output_File`` Asset_Type tag. The unit tests pinned the
*reference* implementation in ``update_asset_execution_table``;
no test exercised the *production* upload path end-to-end. This
file closes that gap so a future regression in the bag-commit
path is caught immediately.

Live-catalog integration tests; require ``DERIVA_HOST``.
"""

from __future__ import annotations

import pytest

from deriva_ml import ExecAssetType, MLAsset
from deriva_ml.execution.execution import ExecutionConfiguration


@pytest.fixture
def test_workflow(workflow_terms):
    """Workflow used for the role-contract tests."""
    ml = workflow_terms
    return ml.create_workflow(
        name="Asset Role Contract Test Workflow",
        workflow_type="Test Workflow",
        description="Workflow exercising the Input/Output role contract",
    )


@pytest.fixture
def basic_execution(workflow_terms, test_workflow):
    """Fresh execution without preregistered datasets/assets."""
    ml = workflow_terms
    config = ExecutionConfiguration(
        description="Asset Role Contract Test Execution",
        workflow=test_workflow,
    )
    return ml.create_execution(config)


def _asset_type_set(asset) -> set[str]:
    """Return the asset's Asset_Type tags as a set for membership assertions."""
    return set(asset.asset_types)


class TestOutputAssetRoleContract:
    """Assets uploaded via ``asset_file_path`` + ``commit_output_assets``
    must carry ``Asset_Role="Output"`` + ``Output_File`` tag.
    """

    def test_uploaded_output_carries_output_role_and_output_file_tag(self, basic_execution):
        """End-to-end: upload one asset; verify both Output side of the contract.

        This is the regression test for the
        bag-commit-missing-Output_File bug. Pre-fix the asset
        landed in the catalog with ``asset_types = ["Model_File"]``
        (just the user-supplied content tag, missing the
        directional ``Output_File``). Post-fix the catalog row
        has both.
        """
        ml = basic_execution._ml_object

        # Stage a single output asset, mark it as a model file
        # (a content tag, NOT the directional tag).
        with basic_execution.execute() as execution:
            asset_path = execution.asset_file_path(
                MLAsset.execution_asset,
                "RoleContract/output.txt",
                asset_types=ExecAssetType.model_file.value,
            )
            with asset_path.open("w") as fp:
                fp.write("output content")

        uploaded_report = basic_execution.commit_output_assets()
        uploaded = basic_execution.uploaded_assets
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Look up the resulting catalog state and assert BOTH halves
        # of the directional contract.
        asset = ml.lookup_asset(asset_rid)
        tags = _asset_type_set(asset)

        # 1. The user-supplied content tag is preserved.
        assert ExecAssetType.model_file.value in tags, (
            f"Content tag {ExecAssetType.model_file.value!r} missing from uploaded asset's tags: {sorted(tags)}"
        )

        # 2. The directional Output_File tag is auto-added.
        # This is the bug fix: pre-fix this assertion would fail
        # because the bag-commit path never added Output_File.
        assert ExecAssetType.output_file.value in tags, (
            f"Output_File directional tag missing from uploaded asset's tags: "
            f"{sorted(tags)}. "
            f"Per the contract (docs/user-guide/executions.md, 'How "
            f"execution-asset roles work'), every uploaded execution "
            f"asset must carry Output_File. If this fails, the bag-commit "
            f"path's directional-tag auto-add regressed."
        )

        # 3. The {Asset}_Execution row has Asset_Role="Output".
        output_executions = asset.list_executions(asset_role="Output")
        assert len(output_executions) == 1
        assert output_executions[0].execution_rid == basic_execution.execution_rid

        # 4. Negative: this asset did NOT get Input_File or Input role.
        assert ExecAssetType.input_file.value not in tags
        assert asset.list_executions(asset_role="Input") == []

    def test_upload_without_content_tag_still_gets_output_file(self, basic_execution):
        """An upload that passes ``asset_types=[]`` still gets ``Output_File``.

        Pins the "every execution asset gets a directional tag"
        rule for the zero-content-tag case. Pre-fix this would
        produce an asset with empty tags (no content, no
        directional); post-fix the directional tag is the floor.
        """
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            # asset_types=None defaults to [asset_name]; we use an
            # explicit minimal pass to exercise the auto-add path
            # without a user-supplied content tag conflicting.
            asset_path = execution.asset_file_path(
                MLAsset.execution_asset,
                "RoleContract/notags.txt",
            )
            with asset_path.open("w") as fp:
                fp.write("notags")

        uploaded_report = basic_execution.commit_output_assets()
        uploaded = basic_execution.uploaded_assets
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)
        tags = _asset_type_set(asset)

        # The default-asset_types-to-asset_name behaviour means
        # ``Execution_Asset`` ends up as the content tag — that's
        # an existing contract independent of this fix. What this
        # test pins is that Output_File is *also* there alongside
        # whatever the default content tag was.
        assert ExecAssetType.output_file.value in tags, (
            f"Output_File missing from default-asset-types upload: {sorted(tags)}"
        )

    def test_explicit_output_file_tag_not_duplicated(self, basic_execution):
        """When the caller explicitly passes ``Output_File``, the auto-add
        must not produce a duplicate row."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            asset_path = execution.asset_file_path(
                MLAsset.execution_asset,
                "RoleContract/explicit.txt",
                asset_types=[
                    ExecAssetType.model_file.value,
                    ExecAssetType.output_file.value,  # caller passes it explicitly
                ],
            )
            with asset_path.open("w") as fp:
                fp.write("explicit")

        uploaded_report = basic_execution.commit_output_assets()
        uploaded = basic_execution.uploaded_assets
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)
        tags = list(asset.asset_types)

        # Output_File appears exactly once — no duplicate row.
        output_file_count = sum(1 for t in tags if t == ExecAssetType.output_file.value)
        assert output_file_count == 1, (
            f"Output_File appeared {output_file_count} times in {tags}; explicit-pass-through should be deduplicated."
        )
        # Other content tags preserved.
        assert ExecAssetType.model_file.value in tags


class TestInputAssetRoleContract:
    """Assets downloaded via ``download_asset`` must carry
    ``Asset_Role="Input"`` + ``Input_File`` tag.
    """

    def test_downloaded_input_carries_input_role_and_input_file_tag(self, workflow_terms, test_workflow, tmp_path):
        """End-to-end: create-then-download to verify the Input contract.

        Setup: a first execution uploads an asset (which by the
        contract gets Output_File). A second execution downloads
        that asset — which by the contract should add the Input
        role + Input_File tag, without removing the prior
        Output_File tag.
        """
        ml = workflow_terms

        # First execution: create the asset as an Output.
        creator_config = ExecutionConfiguration(
            description="Role contract — creator execution",
            workflow=test_workflow,
        )
        creator = ml.create_execution(creator_config)
        with creator.execute() as exe:
            asset_path = exe.asset_file_path(
                MLAsset.execution_asset,
                "RoleContract/cross-exec.txt",
                asset_types=ExecAssetType.model_file.value,
            )
            with asset_path.open("w") as fp:
                fp.write("cross-execution test")

        uploaded_report = creator.commit_output_assets()
        uploaded = creator.uploaded_assets
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Sanity: creator's upload tagged it as an Output.
        asset_after_create = ml.lookup_asset(asset_rid)
        assert ExecAssetType.output_file.value in asset_after_create.asset_types
        assert ExecAssetType.model_file.value in asset_after_create.asset_types

        # Second execution: consume the asset as an Input.
        consumer_config = ExecutionConfiguration(
            description="Role contract — consumer execution",
            workflow=test_workflow,
            assets=[asset_rid],
        )
        consumer = ml.create_execution(consumer_config)

        # ``create_execution`` already downloads input assets and
        # writes the Input_File tag. Look up the asset and confirm.
        asset_after_consume = ml.lookup_asset(asset_rid)
        tags = _asset_type_set(asset_after_consume)

        # 1. Input_File tag added.
        assert ExecAssetType.input_file.value in tags, f"Input_File missing from consumed asset's tags: {sorted(tags)}"

        # 2. Prior Output_File tag preserved (Input doesn't overwrite
        # content tags — they're additive).
        assert ExecAssetType.output_file.value in tags, (
            f"Output_File should be preserved across consumer download; got: {sorted(tags)}"
        )

        # 3. The asset is now linked to BOTH executions with their
        # respective roles.
        outputs = asset_after_consume.list_executions(asset_role="Output")
        inputs = asset_after_consume.list_executions(asset_role="Input")
        assert len(outputs) == 1
        assert outputs[0].execution_rid == creator.execution_rid
        assert len(inputs) == 1
        assert inputs[0].execution_rid == consumer.execution_rid


class TestRoleSymmetry:
    """Pin the symmetric Input/Output public-API behaviour
    documented in ``docs/user-guide/executions.md``.
    """

    def test_list_assets_by_role_returns_correct_partitions(self, workflow_terms, test_workflow):
        """``exe.list_assets(asset_role=...)`` returns the right partition
        of the execution's assets.

        Pre-fix the Output_File tag was missing on outputs but
        the ``{Asset}_Execution`` row still had
        ``Asset_Role="Output"``, so this call would have
        partially worked. Post-fix both layers (Asset_Role on
        the Execution row AND directional Asset_Type tag) are
        consistent.
        """
        ml = workflow_terms

        # Create an asset (acts as Output for execution A,
        # Input for execution B).
        creator_config = ExecutionConfiguration(
            description="symmetry-creator",
            workflow=test_workflow,
        )
        creator = ml.create_execution(creator_config)
        with creator.execute() as exe:
            asset_path = exe.asset_file_path(
                MLAsset.execution_asset,
                "Symmetry/shared.txt",
                asset_types=ExecAssetType.model_file.value,
            )
            with asset_path.open("w") as fp:
                fp.write("shared")
        uploaded_report = creator.commit_output_assets()
        uploaded = creator.uploaded_assets
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Creator: listing by Output role should include this asset.
        creator_outputs = creator.list_assets(asset_role="Output")
        output_rids = {a.asset_rid for a in creator_outputs}
        assert asset_rid in output_rids, f"Output listing missing asset {asset_rid}; got {output_rids}"
        creator_inputs = creator.list_assets(asset_role="Input")
        input_rids = {a.asset_rid for a in creator_inputs}
        assert asset_rid not in input_rids

        # Consumer: same asset, opposite role.
        consumer_config = ExecutionConfiguration(
            description="symmetry-consumer",
            workflow=test_workflow,
            assets=[asset_rid],
        )
        consumer = ml.create_execution(consumer_config)
        consumer_inputs = consumer.list_assets(asset_role="Input")
        c_input_rids = {a.asset_rid for a in consumer_inputs}
        assert asset_rid in c_input_rids, f"Consumer's Input listing missing asset {asset_rid}; got {c_input_rids}"
        consumer_outputs = consumer.list_assets(asset_role="Output")
        c_output_rids = {a.asset_rid for a in consumer_outputs}
        assert asset_rid not in c_output_rids
