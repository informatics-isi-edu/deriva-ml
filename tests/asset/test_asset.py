"""Tests for the Asset class.

This module provides tests for DerivaML Asset functionality:

Test Classes:
    TestAssetLookup: Asset lookup and retrieval operations
    TestAssetTypes: Asset type management
    TestAssetExecutions: Asset-execution relationship tracking
"""

import pytest

from deriva_ml import DerivaML, ExecAssetType, MLAsset
from deriva_ml import MLVocab as vc
from deriva_ml.asset.asset import Asset
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution.execution import ExecutionConfiguration


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def workflow_terms(test_ml):
    """Add required vocabulary terms for workflow testing."""
    test_ml.add_term(vc.asset_type, "Test Model", description="Model for our Test workflow")
    test_ml.add_term(vc.workflow_type, "Test Workflow", description="A ML Workflow that uses Deriva ML API")
    return test_ml


@pytest.fixture
def test_workflow(workflow_terms):
    """Create a test workflow."""
    ml = workflow_terms
    return ml.create_workflow(
        name="Test Workflow",
        workflow_type="Test Workflow",
        description="A test workflow for asset testing",
    )


@pytest.fixture
def basic_execution(workflow_terms, test_workflow):
    """Create a basic execution without datasets."""
    ml = workflow_terms
    config = ExecutionConfiguration(
        description="Test Execution",
        workflow=test_workflow,
    )
    return ml.create_execution(config)


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_asset(execution, filename="test_asset.txt", content="Test content"):
    """Helper to create a test asset file within an execution."""
    asset_path = execution.asset_file_path(
        MLAsset.execution_asset,
        f"TestAsset/{filename}",
        asset_types=ExecAssetType.model_file,
    )
    with asset_path.open("w") as fp:
        fp.write(content)
    return asset_path


# =============================================================================
# TestAssetLookup - Asset lookup and retrieval
# =============================================================================


class TestAssetLookup:
    """Tests for asset lookup operations."""

    def test_lookup_asset(self, basic_execution):
        """Test looking up an asset by RID."""
        ml = basic_execution._ml_object

        # Create and upload an asset
        with basic_execution.execute() as execution:
            create_test_asset(execution, "lookup_test.txt", "Lookup test content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Look up the asset
        asset = ml.lookup_asset(asset_rid)

        assert asset is not None
        assert isinstance(asset, Asset)
        assert asset.asset_rid == asset_rid
        assert asset.asset_table == "Execution_Asset"
        assert asset.filename == "lookup_test.txt"
        assert "Model_File" in asset.asset_types

    def test_lookup_asset_not_found(self, test_ml):
        """Test that looking up a non-existent asset raises an exception."""
        with pytest.raises(DerivaMLException):
            test_ml.lookup_asset("NONEXISTENT")

    def test_lookup_asset_not_asset_table(self, test_ml):
        """Test that looking up a non-asset RID raises an exception."""
        # Add a vocabulary term (not an asset)
        term = test_ml.add_term(vc.workflow_type, "Test", description="Test")

        with pytest.raises(DerivaMLException, match="not an asset"):
            test_ml.lookup_asset(term.rid)

    def test_find_assets(self, basic_execution):
        """Test finding assets in the catalog."""
        ml = basic_execution._ml_object

        # Create and upload multiple assets
        with basic_execution.execute() as execution:
            create_test_asset(execution, "find_test_1.txt", "Content 1")
            create_test_asset(execution, "find_test_2.txt", "Content 2")

        basic_execution.upload_execution_outputs()

        # Find all assets in Execution_Asset table
        assets = list(ml.find_assets(asset_table="Execution_Asset"))

        assert len(assets) >= 2
        assert all(isinstance(a, Asset) for a in assets)
        assert all(a.asset_table == "Execution_Asset" for a in assets)

    def test_find_assets_by_type(self, basic_execution):
        """Test finding assets filtered by type."""
        ml = basic_execution._ml_object

        # Create and upload an asset with specific type
        with basic_execution.execute() as execution:
            create_test_asset(execution, "typed_asset.txt", "Typed content")

        basic_execution.upload_execution_outputs()

        # Find assets with the Model_File type
        assets = list(ml.find_assets(asset_type="Model_File"))

        assert len(assets) >= 1
        assert all("Model_File" in a.asset_types for a in assets)


# =============================================================================
# TestAssetTypes - Asset type management
# =============================================================================


class TestAssetTypes:
    """Tests for asset type management."""

    def test_asset_types_property(self, basic_execution):
        """Test that asset types are correctly retrieved."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            create_test_asset(execution, "types_test.txt", "Types content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)

        assert isinstance(asset.asset_types, list)
        assert "Model_File" in asset.asset_types

    def test_add_asset_type(self, basic_execution):
        """Test adding a type to an asset."""
        ml = basic_execution._ml_object

        # Create a new asset type
        ml.add_term(vc.asset_type, "New_Type", description="A new asset type")

        with basic_execution.execute() as execution:
            create_test_asset(execution, "add_type_test.txt", "Content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)

        # Add a new type
        asset.add_asset_type("New_Type")

        # Verify type was added
        refreshed = ml.lookup_asset(asset_rid)
        assert "New_Type" in refreshed.asset_types
        assert "Model_File" in refreshed.asset_types  # Original type still there

    def test_remove_asset_type(self, basic_execution):
        """Test removing a type from an asset."""
        ml = basic_execution._ml_object

        # Create a new asset type and add it
        ml.add_term(vc.asset_type, "Remove_Type", description="Type to remove")

        with basic_execution.execute() as execution:
            create_test_asset(execution, "remove_type_test.txt", "Content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)

        # Add then remove a type
        asset.add_asset_type("Remove_Type")
        assert "Remove_Type" in asset.asset_types

        asset.remove_asset_type("Remove_Type")

        # Verify type was removed
        refreshed = ml.lookup_asset(asset_rid)
        assert "Remove_Type" not in refreshed.asset_types


# =============================================================================
# TestAssetExecutions - Asset-execution relationship tracking
# =============================================================================


class TestAssetExecutions:
    """Tests for asset-execution relationship tracking."""

    def test_list_executions_output(self, basic_execution):
        """Test listing executions that created an asset."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            create_test_asset(execution, "exec_test.txt", "Content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)

        # List executions with Output role - now returns ExecutionRecord objects
        executions = asset.list_executions(asset_role="Output")

        assert len(executions) == 1
        assert executions[0].execution_rid == basic_execution.execution_rid

    def test_list_executions_input(self, workflow_terms, test_workflow, basic_execution):
        """Test listing executions that used an asset as input."""
        ml = workflow_terms

        # Create and upload an asset
        with basic_execution.execute() as execution:
            create_test_asset(execution, "input_test.txt", "Content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Create a new execution that uses this asset as input
        config = ExecutionConfiguration(
            description="Input Test Execution",
            workflow=test_workflow,
            assets=[asset_rid],
        )
        input_execution = ml.create_execution(config)

        asset = ml.lookup_asset(asset_rid)

        # List all executions - now returns ExecutionRecord objects
        all_executions = asset.list_executions()
        assert len(all_executions) == 2

        # List only input executions
        input_executions = asset.list_executions(asset_role="Input")
        assert len(input_executions) == 1
        assert input_executions[0].execution_rid == input_execution.execution_rid

    def test_execution_rid_property(self, basic_execution):
        """Test that execution_rid property returns the creating execution."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            create_test_asset(execution, "execution_rid_test.txt", "Content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)

        # The execution_rid property should return the creating execution
        assert asset.execution_rid == basic_execution.execution_rid


# =============================================================================
# TestAssetMetadata - Asset metadata operations
# =============================================================================


class TestAssetMetadata:
    """Tests for asset metadata operations."""

    def test_get_metadata(self, basic_execution):
        """Test retrieving all metadata for an asset."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            create_test_asset(execution, "metadata_test.txt", "Metadata content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)
        metadata = asset.get_metadata()

        assert "RID" in metadata
        assert "Filename" in metadata
        assert "URL" in metadata
        assert metadata["Filename"] == "metadata_test.txt"

    def test_get_chaise_url(self, basic_execution):
        """Test generating Chaise URL for an asset."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            create_test_asset(execution, "chaise_test.txt", "Chaise content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)
        url = asset.get_chaise_url()

        assert "chaise/record" in url
        assert asset_rid in url
        assert "Execution_Asset" in url


# =============================================================================
# TestAssetRepr - Asset representation
# =============================================================================


class TestAssetRepr:
    """Tests for asset string representation."""

    def test_repr(self, basic_execution):
        """Test that Asset has a useful repr."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            create_test_asset(execution, "repr_test.txt", "Repr content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        asset = ml.lookup_asset(asset_rid)
        repr_str = repr(asset)

        assert "deriva_ml.Asset" in repr_str
        assert asset_rid in repr_str
        assert "Execution_Asset" in repr_str
