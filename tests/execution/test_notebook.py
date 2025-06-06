"""
Tests for notebook execution functionality.
"""

import os
import pytest

from deriva_ml import ExecutionConfiguration, MLVocab, DatasetSpec


def test_notebook_execution(ml_instance):
    """Test notebook execution workflow."""
    # Add required terms
    ml_instance.add_term(
        MLVocab.workflow_type, 
        "Manual Workflow", 
        description="Initial setup of Model File"
    )
    ml_instance.add_term(
        MLVocab.asset_type, 
        "API_Model", 
        description="Model for our API workflow"
    )

    # Create workflow
    api_workflow = ml_instance.create_workflow(
        name="Manual Workflow",
        workflow_type="Manual Workflow",
        description="A manual operation"
    )

    # Test execution with empty parameters
    manual_execution = ml_instance.create_execution(
        ExecutionConfiguration(
            description="Sample Execution",
            workflow=api_workflow,
            datasets=[],
            assets=[],
            parameters=None
        )
    )

    # Verify execution state
    assert manual_execution.parameters is None
    assert len(manual_execution.datasets) == 0
    assert len(manual_execution.asset_paths) == 0

    # Test output upload
    results = manual_execution.upload_execution_outputs()
    assert isinstance(results, dict) 