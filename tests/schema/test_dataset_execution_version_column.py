"""Fresh-catalog schema: Dataset_Execution carries a nullable Dataset_Version FK.

The "authorship-canonical" provenance model makes ``Dataset_Execution`` the
*input-only* edge table (output edges live in ``Dataset_Version.Execution``).
To record *which* version of a dataset an execution consumed, the input edge
gains a nullable ``Dataset_Version`` FK. This test pins that the fresh-catalog
schema produced by ``create_ml_catalog`` actually carries that column + FK.

Pattern follows ``test_vocab_fk_convention.py``: create a fresh catalog,
introspect the live ``deriva-ml`` schema, delete on teardown. Requires
DERIVA_HOST.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_dataset_execution_has_dataset_version_fk(demo_model, ml_schema):
    """Dataset_Execution has a nullable Dataset_Version column FK'd to Dataset_Version."""
    de = demo_model.schemas[ml_schema].tables["Dataset_Execution"]
    cols = {c.name for c in de.columns}
    assert "Dataset_Version" in cols, "Dataset_Execution must have a Dataset_Version column"

    dv_col = next(c for c in de.columns if c.name == "Dataset_Version")
    assert dv_col.nullok is True, "Dataset_Version must be nullable"

    # deriva-py FK objects expose ``foreign_key_columns`` (the referencing
    # columns on this table) and ``referenced_columns`` (the target columns),
    # both as lists of Column instances with ``.name`` and ``.table.name``.
    fk_targets = {
        fk.referenced_columns[0].table.name
        for fk in de.foreign_keys
        if any(c.name == "Dataset_Version" for c in fk.foreign_key_columns)
    }
    assert "Dataset_Version" in fk_targets, (
        "Dataset_Execution.Dataset_Version must FK to the Dataset_Version table"
    )
