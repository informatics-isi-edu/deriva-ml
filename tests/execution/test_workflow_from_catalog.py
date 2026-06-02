"""Tests for Workflow reconstruction from a catalog row (the read path).

When a Workflow is rebuilt from an existing catalog row (it carries a
``workflow_rid``), its provenance fields are already known and must NOT be
re-derived from the runtime environment. Re-deriving previously crashed in
non-Docker, non-git environments::

    LookupError: setuptools-scm was unable to detect version for /app.

(seen when deriva-ml-mcp-plugin read Execution/Workflow rows on a server
whose working directory is not a git checkout). The ``setup_url_checksum``
validator must short-circuit for catalog-loaded workflows.
"""

import pytest

from deriva_ml.execution.workflow import Workflow


@pytest.fixture
def non_docker_non_git_cwd(tmp_path, monkeypatch):
    """Force the worst-case env: no Docker/notebook escape, CWD not a git repo.

    This is exactly the condition under which the create-time provenance
    derivation would call setuptools-scm and crash.
    """
    # Remove the env-var escape hatches that would otherwise short-circuit
    # the validator before it reaches git/setuptools-scm introspection.
    for var in (
        "DERIVA_MCP_IN_DOCKER",
        "DERIVA_ML_WORKFLOW_URL",
        "DERIVA_ML_WORKFLOW_CHECKSUM",
        "DERIVA_ML_NOTEBOOK_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    # A directory with no .git, so Path.cwd() / setuptools-scm has no repo.
    workdir = tmp_path / "no_git_here"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    assert not (workdir / ".git").exists()
    return workdir


def test_workflow_from_catalog_row_skips_derivation(non_docker_non_git_cwd):
    """A Workflow loaded from a catalog row builds without re-deriving provenance.

    Mirrors how ``lookup_workflow`` / ``find_workflows`` reconstruct a row:
    ``workflow_rid`` is set and ``url`` / ``checksum`` come from the row. Even
    with a blank stored ``version`` and no git/Docker environment, this must
    not raise -- and the stored fields must be preserved verbatim.
    """
    workflow = Workflow(
        name="Reconstructed",
        url="https://github.com/org/repo/blob/abc123/train.py",
        workflow_type=[],
        version="",  # blank stored version -- the worst case
        description="loaded from catalog",
        workflow_rid="1-ABCD",
        checksum="abc123",
    )

    # Stored provenance preserved, not re-derived.
    assert workflow.url == "https://github.com/org/repo/blob/abc123/train.py"
    assert workflow.checksum == "abc123"
    assert workflow.version == ""
    assert workflow.workflow_rid == "1-ABCD"


def test_fresh_workflow_still_derives(non_docker_non_git_cwd):
    """A Workflow with no workflow_rid (a fresh create) is NOT short-circuited.

    The guard must key on 'loaded from catalog' (workflow_rid set), so the
    create path is unaffected. With no git repo and no Docker env, deriving
    provenance for a fresh workflow still fails -- proving the guard did not
    wrongly swallow the create path.
    """
    with pytest.raises(Exception):  # noqa: B017 - git/setuptools-scm derivation failure
        Workflow(
            name="Fresh",
            workflow_type=[],
            description="not yet in catalog",
        )
