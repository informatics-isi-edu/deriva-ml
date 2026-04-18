"""Live-catalog integration tests for the ERMrest PagedClient adapter.

Requires DERIVA_HOST and a test catalog. Uses the `test_ml` fixture
from the top-level conftest.

Run with: pytest tests/local_db/test_paged_fetcher_live.py -v -m integration
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import Column, MetaData, String, Table, select

from deriva_ml.local_db.paged_fetcher import PagedFetcher
from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient
from deriva_ml.local_db.workspace import Workspace

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("DERIVA_HOST"),
        reason="DERIVA_HOST not set; skipping live integration tests",
    ),
]


def _make_dataset_table(engine) -> Table:
    md = MetaData()
    t = Table(
        "Dataset_cache",
        md,
        Column("RID", String, primary_key=True),
        Column("Description", String),
    )
    md.create_all(engine)
    return t


class TestErmrestPagedClientLive:
    @pytest.fixture
    def workspace_on_ml(self, test_ml):
        ws = Workspace(
            working_dir=test_ml.working_dir,
            hostname=test_ml.host_name,
            catalog_id=test_ml.catalog_id,
        )
        yield ws
        ws.close()

    def test_count_returns_nonnegative(self, test_ml, workspace_on_ml) -> None:
        client = ErmrestPagedClient(catalog=test_ml.catalog)
        total = client.count(f"{test_ml.ml_schema}:Dataset")
        assert total >= 0

    def test_fetch_predicate_writes_rows(self, test_ml, workspace_on_ml) -> None:
        target = _make_dataset_table(workspace_on_ml.engine)
        client = ErmrestPagedClient(catalog=test_ml.catalog)
        f = PagedFetcher(client=client, engine=workspace_on_ml.engine)

        n = f.fetch_predicate(
            table=f"{test_ml.ml_schema}:Dataset",
            predicate=None,
            target_table=target,
            sort=("RID",),
            page_size=50,
        )
        assert n >= 0
        with workspace_on_ml.engine.connect() as conn:
            got = conn.execute(select(target)).fetchall()
        assert len(got) == n

    def test_fetch_by_rids_roundtrip(self, test_ml, workspace_on_ml) -> None:
        target = _make_dataset_table(workspace_on_ml.engine)
        client = ErmrestPagedClient(catalog=test_ml.catalog)
        f = PagedFetcher(client=client, engine=workspace_on_ml.engine)

        f.fetch_predicate(
            table=f"{test_ml.ml_schema}:Dataset",
            predicate=None,
            target_table=target,
            sort=("RID",),
            page_size=5,
        )
        with workspace_on_ml.engine.connect() as conn:
            rids = [r[0] for r in conn.execute(select(target.c.RID))]

        if not rids:
            pytest.skip("No Dataset rows in catalog")

        # Clear and re-fetch by RID
        with workspace_on_ml.engine.begin() as conn:
            conn.execute(target.delete())

        f2 = PagedFetcher(client=client, engine=workspace_on_ml.engine)
        n = f2.fetch_by_rids(
            table=f"{test_ml.ml_schema}:Dataset",
            rids=rids[:3],
            target_table=target,
            rid_column="RID",
        )
        assert n == min(3, len(rids))
