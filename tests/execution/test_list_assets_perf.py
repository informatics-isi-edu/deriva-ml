"""Tests for the ``list_assets`` perf + error-surfacing fix (audit P1 Ex-listw).

Pre-fix, ``list_assets`` walked every table in every schema on
every call (a 200-table catalog → 400 catalog touches per call)
and wrapped the per-table query in a bare ``except Exception``
that silently turned catalog connectivity errors into "no
assets".

Post-fix:

1. **Cache** the ``*_Execution`` table discovery on the
   :class:`DerivaModel` instance via
   :meth:`DerivaModel.find_asset_execution_tables`. Repeated
   ``list_assets`` calls re-use the cached list.

2. **Drop the outer bare-except** in
   :func:`deriva_ml.execution._helpers.list_assets`. Per-row
   ``lookup_asset`` failures still swallow (single bad row
   shouldn't break the whole listing); outer catalog errors
   propagate.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import MagicMock

import pytest

from deriva_ml.execution import _helpers
from deriva_ml.model.catalog import DerivaModel


class TestFindAssetExecutionTablesCaching:
    """``DerivaModel.find_asset_execution_tables`` discovers + caches."""

    def _build_fake_model(self, *, domain_schemas: list[str], ml_schema: str, tables_by_schema: dict):
        """Build a MagicMock that quacks like ``DerivaModel`` for cache tests.

        ``tables_by_schema`` is a ``{schema_name: [table_name, ...]}``
        mapping. Each table name becomes a stub with ``.name``.
        """
        model = MagicMock(spec=DerivaModel)
        model.domain_schemas = frozenset(domain_schemas)
        model.ml_schema = ml_schema
        model.model = MagicMock()
        model.model.schemas = {}
        for schema_name, table_names in tables_by_schema.items():
            schema = MagicMock()
            schema.tables = {}
            for tname in table_names:
                t = MagicMock()
                t.name = tname
                schema.tables[tname] = t
            model.model.schemas[schema_name] = schema
        # Cache attribute must not pre-exist.
        if hasattr(model, "_asset_execution_tables_cache"):
            del model._asset_execution_tables_cache
        return model

    def test_finds_execution_association_tables_excludes_non_asset_tables(self):
        """``*_Execution`` tables yielded; ``Dataset_Execution`` and
        ``Execution_Execution`` excluded.

        Both excluded tables end in ``_Execution`` but neither is an
        asset-to-execution association table — sweeping them into the
        ``list_assets(asset_role=...)`` walk would either return zero
        matches (Dataset_Execution; harmless) or crash with
        ``AttributeError`` on the missing ``Asset_Role`` column
        (Execution_Execution; correctness-critical).
        """
        model = self._build_fake_model(
            domain_schemas=["my_domain"],
            ml_schema="deriva-ml",
            tables_by_schema={
                "my_domain": ["Image", "Image_Execution"],
                "deriva-ml": [
                    "Execution",
                    "Execution_Asset",
                    "Execution_Asset_Execution",
                    "Execution_Execution",  # nested-execution; must be excluded
                    "Dataset",
                    "Dataset_Execution",  # dataset linkage; must be excluded
                ],
            },
        )
        result = DerivaModel.find_asset_execution_tables(model)
        names = {t for _, t in result}
        assert "Image_Execution" in names
        assert "Execution_Asset_Execution" in names
        # Dataset_Execution is the dataset linkage, not an asset linkage.
        assert "Dataset_Execution" not in names
        # Execution_Execution is the nested-execution hierarchy table
        # — no Asset_Role column. Including it crashes list_assets.
        assert "Execution_Execution" not in names
        # Pure data tables (no _Execution suffix) excluded too.
        assert "Image" not in names
        assert "Execution" not in names

    def test_result_includes_schema_pair(self):
        """Each entry is ``(schema_name, table_name)``."""
        model = self._build_fake_model(
            domain_schemas=["my_domain"],
            ml_schema="deriva-ml",
            tables_by_schema={
                "my_domain": ["Image_Execution"],
                "deriva-ml": [],
            },
        )
        result = DerivaModel.find_asset_execution_tables(model)
        assert ("my_domain", "Image_Execution") in result

    def test_caches_on_instance_under_same_model(self):
        """Second call returns the same list object (cache hit)."""
        model = self._build_fake_model(
            domain_schemas=["my_domain"],
            ml_schema="deriva-ml",
            tables_by_schema={
                "my_domain": ["Image_Execution"],
                "deriva-ml": [],
            },
        )
        first = DerivaModel.find_asset_execution_tables(model)
        second = DerivaModel.find_asset_execution_tables(model)
        # ``is`` — identical object, not just equal.
        assert first is second

    def test_cache_invalidates_when_model_swapped(self):
        """Replacing ``self.model`` invalidates the cache.

        Long-lived sessions can refetch the model (e.g. after a
        schema change). The cache key is the model object's
        identity; swapping it forces a re-walk.
        """
        model = self._build_fake_model(
            domain_schemas=["my_domain"],
            ml_schema="deriva-ml",
            tables_by_schema={"my_domain": ["Image_Execution"], "deriva-ml": []},
        )
        first = DerivaModel.find_asset_execution_tables(model)

        # Swap to a fresh underlying model with a different table set.
        new_model_obj = MagicMock()
        new_model_obj.schemas = {}
        new_schema = MagicMock()
        new_table = MagicMock()
        new_table.name = "Subject_Execution"
        new_schema.tables = {"Subject_Execution": new_table}
        new_model_obj.schemas = {"my_domain": new_schema, "deriva-ml": MagicMock(tables={})}
        model.model = new_model_obj

        second = DerivaModel.find_asset_execution_tables(model)
        assert second is not first
        names = {t for _, t in second}
        assert names == {"Subject_Execution"}

    def test_missing_schema_in_model_is_skipped(self):
        """A schema in ``domain_schemas`` that doesn't exist in the model is skipped.

        Defensive — should never happen in a healthy catalog,
        but the cache builder shouldn't crash on a stale
        domain_schemas set.
        """
        model = self._build_fake_model(
            domain_schemas=["my_domain", "ghost_schema"],
            ml_schema="deriva-ml",
            tables_by_schema={
                "my_domain": ["Image_Execution"],
                "deriva-ml": [],
                # ``ghost_schema`` deliberately absent from tables_by_schema
            },
        )
        # Should not raise.
        result = DerivaModel.find_asset_execution_tables(model)
        names = {t for _, t in result}
        assert names == {"Image_Execution"}


class TestListAssetsErrorSurfacing:
    """``_helpers.list_assets`` propagates outer catalog errors."""

    def test_no_bare_except_around_outer_query(self):
        """Source-level pin: the outer fetch is not wrapped in ``except Exception``.

        Pre-fix, a bare ``except Exception`` swallowed catalog
        connectivity errors as "no assets". The current
        implementation only swallows the inner
        ``lookup_asset`` per-row call. Pin this by source
        inspection — a future regression that re-introduces
        an outer bare-except fails immediately.
        """
        src = inspect.getsource(_helpers.list_assets)
        # Strip docstrings AND comments — both can mention
        # "except Exception" as commentary about the history of
        # this code; we only care about actual ``except`` statements.
        body = re.sub(r'"""[\s\S]*?"""', "", src, count=1)
        body = re.sub(r"#[^\n]*\n", "\n", body)
        # Match actual except clauses (start of statement, with
        # indentation), not bare textual occurrences.
        count = len(re.findall(r"^\s*except\s+Exception", body, re.MULTILINE))
        assert count == 1, (
            f"list_assets should have exactly one ``except Exception`` "
            f"clause (the per-row lookup_asset swallow). Found {count}. "
            f"A regression has re-introduced the outer-query bare-except "
            f"that silently turns catalog connectivity errors into 'no assets'."
        )

    def _table_path_mock(self, ml):
        """Drill into the path-builder chain mocks to the table_path."""
        return ml.pathBuilder.return_value.schemas.__getitem__.return_value.tables.__getitem__.return_value

    def test_outer_query_failure_propagates(self):
        """Live behavioural pin: a catalog fetch failure raises, not returns []."""
        ml = MagicMock()
        ml.model.find_asset_execution_tables.return_value = [("my_domain", "Image_Execution")]
        # Make ``query.entities().fetch()`` raise.
        table_path = self._table_path_mock(ml)
        table_path.filter.return_value.entities.return_value.fetch.side_effect = ConnectionError("catalog down")

        with pytest.raises(ConnectionError, match="catalog down"):
            _helpers.list_assets(ml_instance=ml, execution_rid="exec-1")

    def test_per_row_lookup_failure_is_swallowed(self):
        """Per-row swallow still applies: one bad row doesn't kill the listing."""
        ml = MagicMock()
        ml.model.find_asset_execution_tables.return_value = [("my_domain", "Image_Execution")]
        # Outer query succeeds, returns two rows.
        records = [
            {"Image": "asset-good"},
            {"Image": "asset-bad"},
        ]
        table_path = self._table_path_mock(ml)
        table_path.filter.return_value.entities.return_value.fetch.return_value = records

        # ``lookup_asset`` succeeds for good, raises for bad.
        def lookup_side_effect(rid):
            if rid == "asset-bad":
                raise ValueError("bad asset row")
            return f"<Asset {rid}>"

        ml.lookup_asset.side_effect = lookup_side_effect

        result = _helpers.list_assets(ml_instance=ml, execution_rid="exec-1")
        # Good row survives; bad row swallowed.
        assert result == ["<Asset asset-good>"]
