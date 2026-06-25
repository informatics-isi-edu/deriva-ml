"""Tests for RID resolution functionality."""

import pytest

from deriva_ml.core.mixins.rid_resolution import BatchRidResult


class TestRidResolution:
    """Tests for resolve_rid and resolve_rids methods."""

    def test_resolve_rids_batch(self, catalog_with_datasets):
        """Test batch RID resolution returns correct table information."""
        ml_instance, _ = catalog_with_datasets  # Fixture returns (ml, dataset_description)

        # Get some RIDs from the catalog to test with
        # First, get dataset members which we know exist
        datasets = list(ml_instance.find_datasets())
        assert len(datasets) > 0, "Need at least one dataset for testing"

        dataset = datasets[0]
        members = dataset.list_dataset_members()

        # Collect some RIDs to resolve
        rids_to_resolve = []
        expected_tables = {}
        for table_name, records in members.items():
            for record in records[:2]:  # Take up to 2 from each table
                rid = record["RID"]
                rids_to_resolve.append(rid)
                expected_tables[rid] = table_name

        if not rids_to_resolve:
            pytest.skip("No dataset members to test with")

        # Batch resolve all RIDs
        results = ml_instance.resolve_rids(rids_to_resolve)

        # Verify all RIDs were resolved
        assert len(results) == len(rids_to_resolve)

        # Verify table assignments are correct
        for rid, result in results.items():
            assert isinstance(result, BatchRidResult)
            assert result.rid == rid
            assert result.table_name == expected_tables[rid]
            assert result.table is not None
            assert result.schema_name is not None

    def test_resolve_rids_empty(self, test_ml):
        """Test batch resolution with empty input returns empty dict."""
        ml_instance = test_ml
        results = ml_instance.resolve_rids([])
        assert results == {}

    def test_resolve_rids_invalid(self, test_ml):
        """Test batch resolution with invalid RIDs raises typed exception.

        ``resolve_rids`` raises ``DerivaMLRidsNotFound`` (a
        ``DerivaMLNotFoundError`` subclass) carrying the
        unresolved set as ``e.missing_rids`` — callers can read
        that attribute directly without string-parsing the
        message.
        """
        from deriva_ml.core.exceptions import DerivaMLRidsNotFound

        ml_instance = test_ml
        with pytest.raises(DerivaMLRidsNotFound) as exc_info:
            ml_instance.resolve_rids(["INVALID-RID-123"])
        assert exc_info.value.missing_rids == {"INVALID-RID-123"}

    def test_resolve_rids_with_candidate_tables(self, catalog_with_datasets):
        """Test batch resolution with specific candidate tables."""
        ml_instance, _ = catalog_with_datasets  # Fixture returns (ml, dataset_description)

        # Get element types to use as candidates
        element_types = list(ml_instance.list_dataset_element_types())
        if not element_types:
            pytest.skip("No dataset element types configured")

        # Get some members from datasets
        datasets = list(ml_instance.find_datasets())
        if not datasets:
            pytest.skip("No datasets available")

        dataset = datasets[0]
        members = dataset.list_dataset_members()

        # Find RIDs that belong to one of the element types
        rids_to_resolve = []
        for table in element_types:
            if table.name in members:
                for record in members[table.name][:1]:
                    rids_to_resolve.append(record["RID"])
                break

        if not rids_to_resolve:
            pytest.skip("No matching RIDs found")

        # Resolve with specific candidate tables
        results = ml_instance.resolve_rids(rids_to_resolve, candidate_tables=element_types)
        assert len(results) == len(rids_to_resolve)

    def test_resolve_rid_single(self, catalog_with_datasets):
        """Test single RID resolution still works."""
        ml_instance, _ = catalog_with_datasets  # Fixture returns (ml, dataset_description)

        # Get a dataset RID which we know exists
        datasets = list(ml_instance.find_datasets())
        assert len(datasets) > 0

        dataset_rid = datasets[0].dataset_rid

        # Single resolve
        result = ml_instance.resolve_rid(dataset_rid)
        assert result.rid == dataset_rid
        assert result.table.name == "Dataset"

    def test_resolve_rids_large_batch_chunks_under_url_limit(self, test_ml):
        """resolve_rids resolves a large RID set by chunking its queries.

        A single ``RID = Any(*rids)`` filter goes into the GET URL path, so
        for enough RIDs the request line exceeds the server's URL limit (the
        front Apache rejects it before ERMrest sees it). Pre-fix, resolve_rids
        issued ONE such query and a ``bare except: continue`` swallowed the
        failure — every RID was reported as DerivaMLRidsNotFound even though
        the rows exist. The fix chunks the query into URL-safe batches.

        We insert well over the short-RID URL boundary (~994 three-char RIDs
        ~= 4 KB) so the single-query path is guaranteed to overflow, then
        resolve them all and require every one to come back. RIDs come from
        the freshly-inserted catalog rows, never literals.
        """
        ml_instance = test_ml
        pb = ml_instance.pathBuilder()
        file_path = pb.schemas[ml_instance.ml_schema].tables["File"]

        n = 1100  # comfortably past the ~994 short-RID single-query boundary
        rows = [
            {"URL": f"tag://chunktest,2026-06-24:file:///c/f{i}.txt", "MD5": f"{i:032x}", "Length": 1}
            for i in range(n)
        ]
        inserted = list(file_path.insert(rows))
        rids = [r["RID"] for r in inserted]
        assert len(rids) == n

        # All RIDs are real File rows — every one must resolve to the File
        # table. On the pre-fix single-query path this raises
        # DerivaMLRidsNotFound (the oversized URL fails and is swallowed).
        candidate = [ml_instance.model.name_to_table("File")]
        results = ml_instance.resolve_rids(rids, candidate_tables=candidate)
        assert len(results) == n
        assert all(r.table_name == "File" for r in results.values())
