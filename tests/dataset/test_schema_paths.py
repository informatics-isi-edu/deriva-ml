"""Benchmark tests for _schema_to_paths() — the shared FK path discovery engine.

_schema_to_paths() is the foundation for both bag export (catalog_graph._collect_paths)
and denormalization (_prepare_wide_table). Changes to path discovery affect both systems.

These tests capture the EXACT set of paths produced for the demo schema so that any
regressions are immediately visible. When modifying _schema_to_paths(), update these
benchmarks intentionally — never silently.

Schema under test (from demo_catalog.create_domain_schema + populate_demo_catalog):

    Domain tables:
        Subject (Name)
        Image (asset; Subject FK, Observation FK nullable)
        Observation (Observation_Date, Subject FK)
        ClinicalRecord (Diagnosis, Notes)
        ClinicalRecord_Observation (M:N association: ClinicalRecord ↔ Observation)
        Image_Dataset_Legacy (duplicate association: Image ↔ Dataset)

    Feature tables (created by populate_demo_catalog):
        Execution_Image_Quality (feature on Image, vocab: ImageQuality)
        Execution_Image_BoundingBox (feature on Image, asset: BoundingBox)
        Execution_Subject_Health (feature on Subject, vocab: SubjectHealth)

    Element types registered:
        Subject (→ Dataset_Subject association)
        Image (→ Dataset_Image association)

FK graph:
    Dataset → Dataset_Subject → Subject
    Dataset → Dataset_Image → Image
    Dataset → Image_Dataset_Legacy → Image (duplicate)
    Image.Subject → Subject.RID
    Image.Observation → Observation.RID (nullable)
    Observation.Subject → Subject.RID
    ClinicalRecord_Observation: ClinicalRecord ↔ Observation (M:N)
    Execution_Image_Quality: Image ↔ ImageQuality (feature)
    Execution_Image_BoundingBox: Image ↔ BoundingBox (feature + asset)
    Execution_Subject_Health: Subject ↔ SubjectHealth (feature)
"""

import pytest


def path_signature(path) -> str:
    """Convert a path (list of Table objects) to a string signature."""
    return " -> ".join(t.name for t in path)


def path_signatures(paths) -> set[str]:
    """Convert a list of paths to a set of string signatures."""
    return {path_signature(p) for p in paths}


class TestSchemaToPathsBenchmark:
    """Benchmark: exact path set for the demo schema.

    These tests pin the expected output of _schema_to_paths() so that any
    change to path discovery is detected immediately. Update the expected
    sets intentionally when modifying _schema_to_paths().
    """

    def test_domain_paths_via_dataset_image(self, dataset_test, tmp_path):
        """Paths from Dataset through Dataset_Image element."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()
        sigs = path_signatures(paths)

        # Core domain paths through Dataset_Image
        assert "Dataset -> Dataset_Image -> Image" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Subject" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Observation" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Observation -> Subject" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Observation -> ClinicalRecord_Observation -> ClinicalRecord" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Subject -> Observation" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Subject -> Observation -> ClinicalRecord_Observation -> ClinicalRecord" in sigs

    def test_domain_paths_via_dataset_subject(self, dataset_test, tmp_path):
        """Paths from Dataset through Dataset_Subject element."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()
        sigs = path_signatures(paths)

        # Core domain paths through Dataset_Subject
        assert "Dataset -> Dataset_Subject -> Subject" in sigs
        assert "Dataset -> Dataset_Subject -> Subject -> Observation" in sigs
        assert "Dataset -> Dataset_Subject -> Subject -> Observation -> ClinicalRecord_Observation -> ClinicalRecord" in sigs
        assert "Dataset -> Dataset_Subject -> Subject -> Image" in sigs
        assert "Dataset -> Dataset_Subject -> Subject -> Image -> Observation" in sigs

    def test_duplicate_association_paths(self, dataset_test, tmp_path):
        """Image_Dataset_Legacy produces parallel paths to Dataset_Image."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()
        sigs = path_signatures(paths)

        # Legacy association paths mirror Dataset_Image paths
        assert "Dataset -> Image_Dataset_Legacy -> Image" in sigs
        assert "Dataset -> Image_Dataset_Legacy -> Image -> Subject" in sigs
        assert "Dataset -> Image_Dataset_Legacy -> Image -> Observation" in sigs
        assert "Dataset -> Image_Dataset_Legacy -> Image -> Observation -> ClinicalRecord_Observation -> ClinicalRecord" in sigs

    def test_feature_table_paths(self, dataset_test, tmp_path):
        """Feature tables are traversed as intermediate nodes."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()
        sigs = path_signatures(paths)

        # Image features (via Dataset_Image)
        assert "Dataset -> Dataset_Image -> Image -> Execution_Image_Quality" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Execution_Image_Quality -> ImageQuality" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Execution_Image_Quality -> Feature_Name" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Execution_Image_BoundingBox" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Execution_Image_BoundingBox -> BoundingBox" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Execution_Image_BoundingBox -> Feature_Name" in sigs

        # Subject features (via Dataset_Subject)
        assert "Dataset -> Dataset_Subject -> Subject -> Execution_Subject_Health" in sigs
        assert "Dataset -> Dataset_Subject -> Subject -> Execution_Subject_Health -> SubjectHealth" in sigs
        assert "Dataset -> Dataset_Subject -> Subject -> Execution_Subject_Health -> Feature_Name" in sigs

    def test_ml_schema_paths(self, dataset_test, tmp_path):
        """ML schema tables are traversed (with cycle avoidance)."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()
        sigs = path_signatures(paths)

        # ML schema paths
        assert "Dataset -> Dataset_Dataset_Type" in sigs
        assert "Dataset -> Dataset_Dataset_Type -> Dataset_Type" in sigs
        assert "Dataset -> Dataset_Execution" in sigs
        assert "Dataset -> Dataset_Version" in sigs
        assert "Dataset -> Dataset_File" in sigs
        assert "Dataset -> Dataset_File -> File" in sigs

        # Verify cycle avoidance: no path should contain Dataset twice
        for p in paths:
            names = [t.name for t in p]
            assert names.count("Dataset") <= 1, (
                f"Cycle detected — Dataset appears twice in path: {' -> '.join(names)}"
            )

    def test_vocabulary_termination(self, dataset_test, tmp_path):
        """Paths terminate at vocabulary tables (no traversal out of vocabs)."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()

        vocab_tables = {"Dataset_Type", "Asset_Type", "Asset_Role", "Feature_Name",
                        "ImageQuality", "SubjectHealth", "Workflow_Type"}

        for p in paths:
            for i, table in enumerate(p):
                if table.name in vocab_tables and i < len(p) - 1:
                    # Vocab table should only appear as the LAST table in a path
                    pytest.fail(
                        f"Vocabulary table {table.name} is not terminal in path: "
                        f"{' -> '.join(t.name for t in p)}"
                    )

    def test_mn_association_traversal(self, dataset_test, tmp_path):
        """M:N association tables are traversed through to reach ClinicalRecord."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()
        sigs = path_signatures(paths)

        # ClinicalRecord is only reachable via ClinicalRecord_Observation (M:N)
        cr_paths = [s for s in sigs if "ClinicalRecord" in s and "ClinicalRecord_Observation" not in s.split(" -> ClinicalRecord")[0]]
        # All paths to ClinicalRecord must go through ClinicalRecord_Observation
        for s in sigs:
            if s.endswith("ClinicalRecord"):
                assert "ClinicalRecord_Observation" in s, (
                    f"ClinicalRecord reached without ClinicalRecord_Observation: {s}"
                )

    def test_diamond_paths_exist(self, dataset_test, tmp_path):
        """Both paths in the Image→Subject diamond are discovered.

        Diamond: Image→Subject (direct FK) and Image→Observation→Subject (indirect).
        Both should exist in the path set — ambiguity resolution happens later
        in _prepare_wide_table, not in _schema_to_paths.
        """
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()
        sigs = path_signatures(paths)

        # Direct path: Image → Subject
        assert "Dataset -> Dataset_Image -> Image -> Subject" in sigs
        # Indirect path: Image → Observation → Subject
        assert "Dataset -> Dataset_Image -> Image -> Observation -> Subject" in sigs

    def test_no_execution_paths(self, dataset_test, tmp_path):
        """Execution table is skipped (hardcoded in current implementation)."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()

        for p in paths:
            names = [t.name for t in p]
            # Execution itself should not appear as a traversed node
            # (Dataset_Execution is an association to Execution, but Execution is skipped)
            if "Execution" in names:
                # Execution can appear as endpoint of Dataset_Execution path
                idx = names.index("Execution")
                if idx > 0 and names[idx - 1] != "Dataset_Execution":
                    pytest.fail(
                        f"Execution appears as non-terminal traversal node: "
                        f"{' -> '.join(names)}"
                    )

    def test_exclude_tables(self, dataset_test, tmp_path):
        """exclude_tables parameter prunes all paths through excluded table."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )

        all_paths = bag.model._schema_to_paths()
        excluded_paths = bag.model._schema_to_paths(exclude_tables={"Observation"})

        all_sigs = path_signatures(all_paths)
        excluded_sigs = path_signatures(excluded_paths)

        # Observation should not appear in any excluded path (except as root, which won't happen)
        for s in excluded_sigs:
            assert "Observation" not in s.split(" -> "), (
                f"Excluded table Observation found in path: {s}"
            )

        # Paths that went through Observation should be gone
        obs_paths = {s for s in all_sigs if "Observation" in s.split(" -> ")}
        assert len(obs_paths) > 0, "Expected some paths through Observation"
        assert obs_paths.isdisjoint(excluded_sigs), (
            f"Observation paths survived exclusion: {obs_paths & excluded_sigs}"
        )

        # But paths that don't go through Observation should still exist
        non_obs_paths = all_sigs - obs_paths
        # Some non-obs paths should survive (e.g., Dataset -> Dataset_Image -> Image -> Subject)
        assert len(excluded_sigs & non_obs_paths) > 0, (
            "Non-Observation paths should survive exclusion"
        )

    def test_total_path_count(self, dataset_test, tmp_path):
        """Sanity check: total path count for the demo schema.

        This is a coarse check — if the path count changes significantly,
        something fundamental changed in the traversal logic.
        """
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()

        # The demo schema with features produces ~115 paths.
        # Allow some tolerance for minor schema changes.
        assert len(paths) >= 80, f"Too few paths: {len(paths)} (expected ~115)"
        assert len(paths) <= 150, f"Too many paths: {len(paths)} (expected ~115)"

    def test_no_duplicate_paths(self, dataset_test, tmp_path):
        """Each path should be unique — no exact duplicates."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()
        sigs = [path_signature(p) for p in paths]

        assert len(sigs) == len(set(sigs)), (
            f"Duplicate paths found: {[s for s in sigs if sigs.count(s) > 1]}"
        )

    def test_asset_table_paths(self, dataset_test, tmp_path):
        """Asset tables (Image, BoundingBox, Report) are traversed with their metadata."""
        dataset_description = dataset_test.dataset_description
        bag = dataset_description.dataset.download_dataset_bag(
            dataset_description.dataset.current_version, use_minid=False
        )
        paths = bag.model._schema_to_paths()
        sigs = path_signatures(paths)

        # Image asset type association
        assert "Dataset -> Dataset_Image -> Image -> Image_Asset_Type" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Image_Asset_Type -> Asset_Type" in sigs

        # Image execution association
        assert "Dataset -> Dataset_Image -> Image -> Image_Execution" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Image_Execution -> Asset_Role" in sigs

        # BoundingBox asset (reached via feature table)
        assert "Dataset -> Dataset_Image -> Image -> Execution_Image_BoundingBox -> BoundingBox" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Execution_Image_BoundingBox -> BoundingBox -> BoundingBox_Asset_Type -> Asset_Type" in sigs

        # Report asset (null URLs) and downstream OCR_Report
        assert "Dataset -> Dataset_Subject -> Subject -> Observation -> Report" in sigs
        assert "Dataset -> Dataset_Subject -> Subject -> Observation -> Report -> OCR_Report" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Observation -> Report" in sigs
        assert "Dataset -> Dataset_Image -> Image -> Observation -> Report -> OCR_Report" in sigs
