"""Unit tests for ``deriva_ml.execution.asset_upload`` (audit P1 Ex-god 1st/2nd/3rd sweeps).

Pre-extraction these helpers lived on ``Execution`` as
methods. Each could only be exercised by spinning up a real
catalog + workflow + execution. Post-extraction the
helpers take ``execution`` as an explicit argument; tests
mock the small subset of fields each helper reads.

Coverage layers:

1. ``get_metadata_description`` — pure: filename → description
   lookup with hydra-rename + env-snapshot handling.
2. ``set_asset_descriptions`` — manifest snapshot + per-row
   description resolution + batched ``pathBuilder`` update.
3. ``save_runtime_environment`` — writes JSON to the
   ``asset_file_path`` staging location.
4. ``upload_hydra_config_assets`` — walks
   ``hydra-config/`` and registers each file; tolerant of
   missing dir.
5. ``clean_folder_contents`` — pure filesystem op with
   bounded retry.
6. ``update_asset_execution_table`` — per-role dispatch
   (Input vs Output), dry-run skip, Asset_Role vocab lookup.
7. ``asset_file_path`` — manifest registration, asset-type
   JSONL writeback, empty-list-types preservation, metadata
   normalization.
8. ``metrics_file`` — thin sugar over ``asset_file_path``.
9. ``commit_output_assets`` — state-machine bracketing
   for dry-run, Running auto-stop, Uploaded short-circuit,
   Stopped → Pending_Upload → Uploaded happy path,
   Pending_Upload → Failed on exception.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deriva_ml.execution.asset_upload import (
    asset_file_path,
    clean_folder_contents,
    get_metadata_description,
    metrics_file,
    save_runtime_environment,
    set_asset_descriptions,
    update_asset_execution_table,
    commit_output_assets,
    upload_hydra_config_assets,
)

# ---------------------------------------------------------------------------
# get_metadata_description — pure lookup
# ---------------------------------------------------------------------------


_METADATA = {
    "configuration.json": "DerivaML configuration",
    "uv.lock": "Dependency lockfile",
}
_ENV_DESC = "Runtime environment snapshot"


class TestGetMetadataDescription:
    def test_direct_filename_hit(self):
        assert (
            get_metadata_description(
                "uv.lock",
                metadata_descriptions=_METADATA,
                env_snapshot_description=_ENV_DESC,
            )
            == "Dependency lockfile"
        )

    def test_hydra_renamed_file(self):
        """``hydra-{timestamp}-{original}`` resolves via original-name match."""
        result = get_metadata_description(
            "hydra-20260522_120000-uv.lock",
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
        )
        assert result == "Dependency lockfile"

    def test_env_snapshot(self):
        result = get_metadata_description(
            "environment_snapshot_20260522_120000.txt",
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
        )
        assert result == _ENV_DESC

    def test_unknown_returns_none(self):
        assert (
            get_metadata_description(
                "random_file.bin",
                metadata_descriptions=_METADATA,
                env_snapshot_description=_ENV_DESC,
            )
            is None
        )

    def test_hydra_prefix_without_matching_original_returns_none(self):
        """A hydra-prefixed file whose suffix doesn't match any known
        canonical name falls through to ``None`` (not env-snapshot).
        """
        result = get_metadata_description(
            "hydra-20260522_120000-randomfile.yaml",
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
        )
        assert result is None


# ---------------------------------------------------------------------------
# set_asset_descriptions — manifest snapshot + batched update
# ---------------------------------------------------------------------------


class TestSetAssetDescriptions:
    """Pin the manifest-snapshot + two-source description resolution."""

    @staticmethod
    def _table_path(exe):
        """Drill into the pathBuilder chain mocks to get the table_path mock."""
        pb = exe._ml_object.pathBuilder.return_value
        return pb.schemas.__getitem__.return_value.tables.__getitem__.return_value

    def _build_execution_mock(self, manifest_assets: dict):
        """Build an ``Execution`` mock exposing only what the helper reads."""
        exe = MagicMock()
        manifest = MagicMock()
        manifest.assets = manifest_assets  # snapshot-once invariant
        exe._get_manifest.return_value = manifest
        return exe

    def _build_uploaded(self, table_key: str, file_name: str, asset_rid: str):
        asset = MagicMock()
        asset.file_name = file_name
        asset.asset_rid = asset_rid
        return {table_key: [asset]}

    def test_manifest_description_wins(self):
        """When the manifest carries a description, it's used (not the canonical map)."""
        manifest_entry = MagicMock()
        manifest_entry.description = "User-supplied description"
        exe = self._build_execution_mock(
            {"Image/photo.jpg": manifest_entry}
        )
        uploaded = self._build_uploaded("schema/Image", "photo.jpg", "RID-1")

        set_asset_descriptions(
            exe,
            uploaded,
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
        )

        # The pathBuilder update was called with the manifest description.
        table_path = self._table_path(exe)
        table_path.update.assert_called_once_with(
            [{"RID": "RID-1", "Description": "User-supplied description"}]
        )

    def test_metadata_fallback_for_execution_metadata(self):
        """When the manifest has no description, ``Execution_Metadata``
        files get the canonical description.
        """
        exe = self._build_execution_mock({})  # no manifest entries
        uploaded = self._build_uploaded(
            "deriva-ml/Execution_Metadata", "uv.lock", "RID-2"
        )

        set_asset_descriptions(
            exe,
            uploaded,
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
        )

        table_path = self._table_path(exe)
        table_path.update.assert_called_once_with(
            [{"RID": "RID-2", "Description": "Dependency lockfile"}]
        )

    def test_no_description_skipped(self):
        """Assets without a description (and not Execution_Metadata) get no update."""
        exe = self._build_execution_mock({})
        uploaded = self._build_uploaded("schema/Image", "photo.jpg", "RID-3")

        set_asset_descriptions(
            exe,
            uploaded,
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
        )

        # No update call.
        table_path = self._table_path(exe)
        table_path.update.assert_not_called()

    def test_no_asset_rid_skipped(self):
        """Assets without an ``asset_rid`` get no update (can't address them)."""
        manifest_entry = MagicMock()
        manifest_entry.description = "Some description"
        exe = self._build_execution_mock(
            {"Image/photo.jpg": manifest_entry}
        )
        # asset_rid is None.
        asset = MagicMock()
        asset.file_name = "photo.jpg"
        asset.asset_rid = None
        uploaded = {"schema/Image": [asset]}

        set_asset_descriptions(
            exe,
            uploaded,
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
        )

        table_path = self._table_path(exe)
        table_path.update.assert_not_called()

    def test_manifest_assets_read_once(self):
        """``manifest.assets`` is accessed exactly once per call.

        Pre-extraction the inline implementation re-read the
        manifest on every iteration; for a 10K-asset upload
        that's ~10K SQL round-trips. The single-read pattern
        was the audit's perf note.
        """
        access_count = [0]

        class _CountingManifest:
            @property
            def assets(self):
                access_count[0] += 1
                return {}

        exe = MagicMock()
        exe._get_manifest.return_value = _CountingManifest()
        uploaded = self._build_uploaded("schema/Image", "photo.jpg", "RID-1")

        set_asset_descriptions(
            exe,
            uploaded,
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
        )

        assert access_count[0] == 1, (
            f"manifest.assets must be read exactly once per call; "
            f"got {access_count[0]} reads. Audit P2: this is the "
            f"performance pin — re-reading the manifest in the inner "
            f"loop was a known O(N) regression."
        )


# ---------------------------------------------------------------------------
# save_runtime_environment — writes JSON to the asset_file_path location
# ---------------------------------------------------------------------------


class TestSaveRuntimeEnvironment:
    def test_writes_json_to_asset_file_path(self, tmp_path):
        """The helper calls ``execution.asset_file_path`` and writes
        ``get_execution_environment()`` JSON into the returned path.
        """
        snapshot_file = tmp_path / "env_snapshot.txt"
        exe = MagicMock()
        exe.asset_file_path.return_value = snapshot_file

        save_runtime_environment(
            exe,
            runtime_env_asset_type="Runtime_Env",
            env_snapshot_description=_ENV_DESC,
        )

        # asset_file_path called with the expected stamp.
        assert exe.asset_file_path.called
        kwargs = exe.asset_file_path.call_args.kwargs
        assert kwargs["asset_name"] == "Execution_Metadata"
        assert kwargs["asset_types"] == "Runtime_Env"
        assert kwargs["description"] == _ENV_DESC
        # File starts with the env_snapshot timestamp prefix.
        assert kwargs["file_name"].startswith("environment_snapshot_")

        # The file was written and contains JSON parseable as a dict.
        assert snapshot_file.exists()
        parsed = json.loads(snapshot_file.read_text())
        assert isinstance(parsed, dict)
        # ``get_execution_environment`` returns the six documented keys.
        for key in ("imports", "os", "sys", "sys_path", "site", "platform"):
            assert key in parsed


# ---------------------------------------------------------------------------
# upload_hydra_config_assets — walks hydra-config/ and registers files
# ---------------------------------------------------------------------------


class TestUploadHydraConfigAssets:
    """Pin the hydra-config walk + the "no walk on parent dir" invariant."""

    def test_no_hydra_dir_is_a_noop(self):
        """``hydra_runtime_output_dir is None`` → return without calls."""
        exe = MagicMock()
        exe._ml_object.hydra_runtime_output_dir = None

        upload_hydra_config_assets(
            exe,
            hydra_config_asset_type="Hydra_Config",
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
            execution_metadata_asset_name="Execution_Metadata",
        )
        exe.asset_file_path.assert_not_called()

    def test_missing_hydra_config_subdir_is_a_noop(self, tmp_path):
        """Hydra run dir exists but ``hydra-config/`` doesn't → noop.

        Tolerates older Hydra layouts that don't separate config from log.
        """
        hydra_dir = tmp_path / "2026-05-22_12-00-00"
        hydra_dir.mkdir()
        # NO hydra-config subdir.

        exe = MagicMock()
        exe._ml_object.hydra_runtime_output_dir = hydra_dir

        upload_hydra_config_assets(
            exe,
            hydra_config_asset_type="Hydra_Config",
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
            execution_metadata_asset_name="Execution_Metadata",
        )
        exe.asset_file_path.assert_not_called()

    def test_registers_each_config_file(self, tmp_path):
        """Each file in ``hydra-config/`` gets one ``asset_file_path`` call."""
        hydra_dir = tmp_path / "2026-05-22_12-00-00"
        config_dir = hydra_dir / "hydra-config"
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text("k: v")
        (config_dir / "hydra.yaml").write_text("k: v")
        # And a subdirectory to ensure we don't descend.
        (config_dir / "subdir").mkdir()

        exe = MagicMock()
        exe._ml_object.hydra_runtime_output_dir = hydra_dir

        upload_hydra_config_assets(
            exe,
            hydra_config_asset_type="Hydra_Config",
            metadata_descriptions=_METADATA,
            env_snapshot_description=_ENV_DESC,
            execution_metadata_asset_name="Execution_Metadata",
        )

        # Two files registered (subdir is skipped).
        assert exe.asset_file_path.call_count == 2

        # All calls use the hydra-rename pattern with the run-dir timestamp.
        for call in exe.asset_file_path.call_args_list:
            kwargs = call.kwargs
            assert kwargs["asset_types"] == "Hydra_Config"
            assert kwargs["rename_file"].startswith("hydra-2026-05-22_12-00-00-")


# ---------------------------------------------------------------------------
# clean_folder_contents — pure filesystem op with bounded retry
# ---------------------------------------------------------------------------


class TestCleanFolderContents:
    def test_removes_files_and_subdirs(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "file1.txt").write_text("a")
        (target / "file2.txt").write_text("b")
        (target / "sub").mkdir()
        (target / "sub" / "nested.txt").write_text("c")

        clean_folder_contents(target, remove_folder=True)

        # Whole folder gone.
        assert not target.exists()

    def test_remove_folder_false_keeps_dir(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "file.txt").write_text("a")

        clean_folder_contents(target, remove_folder=False)

        # Folder still exists but is empty.
        assert target.exists()
        assert list(target.iterdir()) == []

    def test_handles_nonexistent_folder(self, tmp_path):
        """Cleaning a folder that doesn't exist logs but doesn't raise."""
        bogus = tmp_path / "does_not_exist"
        # Capture log via a stub logger.
        log = MagicMock()
        clean_folder_contents(bogus, remove_folder=True, logger=log)
        # Logged a warning; didn't raise.
        log.warning.assert_called()

    def test_retries_on_oserror(self, tmp_path, monkeypatch):
        """The retry loop falls back to logging after 3 failed attempts."""
        target = tmp_path / "target"
        target.mkdir()
        (target / "stuck.txt").write_text("x")

        # Patch ``Path.unlink`` to always raise OSError.
        original_unlink = Path.unlink
        unlink_calls = [0]

        def failing_unlink(self):
            unlink_calls[0] += 1
            raise OSError("locked")

        monkeypatch.setattr(Path, "unlink", failing_unlink)

        log = MagicMock()
        # Avoid the per-attempt 1s sleep dominating the test.
        with patch("deriva_ml.execution.asset_upload.time.sleep"):
            clean_folder_contents(target, remove_folder=False, logger=log)

        # Three retry attempts on the stuck file.
        assert unlink_calls[0] == 3
        # Final warning logged for the file.
        assert log.warning.called

        # Restore for safety.
        monkeypatch.setattr(Path, "unlink", original_unlink)


# ---------------------------------------------------------------------------
# update_asset_execution_table — second sweep (audit Ex-god)
# ---------------------------------------------------------------------------


class TestUpdateAssetExecutionTable:
    """Pin the per-role dispatch for asset → execution association rows."""

    def _build_execution_mock(
        self,
        *,
        dry_run: bool = False,
        with_associations: bool = True,
    ) -> MagicMock:
        """Build an ``Execution`` mock with the minimum surface the helper reads."""
        exe = MagicMock()
        exe._dry_run = dry_run
        exe.execution_rid = "EX-1"
        exe._working_dir = "/tmp/wdir"

        # ``_model.find_association`` returns
        # ``(assoc_table_obj, asset_fk_col, exec_fk_col)``.
        assoc_table = MagicMock()
        assoc_table.schema.name = "deriva-ml"
        assoc_table.name = "Image_Execution"

        type_assoc_table = MagicMock()
        type_assoc_table.schema.name = "deriva-ml"
        type_assoc_table.name = "Image_Asset_Type"

        if with_associations:
            def fake_find_assoc(table_name, partner):
                if partner == "Execution":
                    return (assoc_table, "Image", "Execution")
                if partner == "Asset_Type":
                    return (type_assoc_table, None, None)
                raise KeyError(partner)

            exe._model.find_association.side_effect = fake_find_assoc
        return exe

    def _drill_to_table(self, exe, schema, table):
        """Drill into the pathBuilder mock chain to a specific table_path."""
        return exe._ml_object.pathBuilder.return_value.schemas.__getitem__.return_value.tables.__getitem__.return_value

    def test_dry_run_skips_all_work(self):
        """``execution._dry_run`` is True → the helper returns immediately.

        No catalog calls, no vocab lookup, no pathBuilder access.
        """
        exe = self._build_execution_mock(dry_run=True)
        asset = MagicMock()
        asset.asset_rid = "ASSET-1"

        update_asset_execution_table(
            exe,
            {"deriva-ml/Image": [asset]},
            asset_role="Input",
            asset_role_vocab_term="Asset_Role",
            input_file_tag="Input_File",
            output_file_tag="Output_File",
            asset_type_path_fn=lambda *_: Path("/never/called"),
        )

        # No vocab lookup, no pathBuilder access in dry-run.
        exe._ml_object.lookup_term.assert_not_called()
        exe._ml_object.pathBuilder.assert_not_called()

    def test_input_branch_inserts_execution_and_input_file_tag(self):
        """Input role → ``{Asset}_Execution`` insert + ``Input_File`` tag insert."""
        exe = self._build_execution_mock()
        asset = MagicMock()
        asset.asset_rid = "ASSET-1"

        update_asset_execution_table(
            exe,
            {"deriva-ml/Image": [asset]},
            asset_role="Input",
            asset_role_vocab_term="Asset_Role",
            input_file_tag="Input_File",
            output_file_tag="Output_File",
            asset_type_path_fn=lambda *_: Path("/never/called"),
        )

        # Asset_Role vocab lookup happened.
        exe._ml_object.lookup_term.assert_called_once_with("Asset_Role", "Input")

        # Two inserts happened: one for Image_Execution, one for Image_Asset_Type.
        # We can verify the calls via the pathBuilder mock chain.
        pb = exe._ml_object.pathBuilder.return_value
        # pb.schemas["deriva-ml"].tables[...].insert was called twice.
        table_path = pb.schemas.__getitem__.return_value.tables.__getitem__.return_value
        assert table_path.insert.call_count == 2

        # First call is the {Asset}_Execution insert; second is the
        # {Asset}_Asset_Type insert with Input_File tag.
        type_insert_call = table_path.insert.call_args_list[1]
        rows = type_insert_call.args[0]
        assert all(row["Asset_Type"] == "Input_File" for row in rows)
        # on_conflict_skip=True is preserved.
        assert type_insert_call.kwargs == {"on_conflict_skip": True}

    def test_output_branch_reads_type_map_and_adds_output_file_tag(self, tmp_path):
        """Output role → reads per-file type map, auto-adds ``Output_File`` tag."""
        # Stage a JSONL file with the per-file type map.
        type_file = tmp_path / "asset_type.jsonl"
        type_file.write_text(json.dumps({"photo.jpg": ["Model_File"]}) + "\n")

        exe = self._build_execution_mock()
        asset = MagicMock()
        asset.asset_rid = "ASSET-1"
        asset.file_name = "photo.jpg"

        def fake_asset_type_path_fn(working_dir, exec_rid, table):
            return type_file

        update_asset_execution_table(
            exe,
            {"deriva-ml/Image": [asset]},
            asset_role="Output",
            asset_role_vocab_term="Asset_Role",
            input_file_tag="Input_File",
            output_file_tag="Output_File",
            asset_type_path_fn=fake_asset_type_path_fn,
        )

        # Asset_Role vocab lookup happened.
        exe._ml_object.lookup_term.assert_called_once_with("Asset_Role", "Output")

        # The asset's ``asset_types`` was mutated to include Output_File.
        assert "Output_File" in asset.asset_types
        assert "Model_File" in asset.asset_types

        # The type insert was called with rows for BOTH tags.
        pb = exe._ml_object.pathBuilder.return_value
        table_path = pb.schemas.__getitem__.return_value.tables.__getitem__.return_value
        type_insert_call = table_path.insert.call_args_list[1]
        rows = type_insert_call.args[0]
        tag_values = {row["Asset_Type"] for row in rows}
        assert tag_values == {"Model_File", "Output_File"}

    def test_output_branch_does_not_duplicate_existing_output_file_tag(self, tmp_path):
        """When the type map already has ``Output_File``, it's not added twice."""
        type_file = tmp_path / "asset_type.jsonl"
        type_file.write_text(
            json.dumps({"photo.jpg": ["Model_File", "Output_File"]}) + "\n"
        )

        exe = self._build_execution_mock()
        asset = MagicMock()
        asset.asset_rid = "ASSET-1"
        asset.file_name = "photo.jpg"

        update_asset_execution_table(
            exe,
            {"deriva-ml/Image": [asset]},
            asset_role="Output",
            asset_role_vocab_term="Asset_Role",
            input_file_tag="Input_File",
            output_file_tag="Output_File",
            asset_type_path_fn=lambda *_: type_file,
        )

        # ``asset_types`` should still have exactly the two tags.
        assert sorted(asset.asset_types) == ["Model_File", "Output_File"]

    def test_uses_passed_in_vocab_term(self):
        """The ``asset_role_vocab_term`` arg is threaded through to ``lookup_term``.

        Pins the audit-friendly decoupling: the helper doesn't
        import ``MLVocab``, so a future refactor of the enum
        doesn't ripple through this module.
        """
        exe = self._build_execution_mock()
        asset = MagicMock()
        asset.asset_rid = "ASSET-1"

        update_asset_execution_table(
            exe,
            {"deriva-ml/Image": [asset]},
            asset_role="Input",
            asset_role_vocab_term="My_Custom_Vocab",
            input_file_tag="Input_File",
            output_file_tag="Output_File",
            asset_type_path_fn=lambda *_: Path("/never/called"),
        )

        exe._ml_object.lookup_term.assert_called_once_with("My_Custom_Vocab", "Input")


# ---------------------------------------------------------------------------
# asset_file_path — third sweep
# ---------------------------------------------------------------------------


class TestAssetFilePath:
    """Pin the asset-staging contract for the third sweep extraction."""

    def _build_execution(self, tmp_path: Path):
        """Build an ``Execution`` mock with the minimum surface the helper reads."""
        exe = MagicMock()
        exe._working_dir = str(tmp_path)
        exe.execution_rid = "EX-1"

        # Model: ``Image`` is the only asset table.
        def is_asset(name):
            return name in ("Image", "Execution_Metadata")

        def name_to_table(name):
            t = MagicMock()
            t.name = name
            t.schema.name = "deriva-ml"
            return t

        exe._model.is_asset.side_effect = is_asset
        exe._model.name_to_table.side_effect = name_to_table

        # Manifest mock: ``add_asset`` records the call.
        manifest = MagicMock()
        exe._get_manifest.return_value = manifest

        return exe, manifest

    def _flat_dir_fn(self, tmp_path: Path):
        """Build a ``flat_asset_dir`` stand-in that mkdirs and returns a path."""
        def _fn(working_dir, exec_rid, asset_name):
            p = Path(working_dir) / "flat" / exec_rid / asset_name
            p.mkdir(parents=True, exist_ok=True)
            return p

        return _fn

    def _type_path_fn(self, tmp_path: Path):
        """Build an ``asset_type_path`` stand-in that returns a writable JSONL path."""
        def _fn(working_dir, exec_rid, asset_table):
            p = Path(working_dir) / "type-jsonl" / exec_rid
            p.mkdir(parents=True, exist_ok=True)
            return p / f"{asset_table.name}.jsonl"

        return _fn

    def test_rejects_non_asset_table(self, tmp_path):
        """Non-asset table → ``DerivaMLException``."""
        from deriva_ml.core.exceptions import DerivaMLException

        exe, _ = self._build_execution(tmp_path)
        exe._model.is_asset.side_effect = lambda name: False

        with pytest.raises(DerivaMLException, match="not an asset"):
            asset_file_path(
                exe,
                "Subject",
                "subj.csv",
                asset_type_vocab_term="Asset_Type",
                flat_asset_dir_fn=self._flat_dir_fn(tmp_path),
                asset_type_path_fn=self._type_path_fn(tmp_path),
            )

    def test_defaults_asset_types_to_asset_name(self, tmp_path):
        """``asset_types=None`` defaults to ``[asset_name]`` (with vocab lookup)."""
        exe, manifest = self._build_execution(tmp_path)

        asset_file_path(
            exe,
            "Image",
            "photo.jpg",
            asset_type_vocab_term="Asset_Type",
            flat_asset_dir_fn=self._flat_dir_fn(tmp_path),
            asset_type_path_fn=self._type_path_fn(tmp_path),
        )

        # Vocab lookup happened with the default.
        exe._ml_object.lookup_term.assert_called_once_with("Asset_Type", "Image")

    def test_empty_list_asset_types_preserved(self, tmp_path):
        """Explicit ``asset_types=[]`` is honored — no fallback to asset_name.

        Pre-fix the inline ``or`` chain collapsed an empty list
        to ``asset_name`` and then ``lookup_term`` would fail on
        the table name (audit P1 ``asset_file_path`` falsy bug).
        Pin the ``is None`` semantic against future regressions.
        """
        exe, _ = self._build_execution(tmp_path)

        asset_file_path(
            exe,
            "Image",
            "photo.jpg",
            asset_types=[],
            asset_type_vocab_term="Asset_Type",
            flat_asset_dir_fn=self._flat_dir_fn(tmp_path),
            asset_type_path_fn=self._type_path_fn(tmp_path),
        )

        # No vocab lookup — empty list is honored.
        exe._ml_object.lookup_term.assert_not_called()

    def test_legacy_asset_type_kwarg_fallback(self, tmp_path):
        """``legacy_kwargs={"Asset_Type": "X"}`` is used when ``asset_types`` is None."""
        exe, _ = self._build_execution(tmp_path)

        asset_file_path(
            exe,
            "Image",
            "photo.jpg",
            asset_type_vocab_term="Asset_Type",
            flat_asset_dir_fn=self._flat_dir_fn(tmp_path),
            asset_type_path_fn=self._type_path_fn(tmp_path),
            legacy_kwargs={"Asset_Type": "Training_Data"},
        )

        exe._ml_object.lookup_term.assert_called_once_with("Asset_Type", "Training_Data")

    def test_registers_in_manifest_with_correct_key(self, tmp_path):
        """Manifest is keyed as ``{asset_name}/{target_name}``."""
        exe, manifest = self._build_execution(tmp_path)

        asset_file_path(
            exe,
            "Image",
            "photo.jpg",
            asset_type_vocab_term="Asset_Type",
            flat_asset_dir_fn=self._flat_dir_fn(tmp_path),
            asset_type_path_fn=self._type_path_fn(tmp_path),
        )

        assert manifest.add_asset.call_count == 1
        manifest_key = manifest.add_asset.call_args.args[0]
        assert manifest_key == "Image/photo.jpg"

    def test_writes_asset_type_jsonl(self, tmp_path):
        """The asset-type JSONL file gets a line per call (legacy upload contract)."""
        exe, _ = self._build_execution(tmp_path)

        asset_file_path(
            exe,
            "Image",
            "photo.jpg",
            asset_types=["Training_Data"],
            asset_type_vocab_term="Asset_Type",
            flat_asset_dir_fn=self._flat_dir_fn(tmp_path),
            asset_type_path_fn=self._type_path_fn(tmp_path),
        )

        jsonl = tmp_path / "type-jsonl" / "EX-1" / "Image.jsonl"
        assert jsonl.exists()
        line = jsonl.read_text().strip()
        # The JSONL maps target-filename to asset_types list.
        parsed = json.loads(line)
        assert parsed == {"photo.jpg": ["Training_Data"]}

    def test_metadata_dict_passed_through(self, tmp_path):
        """``metadata`` dict is merged with ``legacy_kwargs`` and stored."""
        exe, manifest = self._build_execution(tmp_path)

        asset_file_path(
            exe,
            "Image",
            "photo.jpg",
            asset_type_vocab_term="Asset_Type",
            flat_asset_dir_fn=self._flat_dir_fn(tmp_path),
            asset_type_path_fn=self._type_path_fn(tmp_path),
            metadata={"Subject": "2-DEF"},
            legacy_kwargs={"Acquisition_Date": "2026-01-15"},
        )

        # The AssetEntry passed to manifest.add_asset has both keys.
        entry = manifest.add_asset.call_args.args[1]
        assert entry.metadata == {
            "Subject": "2-DEF",
            "Acquisition_Date": "2026-01-15",
        }

    def test_asset_record_model_dump_normalized(self, tmp_path):
        """A Pydantic AssetRecord's ``model_dump()`` is used, ``None`` values stripped."""
        exe, manifest = self._build_execution(tmp_path)

        record = MagicMock()
        record.model_dump.return_value = {
            "Subject": "2-DEF",
            "Acquisition_Date": None,  # stripped
        }

        asset_file_path(
            exe,
            "Image",
            "photo.jpg",
            asset_type_vocab_term="Asset_Type",
            flat_asset_dir_fn=self._flat_dir_fn(tmp_path),
            asset_type_path_fn=self._type_path_fn(tmp_path),
            metadata=record,
        )

        entry = manifest.add_asset.call_args.args[1]
        assert entry.metadata == {"Subject": "2-DEF"}


# ---------------------------------------------------------------------------
# metrics_file — thin sugar over asset_file_path
# ---------------------------------------------------------------------------


class TestMetricsFile:
    def test_delegates_with_correct_asset_type(self):
        """``metrics_file(filename)`` calls
        ``execution.asset_file_path(execution_metadata, filename,
        asset_types=metrics_file_tag)``."""
        exe = MagicMock()
        exe.asset_file_path.return_value = "<AssetFilePath>"

        result = metrics_file(
            exe,
            "metrics.jsonl",
            execution_metadata_asset_name="Execution_Metadata",
            metrics_file_asset_type="Metrics_File",
        )

        assert result == "<AssetFilePath>"
        exe.asset_file_path.assert_called_once_with(
            "Execution_Metadata",
            "metrics.jsonl",
            asset_types="Metrics_File",
        )


# ---------------------------------------------------------------------------
# commit_output_assets — state-machine bracketing
# ---------------------------------------------------------------------------


class TestUploadExecutionOutputs:
    """Pin the state-machine transitions around the upload work."""

    def _build_execution(
        self,
        *,
        status,
        dry_run: bool = False,
        pending: dict | None = None,
    ):
        """Build an ``Execution`` mock for the upload orchestrator."""
        exe = MagicMock()
        exe._dry_run = dry_run
        exe.execution_rid = "EX-1"
        exe.status = status

        manifest = MagicMock()
        manifest.pending_assets.return_value = pending if pending is not None else {"K": "v"}
        exe._get_manifest.return_value = manifest

        exe._bag_commit_upload.return_value = {"schema/Image": []}
        return exe

    def _statuses(self):
        """Use sentinel enum-like objects so ``is`` comparisons work."""
        class _StatusSentinel:
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return f"<Status.{self.name}>"

        return {
            "Running": _StatusSentinel("Running"),
            "Stopped": _StatusSentinel("Stopped"),
            "Pending_Upload": _StatusSentinel("Pending_Upload"),
            "Uploaded": _StatusSentinel("Uploaded"),
            "Failed": _StatusSentinel("Failed"),
        }

    def test_dry_run_returns_empty(self):
        """Dry-run executions short-circuit to ``{}``."""
        s = self._statuses()
        exe = self._build_execution(status=s["Stopped"], dry_run=True)

        result = commit_output_assets(
            exe,
            pending_upload_status=s["Pending_Upload"],
            uploaded_status=s["Uploaded"],
            failed_status=s["Failed"],
            running_status=s["Running"],
            stopped_status=s["Stopped"],
            format_duration_fn=lambda a, b: "0s",
        )
        assert result == {}
        # No state transitions, no bag-commit calls.
        exe._bag_commit_upload.assert_not_called()
        exe.update_status.assert_not_called()

    def test_uploaded_short_circuit_when_no_pending(self):
        """``status=Uploaded`` + no pending → return ``uploaded_assets``, no transition."""
        s = self._statuses()
        exe = self._build_execution(status=s["Uploaded"], pending={})
        exe.uploaded_assets = {"schema/Image": []}

        result = commit_output_assets(
            exe,
            pending_upload_status=s["Pending_Upload"],
            uploaded_status=s["Uploaded"],
            failed_status=s["Failed"],
            running_status=s["Running"],
            stopped_status=s["Stopped"],
            format_duration_fn=lambda a, b: "0s",
        )
        assert result == {"schema/Image": []}
        exe._bag_commit_upload.assert_not_called()
        exe.update_status.assert_not_called()

    def test_running_auto_stops(self):
        """``status=Running`` → auto-call ``execution_stop()`` before upload."""
        s = self._statuses()
        # After execution_stop, status transitions to Stopped (and then
        # Pending_Upload, then Uploaded). We simulate by toggling the
        # status attribute via a side_effect on execution_stop.
        exe = self._build_execution(status=s["Running"])

        def _stop_fn():
            exe.status = s["Stopped"]

        exe.execution_stop.side_effect = _stop_fn

        # After update_status(Pending_Upload), status becomes Pending_Upload.
        # After the successful upload, the helper transitions to Uploaded.
        update_calls = []

        def _update_fn(target, **kwargs):
            update_calls.append(target)
            exe.status = target

        exe.update_status.side_effect = _update_fn

        commit_output_assets(
            exe,
            pending_upload_status=s["Pending_Upload"],
            uploaded_status=s["Uploaded"],
            failed_status=s["Failed"],
            running_status=s["Running"],
            stopped_status=s["Stopped"],
            format_duration_fn=lambda a, b: "0s",
        )

        # execution_stop was called once.
        assert exe.execution_stop.call_count == 1
        # Status transitions: Pending_Upload, then Uploaded.
        assert update_calls == [s["Pending_Upload"], s["Uploaded"]]

    def test_failure_transitions_to_failed(self):
        """``_bag_commit_upload`` raises → ``Pending_Upload → Failed``."""
        s = self._statuses()
        exe = self._build_execution(status=s["Stopped"])

        exe._bag_commit_upload.side_effect = RuntimeError("bag-load failed")

        update_calls = []

        def _update_fn(target, **kwargs):
            update_calls.append((target, kwargs))
            exe.status = target

        exe.update_status.side_effect = _update_fn

        with pytest.raises(RuntimeError, match="bag-load failed"):
            commit_output_assets(
                exe,
                pending_upload_status=s["Pending_Upload"],
                uploaded_status=s["Uploaded"],
                failed_status=s["Failed"],
                running_status=s["Running"],
                stopped_status=s["Stopped"],
                format_duration_fn=lambda a, b: "0s",
            )

        # Status transitions: Pending_Upload (pre-upload bracket),
        # then Failed (catch handler).
        assert [c[0] for c in update_calls] == [s["Pending_Upload"], s["Failed"]]
        # Failed transition carried an error= kwarg.
        assert "error" in update_calls[1][1]

    def test_happy_path_clean_folder_runs(self):
        """Successful upload with ``clean_folder=True`` cleans the execution root."""
        s = self._statuses()
        exe = self._build_execution(status=s["Stopped"])

        def _update_fn(target, **kwargs):
            exe.status = target

        exe.update_status.side_effect = _update_fn

        commit_output_assets(
            exe,
            clean_folder=True,
            pending_upload_status=s["Pending_Upload"],
            uploaded_status=s["Uploaded"],
            failed_status=s["Failed"],
            running_status=s["Running"],
            stopped_status=s["Stopped"],
            format_duration_fn=lambda a, b: "0s",
        )

        exe._clean_folder_contents.assert_called_once_with(exe._execution_root)
