"""File management mixin for DerivaML.

This module provides the FileMixin class which handles
file operations including adding and listing files.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable
from urllib.parse import urlsplit

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
datapath = importlib.import_module("deriva.core.datapath")

from deriva_ml.core.definitions import RID, FileSpec, MLTable, MLVocab, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLInvalidTerm, DerivaMLTableTypeError
from deriva_ml.dataset.aux_classes import DatasetVersion
from deriva_ml.dataset.history import iso_to_snap

if TYPE_CHECKING:
    from deriva.core.ermrest_catalog import ResolveRidResult

    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.model.catalog import DerivaModel


class FileMixin:
    """Mixin providing file management operations.

    This mixin requires the host class to have:
        - model: DerivaModel instance
        - ml_schema: str - name of the ML schema
        - pathBuilder(): method returning catalog path builder
        - resolve_rid(): method for RID resolution (from RidResolutionMixin)
        - lookup_term(): method for vocabulary lookup (from VocabularyMixin)
        - list_vocabulary_terms(): method for listing vocab terms (from VocabularyMixin)
        - find_datasets(): method for finding datasets (from DatasetMixin)

    Methods:
        add_files: Add files to the catalog with metadata
        list_files: List files in the catalog
        _bootstrap_versions: Initialize dataset versions
        _synchronize_dataset_versions: Sync dataset versions
        _set_version_snapshot: Update version snapshots
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    ml_schema: str
    pathBuilder: Callable[[], Any]
    resolve_rid: Callable[[RID], "ResolveRidResult"]
    lookup_term: Callable[[str, str], VocabularyTerm]
    list_vocabulary_terms: Callable[[str], list[VocabularyTerm]]
    find_datasets: Callable[..., Iterable["Dataset"]]

    def add_files(
        self,
        files: Iterable[FileSpec],
        execution_rid: RID,
        dataset_types: str | list[str] | None = None,
        description: str = "",
    ) -> "Dataset":
        """Adds files to the catalog with their metadata.

        Registers files in the catalog along with their metadata (MD5, length, URL) and associates them with
        specified file types. Links files to the specified execution record for provenance tracking.

        Args:
            files: File specifications containing MD5 checksum, length, and URL.
            execution_rid: Execution RID to associate files with (required for provenance).
            dataset_types: One or more dataset type terms from File_Type vocabulary.
            description: Description of the files.

        Returns:
            Dataset: Dataset that represents the newly added files.

        Raises:
            DerivaMLException: If file_types are invalid or execution_rid is not an execution record.

        Examples:
            Add files via an execution:
                >>> with ml.create_execution(config) as exe:
                ...     files = [FileSpec(url="path/to/file.txt", md5="abc123", length=1000)]
                ...     dataset = exe.add_files(files, dataset_types="text")
        """
        # Import here to avoid circular imports
        from deriva_ml.dataset.dataset import Dataset

        if self.resolve_rid(execution_rid).table.name != "Execution":
            raise DerivaMLTableTypeError("Execution", execution_rid)

        filespec_list = list(files)

        # Get a list of all defined file types and their synonyms.
        defined_types = set(
            chain.from_iterable([[t.name] + list(t.synonyms or []) for t in self.list_vocabulary_terms(MLVocab.asset_type)])
        )

        # Get a list of all of the file types used in the filespec_list
        spec_types = set(chain.from_iterable(filespec.file_types for filespec in filespec_list))

        # Now make sure that all of the file types and dataset_types in the spec list are defined.
        if spec_types - defined_types:
            raise DerivaMLInvalidTerm(MLVocab.asset_type.name, f"{spec_types - defined_types}")

        # Normalize dataset_types, make sure File type is included.
        if isinstance(dataset_types, list):
            dataset_types = ["File"] + dataset_types if "File" not in dataset_types else dataset_types
        else:
            dataset_types = ["File", dataset_types] if dataset_types else ["File"]
        for ds_type in dataset_types:
            self.lookup_term(MLVocab.dataset_type, ds_type)

        # Add files to the file table, and collect up the resulting entries by directory name.
        pb = self.pathBuilder()
        file_records = list(
            pb.schemas[self.ml_schema].tables["File"].insert([f.model_dump(by_alias=True) for f in filespec_list])
        )

        # Get the name of the association table between file_table and file_type and add file_type records
        atable = self.model.find_association(MLTable.file, MLVocab.asset_type)[0].name
        # Need to get a link between file record and file_types.
        type_map = {
            file_spec.md5: file_spec.file_types + ([] if "File" in file_spec.file_types else [])
            for file_spec in filespec_list
        }
        file_type_records = [
            {MLVocab.asset_type.value: file_type, "File": file_record["RID"]}
            for file_record in file_records
            for file_type in type_map[file_record["MD5"]]
        ]
        pb.schemas[self.ml_schema].tables[atable].insert(file_type_records)

        # Link files to the execution for provenance tracking.
        pb.schemas[self.ml_schema].File_Execution.insert(
            [
                {"File": file_record["RID"], "Execution": execution_rid, "Asset_Role": "Output"}
                for file_record in file_records
            ]
        )

        # Now create datasets to capture the original directory structure of the files.
        dir_rid_map = defaultdict(list)
        for e in file_records:
            dir_rid_map[Path(urlsplit(e["URL"]).path).parent].append(e["RID"])

        nested_datasets = []
        path_length = 0
        dataset = None
        # Start with the longest path so we get subdirectories first.
        for p, rids in sorted(dir_rid_map.items(), key=lambda kv: len(kv[0].parts), reverse=True):
            dataset = Dataset.create_dataset(
                self,  # type: ignore[arg-type]
                dataset_types=dataset_types,
                execution_rid=execution_rid,
                description=description,
            )
            members = rids
            if len(p.parts) < path_length:
                # Going up one level in directory, so Create nested dataset
                members = [m.dataset_rid for m in nested_datasets] + rids
                nested_datasets = []
            dataset.add_dataset_members(members=members, execution_rid=execution_rid)
            nested_datasets.append(dataset)
            path_length = len(p.parts)

        return dataset

    def _bootstrap_versions(self) -> None:
        """Initialize dataset versions for datasets that don't have versions."""
        datasets = [ds.dataset_rid for ds in self.find_datasets()]
        ds_version = [
            {
                "Dataset": d,
                "Version": "0.1.0",
                "Description": "Dataset at the time of conversion to versioned datasets",
            }
            for d in datasets
        ]
        schema_path = self.pathBuilder().schemas[self.ml_schema]
        version_path = schema_path.tables["Dataset_Version"]
        dataset_path = schema_path.tables["Dataset"]
        history = list(version_path.insert(ds_version))
        dataset_versions = [{"RID": h["Dataset"], "Version": h["Version"]} for h in history]
        dataset_path.update(dataset_versions)

    def _synchronize_dataset_versions(self) -> None:
        """Synchronize dataset versions with the latest version in Dataset_Version table."""
        schema_path = self.pathBuilder().schemas[self.ml_schema]
        dataset_version_path = schema_path.tables["Dataset_Version"]
        # Get the maximum version number for each dataset.
        versions = {}
        for v in dataset_version_path.entities().fetch():
            if v["Version"] > versions.get("Dataset", DatasetVersion(0, 0, 0)):
                versions[v["Dataset"]] = v
        dataset_path = schema_path.tables["Dataset"]
        dataset_path.update([{"RID": dataset, "Version": version["RID"]} for dataset, version in versions.items()])

    def _set_version_snapshot(self) -> None:
        """Update the Snapshot column of the Dataset_Version table to the correct time."""
        dataset_version_path = self.pathBuilder().schemas[self.model.ml_schema].tables["Dataset_Version"]
        versions = dataset_version_path.entities().fetch()
        dataset_version_path.update(
            [{"RID": h["RID"], "Snapshot": iso_to_snap(h["RCT"])} for h in versions if not h["Snapshot"]]
        )

    def list_files(self, file_types: list[str] | None = None) -> list[dict[str, Any]]:
        """Lists files in the catalog with their metadata.

        Returns a list of files with their metadata including URL, MD5 hash, length, description,
        and associated file types. Files can be optionally filtered by type.

        Args:
            file_types: Filter results to only include these file types.

        Returns:
            list[dict[str, Any]]: List of file records, each containing:
                - RID: Resource identifier
                - URL: File location
                - MD5: File hash
                - Length: File size
                - Description: File description
                - File_Types: List of associated file types

        Examples:
            List all files:
                >>> files = ml.list_files()
                >>> for f in files:
                ...     print(f"{f['RID']}: {f['URL']}")

            Filter by file type:
                >>> image_files = ml.list_files(["image", "png"])
        """
        asset_type_atable, file_fk, asset_type_fk = self.model.find_association("File", "Asset_Type")
        ml_path = self.pathBuilder().schemas[self.ml_schema]
        file = ml_path.File
        asset_type = ml_path.tables[asset_type_atable.name]

        path = file.path
        path = path.link(asset_type.alias("AT"), on=file.RID == asset_type.columns[file_fk], join_type="left")
        if file_types:
            path = path.filter(asset_type.columns[asset_type_fk] == datapath.Any(*file_types))
        path = path.attributes(
            path.File.RID,
            path.File.URL,
            path.File.MD5,
            path.File.Length,
            path.File.Description,
            path.AT.columns[asset_type_fk],
        )

        file_map = {}
        for f in path.fetch():
            entry = file_map.setdefault(f["RID"], {**f, "File_Types": []})
            if ft := f.get("Asset_Type"):  # assign-and-test in one go
                entry["File_Types"].append(ft)

        # Now get rid of the File_Type key and return the result
        return [(f, f.pop("Asset_Type"))[0] for f in file_map.values()]
