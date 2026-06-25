"""File management mixin for DerivaML.

This module provides the FileMixin class which handles
file operations including adding and listing files.
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
import os
from collections import defaultdict
from itertools import batched, chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable
from urllib.parse import urlsplit

datapath = importlib.import_module("deriva.core.datapath")

from deriva.core.ermrest_model import timestamptz_to_snaptime

from deriva_ml.core.definitions import RID, FileSpec, MLTable, MLVocab, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLInvalidTerm, DerivaMLTableTypeError
from deriva_ml.dataset.aux_classes import DatasetVersion

if TYPE_CHECKING:
    from deriva.core.ermrest_catalog import ResolveRidResult

    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.model.catalog import DerivaModel


__all__ = ["FileMixin"]


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
        chunk_size: int = 500,
    ) -> "Dataset":
        """Register external file *references* and link them as execution inputs.

        Inserts a ``File``-table row per file (URL + MD5 + length) — a
        *reference* to bytes the catalog does **not** host (no Hatrac upload).
        Each file is linked to the execution as an **input**
        (``File_Execution.Asset_Role="Input"``).

        Role is intrinsic, not a choice: referencing an external file means
        *naming* a file the run consumed/depended on, so it is always an
        Input. A file the run *produced* is a Hatrac-backed execution asset —
        register those via ``asset_file_path`` + ``commit_output_assets``,
        never here. (Provenance contract: asset role is derived from context;
        a ``File`` reference is an input by its nature.)

        ``files`` is consumed lazily in batches of ``chunk_size`` — the inserts
        are streamed, so a generator source (e.g.
        ``FileSpec.create_filespecs`` over a large directory tree) never has to
        be fully materialized in memory. Only the directory→RID map used to
        build the directory-structure datasets is retained across the stream,
        and it grows with the number of distinct directories, not the number
        of files.

        Args:
            files: File specifications containing MD5 checksum, length, and URL.
                May be any iterable, including a generator; it is consumed once.
            execution_rid: Execution RID to associate files with (required for provenance).
            dataset_types: One or more dataset type terms from File_Type vocabulary.
            description: Description of the files. Recorded verbatim on every
                directory dataset; the source folder each dataset represents is
                stored structurally in the ``Directory_Dataset`` table.
            chunk_size: Number of File rows inserted per batch. Larger values
                mean fewer, bigger requests; smaller values bound per-request
                size and memory. A value at least as large as the input is a
                single batch (the historical behavior).

        Returns:
            Dataset: Dataset that represents the newly added files.

        Raises:
            DerivaMLException: If file_types are invalid or execution_rid is not an execution record.

        Examples:
            Add files via an execution:
                >>> with ml.create_execution(config) as exe:  # doctest: +SKIP
                ...     files = [FileSpec(url="path/to/file.txt", md5="abc123", length=1000)]
                ...     dataset = exe.add_files(files, dataset_types="text")
        """
        # Import here to avoid circular imports
        from deriva_ml.dataset.dataset import Dataset

        if self.resolve_rid(execution_rid).table.name != "Execution":
            raise DerivaMLTableTypeError("Execution", execution_rid)

        # Normalize dataset_types. Two types are force-included on every dataset
        # this routine creates, in addition to any caller-supplied types:
        #   - "File": the datasets hold File-asset members.
        #   - "Directory": marks the dataset as an auto-created directory-structure
        #     dataset, distinguishing these byproducts from curated datasets so a
        #     query can find (or exclude) them.
        caller_types = [dataset_types] if isinstance(dataset_types, str) else list(dataset_types or [])
        builtin_types = ["File", "Directory"]
        dataset_types = builtin_types + [t for t in caller_types if t not in builtin_types]
        for ds_type in dataset_types:
            self.lookup_term(MLVocab.dataset_type, ds_type)

        # Resolve the vocab/association lookups ONCE — they do not vary per
        # chunk. ``defined_types`` is the set of valid Asset_Type names (plus
        # synonyms); ``atable`` is the File↔Asset_Type association table name.
        defined_types = set(
            chain.from_iterable(
                [[t.name] + list(t.synonyms or []) for t in self.list_vocabulary_terms(MLVocab.asset_type)]
            )
        )
        atable = self.model.find_association(MLTable.file, MLVocab.asset_type)[0].name

        pb = self.pathBuilder()
        file_path = pb.schemas[self.ml_schema].tables["File"]
        atable_path = pb.schemas[self.ml_schema].tables[atable]
        file_execution_path = pb.schemas[self.ml_schema].File_Execution

        # Stream ``files`` in batches of ``chunk_size``. Each batch is validated,
        # inserted, tagged, and linked to the execution before the next batch is
        # pulled — so a generator source is never fully materialized. The only
        # state retained across batches is ``dir_rid_map`` (directory → File
        # RIDs), used afterward to build the directory-structure datasets; it
        # grows with the number of distinct directories, not the number of files.
        dir_rid_map = defaultdict(list)
        for batch in batched(files, chunk_size):
            # Validate this batch's file types against the defined vocabulary.
            spec_types = set(chain.from_iterable(filespec.file_types for filespec in batch))
            if spec_types - defined_types:
                raise DerivaMLInvalidTerm(MLVocab.asset_type.name, f"{spec_types - defined_types}")

            # Insert the File rows; the returned records carry the new RIDs.
            file_records = list(file_path.insert([f.model_dump(by_alias=True) for f in batch]))

            # Tag each File row with its Asset_Type terms (keyed by MD5).
            type_map = {
                filespec.md5: filespec.file_types + ([] if "File" in filespec.file_types else []) for filespec in batch
            }
            file_type_records = [
                {MLVocab.asset_type.value: file_type, "File": file_record["RID"]}
                for file_record in file_records
                for file_type in type_map[file_record["MD5"]]
            ]
            if file_type_records:
                atable_path.insert(file_type_records)

            # Link each file to the execution as an INPUT. A File-table row is a
            # reference to externally-hosted bytes — naming a file the run
            # consumed — so its role is intrinsically Input (never a parameter).
            file_execution_path.insert(
                [
                    {"File": file_record["RID"], "Execution": execution_rid, "Asset_Role": "Input"}
                    for file_record in file_records
                ]
            )

            # Group this batch's RIDs by source directory for dataset building.
            for e in file_records:
                dir_rid_map[Path(urlsplit(e["URL"]).path).parent].append(e["RID"])

        # Now create datasets that mirror the original directory structure, as a
        # single nested tree rooted at the ingest root. The tree is built from
        # real path CONTAINMENT (a directory nests into its nearest ancestor
        # directory), NOT from raw path depth — so sibling branches whose common
        # ancestor holds no files of its own still converge on one root instead
        # of being orphaned.
        #
        # ``os.path.commonpath`` gives the ingest root (the common ancestor of
        # every file-bearing directory). For a single directory it is that
        # directory itself.
        if not dir_rid_map:
            raise DerivaMLException("add_files received no files to add.")
        ingest_root = Path(os.path.commonpath([str(d) for d in dir_rid_map]))

        # The tree's nodes are every file-bearing directory PLUS every
        # intermediate ancestor up to the ingest root — the intermediates need a
        # dataset to hold their child-directory datasets even when they contain
        # no files directly (e.g. ``root/`` over ``root/a/x`` + ``root/b/y``).
        nodes: set[Path] = set()
        for directory in dir_rid_map:
            node = directory
            nodes.add(node)
            while node != ingest_root:
                node = node.parent
                nodes.add(node)

        # The ingest root keeps the bare caller description; every node dataset
        # uses the same description. The folder each node represents is recorded
        # structurally in Directory_Dataset (below), not in the prose Description.
        node_dataset: dict[Path, "Dataset"] = {
            directory: Dataset.create_dataset(
                self,  # type: ignore[arg-type]
                dataset_types=dataset_types,
                execution_rid=execution_rid,
                description=description,
            )
            for directory in nodes
        }

        # Record each directory dataset's source folder as a path relative to the
        # ingest root (the root stores "."). Structured + queryable; consumers
        # never parse the Description.
        pb.schemas[self.ml_schema].tables["Directory_Dataset"].insert(
            [
                {
                    "Dataset": ds.dataset_rid,
                    "Path": "." if directory == ingest_root else directory.relative_to(ingest_root).as_posix(),
                }
                for directory, ds in node_dataset.items()
            ]
        )

        # Wire membership: each node's dataset gets its own files plus its
        # immediate child-directory datasets (the nodes whose parent is this
        # node). Walk deepest-first so children exist before their parent adds
        # them — though, since all datasets are pre-created above, ordering only
        # affects readability here.
        for directory in sorted(nodes, key=lambda d: len(d.parts), reverse=True):
            members = list(dir_rid_map.get(directory, []))
            members += [
                child_ds.dataset_rid
                for child_dir, child_ds in node_dataset.items()
                if child_dir != directory and child_dir.parent == directory
            ]
            if members:
                node_dataset[directory].add_dataset_members(members=members, execution_rid=execution_rid)

        # The ingest root's dataset transitively contains every file.
        return node_dataset[ingest_root]

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
            [{"RID": h["RID"], "Snapshot": timestamptz_to_snaptime(h["RCT"])} for h in versions if not h["Snapshot"]]
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
                >>> files = ml.list_files()  # doctest: +SKIP
                >>> for f in files:  # doctest: +SKIP
                ...     print(f"{f['RID']}: {f['URL']}")

            Filter by file type:
                >>> image_files = ml.list_files(["image", "png"])  # doctest: +SKIP
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
