"""Bag download orchestration for the :class:`Dataset` public API.

Extracted from :mod:`deriva_ml.dataset.dataset` in Phase 3 (audit
§3.A). The cluster handles the **three-tier caching strategy** that
:meth:`Dataset.download_dataset_bag` drives end-to-end:

1. **Tier 1 — Local deterministic cache** (no network beyond a single
   schema-spec query). Cache key is
   ``{spec_hash[:16]}_{snapshot}`` — both the FK traversal plan AND
   the data snapshot must match for a hit, so schema changes
   invalidate the cache even on the same snapshot. The cached bag
   lives at ``{cache_dir}/bags/{checksum}/Dataset_{rid}/`` and is
   tracked in :class:`deriva.bag.cache_index.BagCacheIndex`.

2. **Tier 2 — MINID / S3** (when ``use_minid=True``). Checks the
   Dataset_Version row for a recorded MINID; downloads the bag from
   S3 if the stored spec hash still matches the current spec.
   Regenerates the MINID otherwise.

3. **Tier 3 — Client-side bag generation** (when ``use_minid=False``).
   Builds the bag locally by running ERMrest queries through
   :class:`~deriva_ml.dataset.bag_builder.DatasetBagBuilder` against
   the version's snapshot catalog. Lands the result in the local
   cache so Tier 1 picks it up next time.

Why a separate module? The cluster is ~550 LoC of orchestration with
exactly **one external entry point** (:meth:`Dataset.download_dataset_bag`)
and **one external dependency**
(:class:`~deriva_ml.dataset.bag_builder.DatasetBagBuilder`). Keeping
it in ``dataset.py`` mixed orchestration with the Dataset class's
core CRUD/versioning surface. Splitting it makes :class:`Dataset`
read like a domain model and lets the bag-pipeline reader see the
download flow without scrolling past unrelated methods.

The public entry point :meth:`Dataset.download_dataset_bag` stays on
:class:`Dataset` (where users expect it); these helpers just back
its implementation.

Functions are stateless and take a :class:`Dataset` instance as
their first argument. Each function corresponds 1:1 to a former
``Dataset._<method>`` private:

* :func:`get_dataset_minid` (was ``Dataset._get_dataset_minid``)
* :func:`create_dataset_minid` (was ``Dataset._create_dataset_minid``)
* :func:`download_dataset_minid` (was ``Dataset._download_dataset_minid``)
* :func:`fetch_minid_metadata` (was ``Dataset._fetch_minid_metadata``)
* :func:`materialize_dataset_bag` (was ``Dataset._materialize_dataset_bag``)
"""

from __future__ import annotations

import hashlib
import json
import shutil
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import deriva.core.utils.hash_utils as hash_utils
import requests
from bdbag import bdbag_api as bdb
from bdbag.fetch.fetcher import fetch_single_file
from deriva.core.utils.core_utils import format_exception
from deriva.transfer.download import (
    DerivaDownloadAuthenticationError,
    DerivaDownloadAuthorizationError,
    DerivaDownloadConfigurationError,
    DerivaDownloadError,
    DerivaDownloadTimeoutError,
)
from deriva.transfer.download.deriva_export import DerivaExport

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.logging_config import get_logger
from deriva_ml.dataset.aux_classes import DatasetMinid, DatasetVersion
from deriva_ml.dataset.bag_builder import DatasetBagBuilder

_module_logger = get_logger(__name__)

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset import Dataset


def _hash_spec(spec: Any) -> str:
    """SHA-256 hex digest of a download spec, keyed by sorted-JSON form.

    Used to key the bag cache and to compare a freshly computed spec
    against the ``Minid_Spec_Hash`` recorded on a historical version.
    ``sort_keys=True`` makes the hash order-insensitive across dict
    renderings so semantically-equal specs always hash identically.

    Args:
        spec: The download spec (any JSON-serializable Python object,
            typically the dict returned by
            ``DatasetBagBuilder.generate_dataset_download_spec``).

    Returns:
        Lowercase hex-digest string (64 chars).
    """
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def fetch_minid_metadata(dataset: "Dataset", version: DatasetVersion, url: str) -> DatasetMinid:
    """Fetch MINID metadata from the MINID service.

    Args:
        dataset: The dataset the MINID belongs to (used for logging context
            only; the actual fetch is a single HTTP GET).
        version: The dataset version associated with this MINID.
        url: The MINID landing page URL.

    Returns:
        DatasetMinid: Parsed metadata including bag URL, checksum, and identifiers.

    Raises:
        requests.HTTPError: If the MINID service request fails.
    """
    del dataset  # Currently unused; reserved for future logging / auth context.
    r = requests.get(url, headers={"accept": "application/json"})
    r.raise_for_status()
    return DatasetMinid(dataset_version=version, **r.json())


def download_dataset_minid(dataset: "Dataset", minid: DatasetMinid, use_minid: bool) -> Path:
    """Download and extract a dataset bag archive into the local cache.

    Handles three source types based on how the bag was obtained:

    1. **Local cache hit** (``minid.checksum`` set by Tier 1 in
       :func:`get_dataset_minid`):
       The index already records this checksum and the bag directory
       exists at ``{cache_root}/bags/{checksum}/Dataset_{rid}`` → return
       immediately.

    2. **S3 download** (``use_minid=True``):
       Download the bag archive from S3 via ``minid.bag_url``.

    3. **Client-side bag** (``use_minid=False``):
       The bag was already generated locally by
       :func:`create_dataset_minid` driving
       :meth:`DatasetBagBuilder.build_bag` and is referenced via a
       ``file://`` URI.

    After obtaining the archive, this function:

    - Extracts it under a staging directory (atomic — prevents corrupt caches).
    - Validates the BDBag structure.
    - Moves the staging directory to its final cache location under
      ``{cache_root}/bags/{checksum}/Dataset_{rid}/``.
    - Records the bag in :class:`~deriva.bag.cache_index.BagCacheIndex`
      so Tier-1 lookups can find it on the next call.
    - Cleans up temporary files (including any ``client_export``
      intermediate produced by the non-MINID path).

    Cache layout (post-Phase-2 cutover):
        ``{cache_root}/bags/{checksum}/Dataset_{rid}/``

        ``checksum`` is the deterministic ``{spec_hash[:16]}_{snapshot}``
        string for non-MINID downloads (set by Tier 1/3) or the SHA-256
        of the S3 archive (set by Tier 2).

    Args:
        dataset: The dataset being downloaded (for catalog access and
            logging).
        minid: DatasetMinid with bag URL and cache key (in checksum field).
        use_minid: If True, source is S3. If False, source is local file://.

    Returns:
        Path to the extracted bag directory:
        ``{cache_root}/bags/{checksum}/Dataset_{rid}``.
    """
    from deriva.bag.cache_index import BagCacheIndex

    index = BagCacheIndex(dataset._ml_instance.cache_dir)

    # Tier-1 hit: the index lookup in get_dataset_minid set minid.checksum;
    # the bag may already exist on disk from a prior download.
    bag_root = index.bag_dir_for(minid.checksum)
    bag_dir = bag_root / f"Dataset_{minid.dataset_rid}"
    if bag_dir.exists():
        dataset._logger.info(f"Using cached bag for {minid.dataset_rid} Version:{minid.dataset_version}")
        return bag_dir

    # ----- Download the archive -------------------------------------------
    with TemporaryDirectory() as tmp_dir:
        if use_minid:
            # Tier 2: Download bag archive from S3
            bag_path = Path(tmp_dir) / Path(urlparse(minid.bag_url).path).name
            archive_path = fetch_single_file(minid.bag_url, output_path=bag_path)
        elif minid.bag_url.startswith("file://"):
            # Tier 3: Client-side bag — already on local filesystem
            archive_path = urlparse(minid.bag_url).path
        else:
            # Legacy: Download from catalog export endpoint
            exporter = DerivaExport(host=dataset._ml_instance.catalog.deriva_server.server, output_dir=tmp_dir)
            archive_path = exporter.retrieve_file(minid.bag_url)

        # For non-MINID downloads without a pre-computed cache key (legacy
        # code path), fall back to SHA-256 of the archive as cache key.
        if not use_minid and not minid.checksum:
            hashes = hash_utils.compute_file_hashes(archive_path, hashes=["md5", "sha256"])
            checksum = hashes["sha256"][0]
            bag_root = index.bag_dir_for(checksum)
            bag_dir = bag_root / f"Dataset_{minid.dataset_rid}"
            if bag_dir.exists():
                dataset._logger.info(f"Using cached bag for {minid.dataset_rid} Version:{minid.dataset_version}")
                return bag_dir
            # Rebind minid.checksum so the index record at the end of this
            # function uses the SHA-256 cache key. ``DatasetMinid`` is
            # immutable; rebuild it with the derived checksum.
            minid = minid.model_copy(update={"checksums": [{"function": "sha256", "value": checksum}]})

        # ----- Extract to staging directory (atomic cache population) ------
        # Write to a temporary staging directory first. Only rename to the
        # final cache location after successful extraction and validation.
        # This prevents partial/corrupt cache entries if the process crashes.
        staging_dir = bag_root.parent / f"{bag_root.name}_staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            extracted_bag_path = bdb.extract_bag(archive_path, staging_dir.as_posix())
            bdb.validate_bag_structure(extracted_bag_path)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    # Atomic move: staging → final cache location. The bag directory's
    # parent (``cache_root/bags/{checksum}``) is created by the rename.
    bag_root.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.rename(bag_root)

    # Record the bag in the index so the next Tier-1 lookup finds it.
    try:
        index.record(
            checksum=minid.checksum,
            anchors=[("Dataset", minid.dataset_rid)],
            anchor_summary={"version": str(minid.dataset_version)},
        )
    finally:
        index.dispose()

    # Clean up the client_export temp directory for local file:// bags.
    # After extraction to cache, the original archive is no longer needed.
    if not use_minid and minid.bag_url.startswith("file://"):
        export_dir = Path(archive_path).parent
        if "client_export" in export_dir.parts:
            shutil.rmtree(export_dir, ignore_errors=True)

    return bag_dir


def create_dataset_minid(
    dataset: "Dataset",
    version: DatasetVersion,
    use_minid: bool = True,
    exclude_tables: set[str] | None = None,
    spec: dict | None = None,
    spec_hash: str | None = None,
    timeout: tuple[int, int] | None = None,
) -> str:
    """Create a new MINID (Minimal Viable Identifier) for the dataset.

    This function generates a BDBag export of the dataset and optionally
    registers it with a MINID service for persistent identification.
    The bag is uploaded to S3 storage when using MINIDs.

    Args:
        dataset: The dataset to mint a MINID for.
        version: The dataset version to create a MINID for.
        use_minid: If True, register with MINID service and upload to S3.
            If False, just generate the bag and return a local URL.
        exclude_tables: Optional set of table names to exclude from FK traversal.
        spec: Optional pre-computed download spec dict. If None, the spec is
            generated from the snapshot catalog.
        spec_hash: Optional pre-computed SHA-256 hash of the spec. If None and
            spec is provided, it is computed from the spec.
        timeout: Optional (connect_timeout, read_timeout) in seconds for network
            requests. Defaults to (10, 610).

    Returns:
        URL to the MINID landing page (if ``use_minid=True``) or
        the direct bag download URL (``file://`` URI).
    """
    with TemporaryDirectory() as tmp_dir:
        # Generate spec if not supplied (allows callers to reuse a spec they already computed).
        if spec is None:
            version_snapshot_catalog = dataset._version_snapshot_catalog(version)
            downloader = DatasetBagBuilder(
                ml_instance=version_snapshot_catalog,
                s3_bucket=dataset._ml_instance.s3_bucket,
                use_minid=use_minid,
                exclude_tables=exclude_tables,
            )
            spec = downloader.generate_dataset_download_spec(dataset)

        if spec_hash is None:
            spec_hash = _hash_spec(spec)

        spec_file = Path(tmp_dir) / "download_spec.json"
        with spec_file.open("w", encoding="utf-8") as ds:
            json.dump(spec, ds)

        dataset._logger.info(
            "Downloading dataset %s for catalog: %s@%s"
            % (
                "minid" if use_minid else "bag",
                dataset.dataset_rid,
                str(version),
            )
        )

        if use_minid:
            # Server-side export: generates bag, uploads to S3, registers MINID.
            try:
                exporter = DerivaExport(
                    host=dataset._ml_instance.catalog.deriva_server.server,
                    config_file=spec_file,
                    output_dir=tmp_dir,
                    defer_download=True,
                    timeout=timeout or (10, 610),
                    envars={"RID": dataset.dataset_rid},
                )
                minid_page_url = exporter.export()[0]
            except (
                DerivaDownloadError,
                DerivaDownloadConfigurationError,
                DerivaDownloadAuthenticationError,
                DerivaDownloadAuthorizationError,
                DerivaDownloadTimeoutError,
            ) as e:
                raise DerivaMLException(format_exception(e))
            # Update version table with MINID and spec hash.
            version_path = (
                dataset._ml_instance.pathBuilder().schemas[dataset._ml_instance.ml_schema].tables["Dataset_Version"]
            )
            version_rid = [h for h in dataset.dataset_history() if str(h.dataset_version) == str(version)][
                0
            ].version_rid
            version_path.update([{"RID": version_rid, "Minid": minid_page_url, "Minid_Spec_Hash": spec_hash}])
            return minid_page_url
        else:
            # Client-side bag construction: drive CatalogBagBuilder.build()
            # via DatasetBagBuilder.build_bag against the snapshot catalog.
            # The pre-computed ``spec`` argument is vestigial on this arm
            # (CatalogBagBuilder recomputes its own spec from the same
            # anchors/policy); we keep the parameter on the function
            # signature because the MINID arm above still consumes it.
            #
            # The bag is built into a working subdirectory under
            # ``working_dir/client_export/`` (NOT ``tmp_dir``) so the zip
            # survives the ``TemporaryDirectory`` cleanup at the end of
            # this function — the caller (:func:`download_dataset_minid`)
            # consumes the file:// URI returned here. The cleanup path in
            # :func:`download_dataset_minid` recognizes ``client_export`` in
            # the archive's parent and removes it after extraction.
            version_snapshot_catalog = dataset._version_snapshot_catalog(version)
            builder = DatasetBagBuilder(
                ml_instance=version_snapshot_catalog,
                s3_bucket=dataset._ml_instance.s3_bucket,
                use_minid=False,
                exclude_tables=exclude_tables,
            )
            client_export_dir = Path(dataset._ml_instance.working_dir) / "client_export" / spec_hash[:8]
            client_export_dir.mkdir(parents=True, exist_ok=True)
            try:
                zip_path = builder.build_bag(dataset, output_dir=client_export_dir, timeout=timeout)
            except (
                DerivaDownloadError,
                DerivaDownloadConfigurationError,
                DerivaDownloadAuthenticationError,
                DerivaDownloadAuthorizationError,
                DerivaDownloadTimeoutError,
            ) as e:
                # Preserve the actionable-message contract callers relied on
                # from the legacy _create_dataset_bag_client. The original
                # advice (add direct dataset members for tables whose deep
                # FK joins timed out) still applies — surface it alongside
                # the deriva-py error so users have a fix to try.
                raise DerivaMLException(
                    f"Dataset bag export failed: {format_exception(e)}. "
                    "This typically happens when deep multi-table joins "
                    "exceed server query time limits. To fix this, add the "
                    "desired records as direct dataset members using "
                    "add_dataset_members() with the relevant table's RIDs."
                )

            return zip_path.as_uri()


def _resolve_version_record(dataset: "Dataset", version: DatasetVersion) -> Any:
    """Look up the ``DatasetHistory`` row for ``version`` on ``dataset``.

    Args:
        dataset: The dataset whose history to walk.
        version: The version to find.

    Returns:
        The ``DatasetHistory`` record carrying ``snapshot``, ``minid``,
        and ``spec_hash`` for the version.

    Raises:
        DerivaMLException: If no history record matches ``version``.
    """
    version_str = str(version)
    try:
        return next(v for v in dataset.dataset_history() if str(v.dataset_version) == version_str)
    except StopIteration:
        raise DerivaMLException(
            f"Version {version_str} does not exist for RID {dataset.dataset_rid}"
        ) from None


def _build_version_rid(dataset_rid: str, snapshot: str | None) -> str:
    """Format a version RID, omitting the ``@snap`` suffix when no snapshot.

    The ``DatasetMinid.RID`` pattern accepts both ``{rid}`` and
    ``{rid}@{snap}``; a literal ``{rid}@None`` from an f-string
    fails the pattern validator with a cryptic regex error.
    Centralised here because Tier 1 and Tier 3 both needed the
    guard pre-extraction.

    Args:
        dataset_rid: The dataset's RID.
        snapshot: Optional snapshot identifier; ``None`` is a
            legitimate value for in-progress datasets that have
            not yet been snapshotted.

    Returns:
        Either ``"{rid}"`` (when snapshot is ``None``) or
        ``"{rid}@{snapshot}"``.

    Example:
        >>> _build_version_rid("3WX", "2026-05-22T12:00:00")
        '3WX@2026-05-22T12:00:00'
        >>> _build_version_rid("3WX", None)
        '3WX'
    """
    if snapshot is None:
        return dataset_rid
    return f"{dataset_rid}@{snapshot}"


def _tier1_local_cache_lookup(
    dataset: "Dataset",
    version: DatasetVersion,
    cache_suffix: str,
    snapshot: str | None,
) -> DatasetMinid | None:
    """Check the local ``BagCacheIndex`` for a bag matching ``cache_suffix``.

    Returns the resolved :class:`DatasetMinid` on a hit, ``None``
    on a miss. The miss path is silent — callers fall through to
    Tier 2 or Tier 3.

    Args:
        dataset: The dataset whose cache to consult.
        version: The version being downloaded; copied into the
            returned :class:`DatasetMinid`.
        cache_suffix: ``{spec_hash[:16]}_{snapshot}`` — must match
            both the schema-derived spec hash and the data
            snapshot to count as a hit.
        snapshot: Same snapshot threaded through to
            :func:`_build_version_rid`.

    Returns:
        :class:`DatasetMinid` pointing at the cached bag, or
        ``None`` when no matching bag is on disk.
    """
    from deriva.bag.cache_index import BagCacheIndex

    index = BagCacheIndex(dataset._ml_instance.cache_dir)
    try:
        cached_checksums = index.find_bags_for_rid(table="Dataset", rid=dataset.dataset_rid)
    finally:
        index.dispose()

    if cache_suffix not in cached_checksums:
        return None

    # Re-open the index in a short scope to compute the bag path.
    cached_bag_path = (
        BagCacheIndex(dataset._ml_instance.cache_dir).bag_dir_for(cache_suffix)
        / f"Dataset_{dataset.dataset_rid}"
    )
    if not cached_bag_path.exists():
        return None

    dataset._logger.info(
        "Local cache hit for %s version %s (spec+snapshot match: %s)",
        dataset.dataset_rid,
        version,
        cache_suffix,
    )
    return DatasetMinid(
        dataset_version=version,
        RID=_build_version_rid(dataset.dataset_rid, snapshot),
        location=cached_bag_path.parent.as_uri(),
        checksums=[{"function": "sha256", "value": cache_suffix}],
    )


def _tier2_minid_path(
    dataset: "Dataset",
    version: DatasetVersion,
    version_record: Any,
    spec: Any,
    spec_hash: str,
    minid_url: str | None,
    create: bool,
    exclude_tables: set[str] | None,
    timeout: tuple[int, int] | None,
) -> DatasetMinid:
    """Tier 2 — fetch existing MINID or regenerate when spec drifted.

    Args:
        dataset: The dataset to resolve a bag for.
        version: The version being downloaded.
        version_record: The :class:`DatasetHistory` record from
            :func:`_resolve_version_record`.
        spec: Current download spec — passed to
            :func:`create_dataset_minid` if we need to regenerate.
        spec_hash: SHA-256 of ``spec``; compared against the
            stored ``Minid_Spec_Hash`` to detect schema drift.
        minid_url: Previously-registered MINID URL (if any).
        create: When ``False`` and no MINID is registered, raise
            instead of regenerating.
        exclude_tables: Threaded into bag regeneration.
        timeout: Threaded into bag regeneration.

    Returns:
        :class:`DatasetMinid` pointing at the MINID metadata URL.

    Raises:
        DerivaMLException: When ``create=False`` and no
            MINID is registered (or spec drift would require
            regeneration).
    """
    if minid_url and version_record.spec_hash == spec_hash:
        # S3 bag is current — download it (populates local cache for Tier 1).
        return fetch_minid_metadata(dataset, version, minid_url)

    # No MINID, or spec has changed — need to regenerate.
    if not create:
        raise DerivaMLException(f"Minid for dataset {dataset.dataset_rid} doesn't exist")
    if minid_url:
        dataset._logger.info(
            "Spec hash changed for dataset %s version %s — regenerating MINID bag.",
            dataset.dataset_rid,
            version,
        )
    else:
        dataset._logger.info("Creating new MINID for dataset %s", dataset.dataset_rid)
    new_minid_url = create_dataset_minid(
        dataset,
        version,
        use_minid=True,
        exclude_tables=exclude_tables,
        spec=spec,
        spec_hash=spec_hash,
        timeout=timeout,
    )
    return fetch_minid_metadata(dataset, version, new_minid_url)


def _tier3_client_path(
    dataset: "Dataset",
    version: DatasetVersion,
    spec: Any,
    spec_hash: str,
    snapshot: str | None,
    minid_url: str | None,
    create: bool,
    cache_suffix: str,
    exclude_tables: set[str] | None,
    timeout: tuple[int, int] | None,
) -> DatasetMinid:
    """Tier 3 — generate the bag client-side under the deterministic cache key.

    Args:
        dataset: The dataset to resolve a bag for.
        version: The version being downloaded.
        spec: Current download spec.
        spec_hash: SHA-256 of ``spec``; embedded in the returned
            checksum so Tier 1 finds the bag next time.
        snapshot: Threaded through to :func:`_build_version_rid`.
        minid_url: Existing MINID URL (if any) — only consulted
            for the ``create=False`` guard.
        create: When ``False`` and no MINID is registered, raise.
        cache_suffix: ``{spec_hash[:16]}_{snapshot}`` — used as
            both the bag's storage suffix and the returned
            checksum.
        exclude_tables: Threaded into bag generation.
        timeout: Threaded into bag generation.

    Returns:
        :class:`DatasetMinid` pointing at the generated bag URL.

    Raises:
        DerivaMLException: When ``create=False`` and no existing
            MINID URL is registered.
    """
    if not create and not minid_url:
        raise DerivaMLException(f"Minid for dataset {dataset.dataset_rid} doesn't exist")

    dataset._logger.info(
        "Cache miss for %s version %s — generating bag client-side",
        dataset.dataset_rid,
        version,
    )
    bag_url = create_dataset_minid(
        dataset,
        version,
        use_minid=False,
        exclude_tables=exclude_tables,
        spec=spec,
        spec_hash=spec_hash,
        timeout=timeout,
    )
    return DatasetMinid(
        dataset_version=version,
        RID=_build_version_rid(dataset.dataset_rid, snapshot),
        location=bag_url,
        checksums=[{"function": "sha256", "value": cache_suffix}],
    )


def get_dataset_minid(
    dataset: "Dataset",
    version: DatasetVersion,
    create: bool,
    use_minid: bool,
    exclude_tables: set[str] | None = None,
    timeout: tuple[int, int] | None = None,
) -> DatasetMinid | None:
    """Locate or create a dataset bag, using a three-tier caching strategy.

    The download algorithm proceeds through three tiers, stopping at the
    first success. This applies identically to both MINID and non-MINID paths:

    **Tier 1 — Local deterministic cache (filesystem lookup, no network beyond spec)**

    The cache key is ``{rid}_{spec_hash[:16]}_{snapshot}``, combining:
    - **spec_hash**: SHA-256 of the download spec (captures FK traversal plan).
      Changes when the schema changes (new tables, new FKs).
    - **snapshot**: Immutable catalog snapshot ID (captures data state).

    Both must match for a cache hit — a snapshot-only match would return
    stale bags created before schema changes.

    Cost: one schema introspection query (to compute spec_hash) + one stat.

    **Tier 2 — MINID / S3 (when ``use_minid=True``)**

    If no local cache exists, check whether a MINID (Minimal Viable
    Identifier) was previously registered for this version.  The stored
    ``Minid_Spec_Hash`` is compared to the current download spec hash:

    - Match → fetch MINID metadata (HTTP GET), download bag from S3.
    - Mismatch or missing → regenerate bag server-side, upload to S3,
      register new MINID.

    The spec hash detects schema or FK-path changes that would alter bag
    contents even for the same snapshot.

    **Tier 3 — Client-side bag generation (when ``use_minid=False``)**

    Build the bag locally by running ERMrest queries against the snapshot
    catalog. The bag is stored under ``{rid}_{spec_hash[:16]}_{snapshot}``
    so Tier 1 finds it on subsequent calls.

    Args:
        dataset: The dataset to look up / create a bag for.
        version: The dataset version to download.
        create: If True, create a new bag/MINID when none is cached.
            If False, raise an exception when nothing is available.
        use_minid: If True, use S3 + MINID service (Tier 2) on cache miss.
            If False, generate bag client-side (Tier 3) on cache miss.
        exclude_tables: Table names to exclude from FK path traversal.
        timeout: Optional (connect_timeout, read_timeout) in seconds.

    Returns:
        DatasetMinid with the bag URL (local ``file://`` or S3) and a
        checksum that doubles as the cache directory suffix.

    Raises:
        DerivaMLException: If the version doesn't exist, or if
            ``create=False`` and no cached/registered bag is available.
    """
    # Post Ds-minid extraction this function is the three-tier
    # dispatcher; the per-tier work lives in
    # ``_tier{1,2,3}_*`` helpers and the shared
    # ``_build_version_rid`` collapses the snapshot-vs-None guard
    # that used to be duplicated across Tier 1 and Tier 3.

    # 1. Resolve the version record (raises on miss).
    version_record = _resolve_version_record(dataset, version)
    snapshot = version_record.snapshot
    minid_url = version_record.minid

    # 2. Compute spec_hash upfront (required for all tiers).
    # Cost: one schema introspection query (no data queries).
    version_snapshot_catalog = dataset._version_snapshot_catalog(version)
    downloader = DatasetBagBuilder(
        ml_instance=version_snapshot_catalog,
        s3_bucket=dataset._ml_instance.s3_bucket,
        use_minid=use_minid,
        exclude_tables=exclude_tables,
    )
    spec = downloader.generate_dataset_download_spec(dataset)
    spec_hash = _hash_spec(spec)
    cache_suffix = f"{spec_hash[:16]}_{snapshot}"

    # 3. Tier 1 — local deterministic cache.
    cached = _tier1_local_cache_lookup(dataset, version, cache_suffix, snapshot)
    if cached is not None:
        return cached

    # 4. Tier 2 or Tier 3 dispatch on ``use_minid``.
    if use_minid:
        return _tier2_minid_path(
            dataset,
            version,
            version_record,
            spec,
            spec_hash,
            minid_url,
            create,
            exclude_tables,
            timeout,
        )
    return _tier3_client_path(
        dataset,
        version,
        spec,
        spec_hash,
        snapshot,
        minid_url,
        create,
        cache_suffix,
        exclude_tables,
        timeout,
    )


def materialize_bag_dir(
    bag_path: Path,
    *,
    fetch_concurrency: int = 1,
    logger: "Logger | None" = None,
) -> Path:
    """Fetch every ``fetch.txt`` entry for an already-extracted bag dir.

    Path-only — needs no catalog connection. ``fetch.txt`` carries
    absolute (Hatrac/S3) URLs, so materialization is a pure local
    operation over a bag that is already on disk. Idempotent: if the
    bag is already fully materialized, returns immediately without
    fetching.

    This is the shared fetch tail used by both
    :func:`materialize_dataset_bag` (after it has downloaded/extracted
    the bag) and :meth:`~deriva_ml.dataset.dataset_bag.DatasetBag.materialize`
    (which operates on a bag already on disk).

    Args:
        bag_path: Path to the extracted BDBag directory (parent of
            ``data/``).
        fetch_concurrency: Maximum number of concurrent file downloads.
        logger: Logger for progress messages. Defaults to this module's
            logger.

    Returns:
        ``Path`` to the bag directory (unchanged; only its contents grow).

    Raises:
        Exception: Propagates any error raised by ``bdb.materialize`` —
            e.g. a ``fetch.txt`` URL that is unreachable. The bag is
            left partially materialized in that case.
    """
    from deriva_ml.dataset.bag_cache import BagCache

    log = logger if logger is not None else _module_logger
    bag_path = Path(bag_path)

    def fetch_progress_callback(current, total):
        log.info(f"Materializing bag: {current} of {total} file(s) downloaded.")
        return True

    def validation_progress_callback(current, total):
        log.info(f"Validating bag: {current} of {total} file(s) validated.")
        return True

    # If the bag already has every fetch.txt entry resolved, skip the
    # materialize call — there's nothing to download.
    if BagCache._is_fully_materialized(bag_path):
        log.info(f"Bag at {bag_path} already materialized.")
        return bag_path

    log.info(f"Materializing bag at {bag_path}")
    # Ensure parent directories exist for all fetch entries.
    fetch_file = bag_path / "fetch.txt"
    if fetch_file.exists():
        with fetch_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    rel_path = parts[2]
                    (bag_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
    bdb.materialize(
        bag_path.as_posix(),
        fetch_callback=fetch_progress_callback,
        validation_callback=validation_progress_callback,
        fetch_concurrency=fetch_concurrency,
    )
    return bag_path


def materialize_dataset_bag(
    dataset: "Dataset",
    minid: DatasetMinid,
    use_minid: bool,
    fetch_concurrency: int = 1,
) -> Path:
    """Materialize a dataset bag by downloading all referenced files.

    This function downloads a BDBag (via :func:`download_dataset_minid`)
    and then "materializes" it by fetching all files referenced in the
    bag's ``fetch.txt`` manifest. This includes data files, assets, and
    any other content referenced by the bag.

    Progress is reported through callbacks that log to the dataset's
    logger.

    Materialization status is determined by directly inspecting whether
    every ``fetch.txt`` entry has a corresponding local file (via
    :meth:`BagCache._is_fully_materialized`) — there is no separate
    marker file. A cache that says "materialized" but is missing files
    is treated as not-materialized and re-fetched.

    Args:
        dataset: The dataset being materialized.
        minid: DatasetMinid containing the bag URL and metadata.
        use_minid: If True, download from S3 using the MINID URL.
        fetch_concurrency: Maximum number of concurrent file downloads
            during materialization.

    Returns:
        Path to the fully materialized bag directory.
    """
    bag_path = download_dataset_minid(dataset, minid, use_minid)
    return materialize_bag_dir(
        bag_path,
        fetch_concurrency=fetch_concurrency,
        logger=dataset._logger,
    )


__all__ = [
    "create_dataset_minid",
    "download_dataset_minid",
    "fetch_minid_metadata",
    "get_dataset_minid",
    "materialize_bag_dir",
    "materialize_dataset_bag",
]
