"""Bag-pipeline clone path (catalog → bag → catalog).

:func:`clone_via_bag` is the bag-oriented replacement for the
bespoke :func:`~deriva_ml.catalog.clone.create_ml_workspace` flow.
Per ADR-0006, catalog cloning becomes a two-step pipeline:

1. :class:`~deriva.bag.catalog_builder.CatalogBagBuilder` walks
   the source catalog and writes a deriva-bag profile bag.
2. :class:`~deriva.bag.catalog_loader.BagCatalogLoader` walks the
   bag's SQLAlchemy mirror and inserts rows into the destination
   catalog in FK-safe order, with assets handled per the
   :class:`~deriva.bag.traversal.AssetMode` policy.

The bag in the middle is a real on-disk artifact: debuggable,
inspectable, citable via MINID, and re-loadable if the destination
push fails mid-way.

This function is the **new** clone path. The legacy
:func:`create_ml_workspace` stays in place during the transition
because it carries production-tested behavior the new path
doesn't yet replicate (oversized-value truncation, async per-table
concurrency, index rebuild on size-limit failure). Use
``clone_via_bag`` for new code and for catalogs where the legacy
path's features aren't needed; use ``create_ml_workspace`` when
the legacy parameters matter.

Feature parity tracking:

==========================  =================  ====================
Legacy parameter            Bag-path mapping   Notes
==========================  =================  ====================
``root_rid``                ``RIDAnchor``      Mapped: builds an
                                               anchor list with
                                               the root RID.
``include_tables``          ``policy.schemas`` Mapped via schema
                                               allow-list when
                                               specified.
``exclude_objects``         ``exclude_tables`` Mapped.
``exclude_schemas``         ``exclude_schemas``Mapped.
``asset_mode``              ``AssetMode``      Mapped: REFERENCES
                                               → ROWS_ONLY,
                                               FULL → UPLOAD_IF_MISSING.
``orphan_strategy``         ``dangling_fk_strategy`` Mapped 1:1.
``prune_hidden_fkeys``      n/a                Legacy-only.
``truncate_oversized``      n/a                Legacy-only.
``table_concurrency``       n/a                Engine-internal.
``copy_annotations``        n/a                Bag profile carries
                                               them implicitly via
                                               schema.json.
==========================  =================  ====================
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

from deriva.bag.anchors import Anchor, RIDAnchor
from deriva.bag.catalog_builder import CatalogBagBuilder
from deriva.bag.catalog_loader import BagCatalogLoader, LoadReport
from deriva.bag.traversal import (
    AssetMode,
    FKTraversalPolicy,
)
from deriva.core import DerivaServer, get_credential

logger = logging.getLogger(__name__)


def _materialize_bag_dir(bag_path: Path) -> Path:
    """Return an on-disk unpacked bag directory, extracting a zip if needed.

    Upstream ``CatalogBagBuilder._run_export`` hard-codes
    ``bag_archiver=zip`` and removes the unpacked directory after
    archiving, but returns the unpacked path as if it still existed.
    Recover the bag by extracting ``{bag_path}.zip`` into
    ``bag_path.parent`` when the directory is missing.

    Args:
        bag_path: Path the builder claimed it produced.

    Returns:
        Path to an extracted bag directory. ``bag_path`` itself when
        the builder already left it on disk; otherwise the directory
        produced by unzipping ``{bag_path}.zip``.

    Raises:
        FileNotFoundError: When neither the directory nor the zip is
            present — the build genuinely failed.
    """
    if bag_path.is_dir():
        return bag_path
    zip_candidate = bag_path.with_suffix(".zip")
    if not zip_candidate.exists():
        raise FileNotFoundError(f"Bag missing at {bag_path}; no {zip_candidate} fallback")
    extract_root = bag_path.parent
    with zipfile.ZipFile(zip_candidate) as zf:
        zf.extractall(extract_root)
    # bdbag produces a single top-level directory inside the zip;
    # use the builder's expected name when it landed, otherwise
    # take the lone extracted directory.
    if bag_path.is_dir():
        return bag_path
    extracted = [p for p in extract_root.iterdir() if p.is_dir() and p.name != ".bag-db"]
    if len(extracted) != 1:
        raise FileNotFoundError(
            f"Could not locate bag directory after unzipping {zip_candidate}; candidates: {[p.name for p in extracted]}"
        )
    return extracted[0]


def _materialize_bag_assets(bag_path: Path) -> None:
    """Fetch any unresolved ``fetch.txt`` entries into the bag.

    Asset uploads (``AssetMode.UPLOAD_IF_MISSING`` /
    ``UPLOAD_FORCE``) require the bytes to be present locally —
    the loader pushes them to the destination Hatrac. When the
    source bag was produced with the deriva-bag profile, asset
    payloads live as URL references in ``fetch.txt`` until a
    materialize step copies them in. Skipping this step would
    surface as ``ValueError: asset_mode=... is incompatible with
    a holey bag`` from
    :meth:`FKTraversalPolicy.validate_with_bag_state`.

    Args:
        bag_path: Path to the bag directory to materialize.
    """
    fetch_file = bag_path / "fetch.txt"
    if not fetch_file.exists() or fetch_file.stat().st_size == 0:
        return
    # Local import — bdbag drags in heavy network deps; keep it
    # lazy so callers who never hit an asset-upload path don't pay.
    from bdbag import bdbag_api as bdb

    logger.info("clone_via_bag: materializing bag assets at %s", bag_path)
    bdb.materialize(str(bag_path))


@dataclass
class CloneViaBagResult:
    """Outcome of a :func:`clone_via_bag` invocation.

    Attributes:
        source_catalog_id: ID of the source catalog the bag was
            built from.
        dest_catalog_id: ID of the destination catalog the bag was
            loaded into.
        bag_path: Path to the bag directory that bridged the two.
            Left on disk after the clone so the artifact is
            inspectable / re-usable.
        load_report: Per-table load statistics from
            :class:`BagCatalogLoader`.
    """

    source_catalog_id: str
    dest_catalog_id: str
    bag_path: Path
    load_report: LoadReport


def clone_via_bag(
    *,
    source_hostname: str,
    source_catalog_id: str,
    dest_hostname: str,
    dest_catalog_id: str,
    anchors: list[Anchor] | None = None,
    root_rid: str | None = None,
    output_dir: Path | None = None,
    policy: FKTraversalPolicy | None = None,
    source_credential: dict | None = None,
    dest_credential: dict | None = None,
) -> CloneViaBagResult:
    """Clone catalog content from source → bag → destination.

    Two-step pipeline:

    1. :class:`CatalogBagBuilder` writes a bag from the source.
    2. :class:`BagCatalogLoader` loads the bag into the destination.

    Args:
        source_hostname: Hostname of the source ERMrest server.
        source_catalog_id: ID of the catalog to read from.
        dest_hostname: Hostname of the destination ERMrest server.
        dest_catalog_id: ID of the catalog to write to. Must
            already exist with a compatible schema.
        anchors: Starting points for the catalog walk. When
            ``None``, ``root_rid`` is converted into a single
            :class:`RIDAnchor` on the ``Dataset`` table for
            convenience with the legacy use case. At least one of
            ``anchors`` / ``root_rid`` must be provided.
        root_rid: Convenience parameter — equivalent to passing
            ``anchors=[RIDAnchor(table="Dataset", rids=[root_rid])]``.
        output_dir: Directory the intermediate bag lives in. When
            ``None``, defaults to ``./clone-{source_catalog_id}-to-{dest_catalog_id}/``
            under the current working directory.
        policy: :class:`FKTraversalPolicy` controlling the walk
            and load. Defaults are sensible for the common case;
            override ``asset_mode`` / ``dangling_fk_strategy`` /
            ``exclude_tables`` for production scenarios.
        source_credential: Optional credential dict for the source
            catalog. When ``None``, looked up via
            :func:`deriva.core.get_credential`.
        dest_credential: Same as ``source_credential`` but for the
            destination.

    Returns:
        :class:`CloneViaBagResult` carrying the resulting bag path
        and the loader's per-table stats.

    Raises:
        ValueError: If neither ``anchors`` nor ``root_rid`` is
            provided.

    Example:
        Clone a slice rooted at a Dataset RID, using the default
        policy (UPLOAD_IF_MISSING for assets, FAIL on orphans)::

            >>> from deriva_ml.catalog.clone_via_bag import clone_via_bag
            >>> result = clone_via_bag(  # doctest: +SKIP
            ...     source_hostname="src.example.org",
            ...     source_catalog_id="1",
            ...     dest_hostname="dst.example.org",
            ...     dest_catalog_id="42",
            ...     root_rid="1-ABCD",
            ... )
            >>> result.load_report.total_rows_inserted  # doctest: +SKIP
            12345
    """
    if anchors is None and root_rid is None:
        raise ValueError("clone_via_bag requires either ``anchors`` or ``root_rid``")

    if anchors is None:
        # Convenience path: caller provided a single Dataset RID.
        # The bag walker resolves the schema for the table name
        # via the source catalog's model.
        assert root_rid is not None  # narrowed by the check above
        anchors = [RIDAnchor(table="Dataset", rids=[root_rid])]

    policy = policy or FKTraversalPolicy()

    if output_dir is None:
        output_dir = Path.cwd() / f"clone-{source_catalog_id}-to-{dest_catalog_id}"
    output_dir = Path(output_dir)

    # Connect to the source catalog.
    source_creds = source_credential or get_credential(source_hostname)
    source_server = DerivaServer("https", source_hostname, credentials=source_creds)
    source_catalog = source_server.connect_ermrest(source_catalog_id)

    # Build the bag. CatalogBagBuilder drives deriva-py's export
    # engine, which already handles paged ERMrest queries, MD5
    # manifest generation, and BDBag finalization.
    builder = CatalogBagBuilder(
        catalog=source_catalog,
        anchors=anchors,
        output_dir=output_dir,
        policy=policy,
        producer="deriva_ml.catalog.clone_via_bag",
    )
    logger.info(
        "clone_via_bag: building bag at %s from source %s/%s",
        output_dir,
        source_hostname,
        source_catalog_id,
    )
    bag_path = builder.build()
    # CatalogBagBuilder archives the bag as ``{bag_path}.zip`` and
    # leaves only the zip on disk; materialize back into a directory
    # the loader can open.
    bag_path = _materialize_bag_dir(bag_path)

    # Materialize asset payloads when the policy will upload them.
    # An asset-uploading mode (``UPLOAD_IF_MISSING`` / ``UPLOAD_FORCE``)
    # requires the bytes to already be local; the loader's
    # ``validate_with_bag_state`` rejects the combination up front.
    # ``ROWS_ONLY`` skips this — the bag's row data alone is enough.
    if policy.asset_mode in (
        AssetMode.UPLOAD_IF_MISSING,
        AssetMode.UPLOAD_FORCE,
    ):
        _materialize_bag_assets(bag_path)

    # Connect to the destination catalog.
    dest_creds = dest_credential or get_credential(dest_hostname)
    dest_server = DerivaServer("https", dest_hostname, credentials=dest_creds)
    dest_catalog = dest_server.connect_ermrest(dest_catalog_id)

    # Load the bag into the destination. BagCatalogLoader walks
    # the bag's SQLAlchemy mirror in FK-safe order and applies
    # the policy's dangling_fk_strategy + asset_mode.
    logger.info(
        "clone_via_bag: loading bag into dest %s/%s",
        dest_hostname,
        dest_catalog_id,
    )
    with BagCatalogLoader(
        catalog=dest_catalog,
        bag=bag_path,
        policy=policy,
    ) as loader:
        report = loader.run()

    return CloneViaBagResult(
        source_catalog_id=source_catalog_id,
        dest_catalog_id=dest_catalog_id,
        bag_path=bag_path,
        load_report=report,
    )


__all__ = ["CloneViaBagResult", "clone_via_bag"]
