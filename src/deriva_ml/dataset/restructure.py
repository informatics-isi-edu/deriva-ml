"""Asset restructuring for downloaded :class:`DatasetBag` objects.

Extracted from :mod:`deriva_ml.dataset.dataset_bag` in Phase 3
(audit §3.B). The cluster takes a downloaded bag and reorganizes
its assets into a directory hierarchy suitable for third-party ML
trainers (e.g. ``torchvision.datasets.ImageFolder``).

The single public entry point :func:`restructure_assets` composes
five private helpers:

* :func:`_build_dataset_type_path_map` — RID → type path (e.g.
  ``["complete", "training"]``).
* :func:`_get_asset_dataset_mapping` — asset RID → most-specific
  containing dataset RID.
* :func:`_get_reachable_assets` — SQL-driven asset finder (walks
  the bag's local SQLite via SQLAlchemy).
* :func:`_detect_asset_table` — auto-detect the asset table when
  the caller didn't specify one.

Why a separate module? :class:`DatasetBag` was a 2200-LoC file
mixing access (queries, members, parents/children, features) with
restructuring (~600 LoC of directory-layout machinery for a single
public function). The audit (§3.B) flagged this as a structural
split. Keeping ``restructure_assets`` and its helpers in
``dataset_bag.py`` made the bag class read like a laundry list.

Functions are stateless and take a :class:`DatasetBag` instance as
their first argument. :class:`DatasetBag` keeps a thin sugar method
:meth:`DatasetBag.restructure_assets` that delegates here, so the
user-facing API ``bag.restructure_assets(...)`` is unchanged.

**On the ``DatasetLike`` question.** The audit suggested lifting
the cluster to take a ``DatasetLike`` protocol object rather than a
concrete :class:`DatasetBag`. After investigation that turned out to
be aspirational — :func:`_get_reachable_assets` reads from the bag's
local SQLite via ``bag.engine`` and ``bag._dataset_table_view``,
which are bag-specific. The orchestration helpers
(:func:`_build_dataset_type_path_map`,
:func:`_detect_asset_table`) *would* work on any ``DatasetLike``,
but cutting the cluster on that line would leave a sub-cluster on
:class:`DatasetBag` and an oddly-shaped module here. Taking a
:class:`DatasetBag` keeps the module symmetric with
:mod:`deriva_ml.dataset.bag_download` (which takes a :class:`Dataset`).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from sqlalchemy.orm import Session

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.feature import FeatureRecord

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset_bag import DatasetBag
    from deriva_ml.dataset.target_resolution import FeatureSelector

logger = logging.getLogger(__name__)


def _default_dir_name_from_target(
    target: "FeatureRecord | dict[str, FeatureRecord] | str | None",
    targets: "list[str] | dict[str, Any] | None",
) -> str:
    """Derive a directory name string from a resolved target without target_transform.

    For single-target with a single-column FeatureRecord: returns the value of the
    first non-FK, non-metadata column (the term column).

    For multi-target or multi-column features: raises DerivaMLException explaining
    that target_transform is required.

    For plain string values (column targets already converted to str): returns as-is.

    Args:
        target: The resolved target from _resolve_targets or column lookup.
        targets: The original targets spec (for error messages).

    Returns:
        Directory name string.

    Raises:
        DerivaMLException: When the target is a dict (multi-target case) and
            no target_transform was provided.
    """
    if target is None:
        return "Unknown"
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        # Multi-target — can't auto-derive a single string
        raise DerivaMLException(
            f"restructure_assets with multi-target {list(targets)!r} requires "
            f"target_transform to derive a single directory name. "
            f"Provide target_transform=lambda rec: ... that returns a str."
        )
    # Single FeatureRecord — find the first term/value column that has a value
    record_data = target.model_dump()
    # Skip well-known metadata columns to find the label column
    _skip_cols = {"RID", "RCT", "RMT", "RCB", "RMB", "Feature_Name", "Execution"}
    for col, val in record_data.items():
        if col in _skip_cols:
            continue
        if val is not None and not isinstance(val, (dict, list)):
            return str(val)
    return "Unknown"



def _build_dataset_type_path_map(
    bag: "DatasetBag",
    type_selector: Callable[[list[str]], str] | None = None,
) -> dict[RID, list[str]]:
    """Build a mapping from dataset RID to its type path in the hierarchy.

    Recursively traverses nested datasets to create a mapping where each
    dataset RID maps to its hierarchical type path (e.g., ["complete", "training"]).

    Args:
        type_selector: Function to select type when dataset has multiple types.
            Receives list of type names, returns selected type name.
            Defaults to selecting first type or "unknown" if no types.

    Returns:
        Dictionary mapping dataset RID to list of type names from root to leaf.
        e.g., {"4-ABC": ["complete", "training"], "4-DEF": ["complete", "testing"]}
    """
    if type_selector is None:
        def type_selector(types: list[str]) -> str:
            return types[0] if types else "Testing"

    type_paths: dict[RID, list[str]] = {}

    def traverse(dataset: DatasetBag, parent_path: list[str], visited: set[RID]) -> None:
        if dataset.dataset_rid in visited:
            return
        visited.add(dataset.dataset_rid)

        current_type = type_selector(dataset.dataset_types)
        # None means this dataset's type is structural/container (e.g. "Split")
        # and should not contribute a path component — traverse children
        # with the same parent_path so they get clean paths.
        if current_type is None:
            current_path = parent_path
        else:
            current_path = parent_path + [current_type]
        type_paths[dataset.dataset_rid] = current_path

        for child in dataset.list_dataset_children():
            traverse(child, current_path, visited)

    traverse(bag, [], set())
    return type_paths


def _get_asset_dataset_mapping(bag, asset_table: str) -> dict[RID, RID]:
    """Map asset RIDs to their containing dataset RID.

    For each asset in the specified table, determines which dataset it belongs to.
    This uses _dataset_table_view to find assets reachable through any FK path
    from the dataset, not just directly associated assets.

    Assets are mapped to their most specific (leaf) dataset in the hierarchy.
    For example, if a Split dataset contains Training and Testing children,
    and images are members of Training, the images map to Training (not Split).

    Args:
        asset_table: Name of the asset table (e.g., "Image")

    Returns:
        Dictionary mapping asset RID to the dataset RID that contains it.
    """
    asset_to_dataset: dict[RID, RID] = {}

    def collect_from_dataset(dataset: DatasetBag, visited: set[RID]) -> None:
        if dataset.dataset_rid in visited:
            return
        visited.add(dataset.dataset_rid)

        # Process children FIRST (depth-first) so leaf datasets get priority
        # This ensures assets are mapped to their most specific dataset
        for child in dataset.list_dataset_children():
            collect_from_dataset(child, visited)

        # Then process this dataset's assets
        # Only set if not already mapped (child/leaf dataset wins)
        for asset in _get_reachable_assets(dataset, asset_table):
            if asset["RID"] not in asset_to_dataset:
                asset_to_dataset[asset["RID"]] = dataset.dataset_rid

    collect_from_dataset(bag, set())
    return asset_to_dataset


def _get_reachable_assets(bag, asset_table: str) -> list[dict[str, Any]]:
    """Get all assets reachable from this dataset through any FK path.

    Unlike list_dataset_members which only returns directly associated entities,
    this method traverses foreign key relationships to find assets that are
    indirectly connected to the dataset. For example, if a dataset contains
    Subjects, and Subject -> Encounter -> Image, this method will find those
    Images even though they're not directly in the Dataset_Image association table.

    Args:
        asset_table: Name of the asset table (e.g., "Image")

    Returns:
        List of asset records as dictionaries.
    """
    # Use the _dataset_table_view query which traverses all FK paths
    sql_query = bag._dataset_table_view(asset_table)

    with Session(bag.engine) as session:
        result = session.execute(sql_query)
        # Convert rows to dictionaries
        rows = [dict(row._mapping) for row in result]

    return rows


def _detect_asset_table(bag) -> str | None:
    """Auto-detect the asset table from dataset members.

    Searches for asset tables in the dataset members by examining
    the schema. Returns the first asset table found, or None if
    no asset tables are in the dataset.

    Returns:
        Name of the detected asset table, or None if not found.
    """
    members = bag.list_dataset_members(recurse=True)
    for table_name in members:
        if table_name == "Dataset":
            continue
        # Check if this table is an asset table
        try:
            table = bag.model.name_to_table(table_name)
            if bag.model.is_asset(table):
                return table_name
        except (KeyError, AttributeError):
            continue
    return None


def restructure_assets(
    bag: "DatasetBag",
    output_dir: Path | str,
    *,
    asset_table: str | None = None,
    targets: "list[str] | dict[str, FeatureSelector] | None" = None,
    target_transform: Callable[..., str] | None = None,
    missing: Literal["error", "skip", "unknown"] = "unknown",
    use_symlinks: bool = True,
    type_selector: Callable[[list[str]], str] | None = None,
    type_to_dir_map: dict[str, str] | None = None,
    enforce_vocabulary: bool = True,
    file_transformer: Callable[[Path, Path], Path] | None = None,
) -> dict[Path, Path]:
    """Restructure downloaded assets into a directory hierarchy.

    Creates a directory structure organizing assets by dataset types and
    target label values. This is useful for ML workflows that expect data
    organized in conventional folder structures (e.g., PyTorch ImageFolder,
    ``torchvision.datasets.ImageFolder``).

    The dataset should be of type Training or Testing, or have nested
    children of those types. The top-level directory name is determined
    by the dataset type (e.g., ``"Training"`` → ``"training"``).

    **Finding assets through foreign key relationships:**

    Assets are found by traversing all foreign key paths from the dataset,
    not just direct associations. For example, if a dataset contains Subjects,
    and the schema has Subject → Encounter → Image relationships, this method
    will find all Images reachable through those paths even though they are
    not directly in a ``Dataset_Image`` association table.

    **Handling datasets without types (prediction scenarios):**

    If a dataset has no type defined, it is treated as Testing. This is
    common for prediction/inference scenarios where you want to apply a
    trained model to new unlabeled data.

    **Handling missing labels:**

    If an asset doesn't have a value for a requested target, the ``missing``
    parameter controls the behavior: ``"unknown"`` (default) places the asset
    in an ``"Unknown"`` directory; ``"skip"`` omits it from the output tree;
    ``"error"`` raises at construction time listing all unlabeled RIDs.

    Args:
        output_dir: Base directory for restructured assets.
        asset_table: Name of the asset table (e.g., ``"Image"``). If None,
            auto-detects from dataset members. Raises ``DerivaMLException``
            if multiple asset tables are found and none is specified.
        targets: Source of directory-naming label data. Three shapes:

            - ``None`` (default) — no label grouping. Assets are placed
              directly under the type-derived directory with no further
              subdirectory levels.
            - ``list[str]`` — feature names (or direct column names) to
              group by. Each name adds one subdirectory level. For a
              single feature the resolved ``FeatureRecord`` is passed to
              ``target_transform`` (if provided). For multiple features
              a ``dict[str, FeatureRecord]`` is passed.
            - ``dict[str, FeatureSelector]`` — feature names mapped to
              per-feature selector callables; passed verbatim to
              ``bag.feature_values(..., selector=...)``. Built-in
              selectors: ``FeatureRecord.select_newest``,
              ``select_first``, ``select_majority_vote(column)``.

            Column names (direct columns on the asset table, not features)
            are resolved via column lookup on the asset record. They are
            converted to strings for the directory name; ``target_transform``
            receives the raw column value (as a string).

            Dotted ``"Feature.column"`` syntax from earlier releases is
            **removed** — pass it as a target string with ``target_transform``
            instead (see Migration note below).

        target_transform: ``Callable`` consuming the resolved label shape
            (a ``FeatureRecord`` for single-target, ``dict[str, FeatureRecord]``
            for multi-target, or the raw column value string for column
            targets) and **returning a ``str``** used as the subdirectory
            name. No-op default derives the name from the feature's primary
            value column (single-target) or raises a clear error explaining
            that ``target_transform`` is required for multi-target and
            multi-column feature cases.

            Runtime constraint: the return type is checked at the first
            call; a non-``str`` return raises ``DerivaMLValidationError``
            with a message explaining the requirement.

        missing: Behavior when a target value is absent for an asset:

            - ``"unknown"`` (default) — place the asset in an ``Unknown/``
              subdirectory. Preserves the pre-alignment behavior.
            - ``"skip"`` — omit the asset from the output tree entirely.
              New behavior; no directory is created for it.
            - ``"error"`` — raise ``DerivaMLException`` at construction
              time listing unlabeled RIDs. Useful for ensuring training
              data is complete before committing to disk.

        use_symlinks: If True (default), create symlinks to original files.
            If False, copy files. Symlinks save disk space but require
            the original bag to remain in place. Ignored when
            ``file_transformer`` is provided.
        type_selector: Function to select type when dataset has multiple types.
            Receives list of type names, returns selected type name.
            Defaults to selecting first type or ``"testing"`` if no types.
        type_to_dir_map: Optional mapping from dataset type names to directory
            names. Defaults to ``{"Training": "training", "Testing": "testing",
            "Unknown": "unknown"}``. Use this to customize directory names or
            add new type mappings.
        enforce_vocabulary: If True (default), only allow features that have
            controlled vocabulary term columns, and raise an error if an asset
            has multiple different values for the same feature. Set to False
            to allow non-vocabulary features and use the first value when
            multiple exist.
        file_transformer: Optional callable invoked instead of the default
            symlink/copy step. Receives ``(src_path, dest_path)`` where
            ``dest_path`` is the suggested destination (preserving the original
            filename and extension). The transformer is responsible for writing
            the output file — it may change the extension or format — and must
            return the actual ``Path`` it wrote. When provided, ``use_symlinks``
            is ignored.

            Example — convert DICOM to PNG on placement::

                def oct_to_png(src: Path, dest: Path) -> Path:
                    img = load_oct_dcm(str(src))
                    out = dest.with_suffix(".png")
                    PILImage.fromarray((img * 255).astype(np.uint8)).save(out)
                    return out

                bag.restructure_assets(
                    output_dir="./ml_data",
                    targets=["Diagnosis"],
                    file_transformer=oct_to_png,
                )

    Returns:
        Manifest dict mapping each source ``Path`` to the actual output
        ``Path`` written. When no ``file_transformer`` is provided, source
        and output paths differ only in directory location. When a
        transformer is provided, the output path may also differ in name
        or extension.

    Raises:
        DerivaMLException: If ``asset_table`` cannot be determined, if
            ``missing="error"`` and any asset lacks a target value, or if
            ``enforce_vocabulary=True`` and a feature has no vocabulary
            term columns.
        DerivaMLValidationError: If ``target_transform`` returns a
            non-``str`` value, or if a dotted ``"Feature.column"`` string
            is passed in ``targets``.

    Examples:
        Basic restructuring with auto-detected asset table::

            manifest = bag.restructure_assets(
                output_dir="./ml_data",
                targets=["Diagnosis"],
            )
            # Creates:
            # ./ml_data/training/Normal/image1.jpg
            # ./ml_data/testing/Abnormal/image2.jpg

        Custom type-to-directory mapping::

            manifest = bag.restructure_assets(
                output_dir="./ml_data",
                targets=["Diagnosis"],
                type_to_dir_map={"Training": "train", "Testing": "test"},
            )
            # Creates:
            # ./ml_data/train/Normal/image1.jpg
            # ./ml_data/test/Abnormal/image2.jpg

        Per-feature selector for multi-annotator datasets::

            from deriva_ml.feature import FeatureRecord

            manifest = bag.restructure_assets(
                output_dir="./ml_data",
                targets={"Diagnosis": FeatureRecord.select_newest},
            )

        Extract a specific column from a multi-column feature::

            manifest = bag.restructure_assets(
                output_dir="./ml_data",
                targets=["Classification"],
                target_transform=lambda rec: rec.Label,
            )

        Convert DICOM files to PNG during restructuring::

            from PIL import Image as PILImage

            def oct_to_png(src: Path, dest: Path) -> Path:
                img = load_oct_dcm(str(src))
                out = dest.with_suffix(".png")
                PILImage.fromarray((img * 255).astype(np.uint8)).save(out)
                return out

            manifest = bag.restructure_assets(
                output_dir="./ml_data",
                asset_table="OCT_DICOM",
                targets=["Diagnosis"],
                type_to_dir_map={"Training": "train", "Testing": "test"},
                file_transformer=oct_to_png,
            )

    Note:
        Migration note (from pre-D2 signature):

        - ``group_by=["Diagnosis"]`` → ``targets=["Diagnosis"]``
        - ``group_by=["Classification.Label"]`` →
          ``targets=["Classification"], target_transform=lambda rec: rec.Label``
        - ``value_selector=FeatureRecord.select_newest`` →
          ``targets={"Feature": FeatureRecord.select_newest}``

    See Also:
        ``DatasetBag.as_torch_dataset``, ``DatasetBag.as_tf_dataset``:
            Framework adapters. Use these when you want lazy in-place
            iteration and do NOT need a class-folder directory tree.
            They share the same ``targets`` / ``target_transform`` /
            ``missing`` vocabulary as ``restructure_assets``. The two
            paths are alternatives, not a pipeline — pick one per the
            User Guide "How to feed a bag to a training framework".
    """
    from deriva_ml.core.exceptions import DerivaMLValidationError
    from deriva_ml.dataset.target_resolution import _resolve_targets

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate targets: dotted "Feature.column" syntax is removed
    if targets is not None:
        target_names = list(targets) if isinstance(targets, dict) else targets
        for t in target_names:
            if "." in t:
                raise DerivaMLValidationError(
                    f"Dotted target syntax {t!r} is no longer supported. "
                    f"Replace with: targets=[{t.split('.')[0]!r}], "
                    f"target_transform=lambda rec: rec.{t.split('.')[1]}"
                )

    # Default type-to-directory mapping
    if type_to_dir_map is None:
        type_to_dir_map = {"Training": "training", "Testing": "testing", "Unknown": "unknown"}

    # Auto-detect asset table if not provided
    if asset_table is None:
        asset_table = _detect_asset_table(bag)
        if asset_table is None:
            raise DerivaMLException(
                "Could not auto-detect asset table. No asset tables found in dataset members. "
                "Specify the asset_table parameter explicitly."
            )
        logger.info(f"Auto-detected asset table: {asset_table}")

    # Step 1: Build dataset type path map with directory name mapping
    def map_type_to_dir(types: list[str]) -> str | None:
        """Map dataset types to directory name using type_to_dir_map.

        If dataset has no types, treat it as Testing (prediction use case).
        Returns None when the type is not in type_to_dir_map, signalling
        that this dataset is a structural container (e.g. a Split parent)
        and should not contribute a path component. Its children will
        still be traversed and their own types will determine the path.
        """
        if not types:
            # No types defined - treat as Testing for prediction scenarios
            return type_to_dir_map.get("Testing", "testing")
        if type_selector:
            selected_type = type_selector(types)
        else:
            selected_type = types[0]
        if selected_type in type_to_dir_map:
            return type_to_dir_map[selected_type]
        # Type not explicitly mapped — treat as transparent container
        return None

    type_path_map = _build_dataset_type_path_map(bag, map_type_to_dir)

    # Step 2: Get asset-to-dataset mapping
    asset_dataset_map = _get_asset_dataset_mapping(bag, asset_table)

    # Step 3: Separate feature-based targets from column-based targets.
    # _resolve_targets only works with features (via bag.feature_values).
    # Direct column names on the asset table must be handled separately.
    feature_target_map: dict[str, Any] = {}
    column_targets: list[str] = []
    feature_targets_spec: "list[str] | dict[str, Any] | None" = None

    if targets is not None:
        target_names_list = list(targets) if isinstance(targets, dict) else list(targets)

        # Classify each target as feature or column by probing lookup_feature.
        feature_names: list[str] = []
        for t_name in target_names_list:
            try:
                bag.lookup_feature(asset_table, t_name)
                feature_names.append(t_name)
            except Exception:
                # Not a recognized feature on asset_table — treat as column
                column_targets.append(t_name)

        # Build the feature-only targets spec to pass to _resolve_targets
        if feature_names:
            if isinstance(targets, dict):
                feature_targets_spec = {k: v for k, v in targets.items() if k in feature_names}
            else:
                feature_targets_spec = feature_names

        # Call _resolve_targets only for feature-based targets.
        # ``enforce_vocabulary`` is enforced inside _resolve_targets so the
        # validation happens once at the same boundary as the resolution.
        if feature_targets_spec:
            feature_target_map = _resolve_targets(
                bag,
                asset_table,
                targets=feature_targets_spec,
                missing=missing,
                enforce_vocabulary=enforce_vocabulary,
            )

    # Step 4: For column-based targets, load a simple {rid: value_str} map
    # by scanning all asset records once.
    column_value_map: dict[str, dict[str, str]] = {col: {} for col in column_targets}

    # Step 6: Get all assets reachable through FK paths
    assets = _get_reachable_assets(bag, asset_table)

    manifest: dict[Path, Path] = {}

    if not assets:
        logger.warning(f"No assets found in table '{asset_table}'")
        return manifest

    # Populate column_value_map from the asset records
    for asset in assets:
        for col in column_targets:
            val = asset.get(col)
            if val is not None:
                column_value_map[col][asset["RID"]] = str(val)

    # Step 7: Process each asset
    for asset in assets:
        asset_rid = asset.get("RID")

        # Get source file path
        filename = asset.get("Filename")
        if not filename:
            logger.warning(f"Asset {asset_rid} has no Filename")
            continue

        source_path = Path(filename)
        if not source_path.exists():
            # Filename may be a bare basename stored in the SQLite cache
            # before image materialization.  Fall back to the canonical
            # BDBag asset layout: data/asset/{RID}/{table}/{filename}.
            try:
                bag_root = Path(bag._catalog._database_model.bag_path)
                source_path = bag_root / "data" / "asset" / asset.get("RID", "") / asset_table / Path(filename).name
            except AttributeError:
                pass  # catalog doesn't have _database_model (e.g. in tests)

        if not source_path.exists():
            logger.warning(f"Asset file not found: {filename}")
            continue

        # Get dataset type path
        dataset_rid = asset_dataset_map.get(asset_rid)
        type_path = type_path_map.get(dataset_rid, ["unknown"])

        # Resolve grouping path components from targets.
        # Each target name contributes one directory level. Target names are
        # processed in order (column targets get column lookup; feature targets
        # get resolution from feature_target_map built by _resolve_targets above).
        group_path: list[str] = []
        skip_asset = False

        if targets is not None:
            target_names_list = list(targets) if isinstance(targets, dict) else list(targets)
            # For single-feature target, feature_target_map[rid] is FeatureRecord|None.
            # For multi-feature target, feature_target_map[rid] is dict[str, FeatureRecord].
            # We unpack accordingly when passing to target_transform or _default_dir_name_from_target.
            feature_raw = feature_target_map.get(asset_rid)

            # If the RID is absent from the feature_target_map entirely AND there are
            # feature targets, apply the missing policy at the group level.
            if feature_targets_spec and asset_rid not in feature_target_map:
                if missing == "skip":
                    skip_asset = True
                elif missing == "error":
                    raise DerivaMLException(
                        f"Asset {asset_rid!r} has no value for target feature(s). "
                        f"Pass missing='skip' to drop unlabeled assets, or "
                        f"missing='unknown' to place them in Unknown/."
                    )
                else:
                    # missing="unknown": place in Unknown dir
                    # Build partial path from column targets, then add Unknown for features
                    for t_name in target_names_list:
                        if t_name in column_targets:
                            col_val = column_value_map.get(t_name, {}).get(asset_rid)
                            group_path.append(str(col_val) if col_val is not None else "Unknown")
                        else:
                            group_path.append("Unknown")
            elif not skip_asset:
                # Normal path: resolve each target in order
                for t_name in target_names_list:
                    if t_name in column_targets:
                        # Column-based target
                        col_val = column_value_map.get(t_name, {}).get(asset_rid)
                        if col_val is not None:
                            if target_transform is not None:
                                dir_name = target_transform(col_val)
                                if not isinstance(dir_name, str):
                                    raise DerivaMLValidationError(
                                        f"restructure_assets target_transform must return str "
                                        f"(for directory naming); got {type(dir_name).__name__}"
                                    )
                            else:
                                dir_name = col_val
                        else:
                            # Missing column value
                            if missing == "error":
                                raise DerivaMLException(
                                    f"Asset {asset_rid!r} has no value for column target {t_name!r}. "
                                    f"Pass missing='skip' to drop unlabeled assets, or "
                                    f"missing='unknown' to place them in Unknown/."
                                )
                            elif missing == "skip":
                                skip_asset = True
                                break
                            else:
                                dir_name = "Unknown"
                        group_path.append(dir_name)
                    else:
                        # Feature-based target: look up from feature_target_map
                        # feature_raw is either FeatureRecord (single) or
                        # dict[str, FeatureRecord] (multi-feature). Extract for this feature.
                        if feature_raw is None:
                            per_feature_val = None
                        elif isinstance(feature_raw, dict):
                            per_feature_val = feature_raw.get(t_name)
                        else:
                            # Single feature target — feature_raw IS the FeatureRecord
                            per_feature_val = feature_raw

                        if per_feature_val is None:
                            dir_name = "Unknown"
                        elif target_transform is not None:
                            dir_name = target_transform(per_feature_val)
                            if not isinstance(dir_name, str):
                                raise DerivaMLValidationError(
                                    f"restructure_assets target_transform must return str "
                                    f"(for directory naming); got {type(dir_name).__name__}"
                                )
                        else:
                            dir_name = _default_dir_name_from_target(per_feature_val, [t_name])
                        group_path.append(dir_name)

        if skip_asset:
            continue

        # Build target directory
        target_dir = output_dir.joinpath(*type_path, *group_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Suggested destination preserves the original filename
        target_path = target_dir / source_path.name

        # Handle existing files at the suggested destination
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()

        if file_transformer is not None:
            # Transformer is responsible for writing the output file.
            # It receives the suggested dest and returns the actual path written,
            # which may differ in name or extension (e.g. DICOM -> PNG).
            actual_path = file_transformer(source_path, target_path)
        elif use_symlinks:
            try:
                target_path.symlink_to(source_path.resolve())
            except OSError as e:
                # Fall back to copy on platforms that don't support symlinks
                logger.warning(f"Symlink failed, falling back to copy: {e}")
                shutil.copy2(source_path, target_path)
            actual_path = target_path
        else:
            shutil.copy2(source_path, target_path)
            actual_path = target_path

        manifest[source_path] = actual_path

    return manifest




__all__ = ["restructure_assets"]
