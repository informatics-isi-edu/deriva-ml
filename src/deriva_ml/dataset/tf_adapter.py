"""TensorFlow adapter for DatasetBag.

Builds a ``tf.data.Dataset`` that lazy-loads samples and labels from an
already-downloaded ``DatasetBag``. TensorFlow is imported at builder
entry (lazy import) so the base library stays importable without
TensorFlow installed.

See design spec §§3.1-3.7 for the full contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.target_resolution import _resolve_targets, resolve_element_rids

if TYPE_CHECKING:
    import tensorflow as tf

    from deriva_ml.dataset.dataset_bag import DatasetBag
    from deriva_ml.feature import FeatureRecord

    FeatureSelector = Callable[[list["FeatureRecord"]], "FeatureRecord | None"]


_TF_INSTALL_HINT = (
    "TensorFlow is not installed. Install with:\n"
    "    pip install 'deriva-ml[tf]'\n"
    "or install tensorflow directly:\n"
    "    pip install 'tensorflow>=2.15'\n"
    "macOS: use 'tensorflow-macos' instead of 'tensorflow'.\n"
    "CUDA:  use 'tensorflow[and-cuda]' for GPU support."
)


def build_tf_dataset(
    bag: "DatasetBag",
    element_type: str,
    *,
    sample_loader: "Callable[[Path | None, dict[str, Any]], Any] | None" = None,
    transform: "Callable[[Any], Any] | None" = None,
    targets: "list[str] | dict[str, FeatureSelector] | None" = None,
    target_transform: "Callable[..., Any] | None" = None,
    missing: Literal["error", "skip", "unknown"] = "error",
    output_signature: "tf.TensorSpec | tuple | None" = None,
    reachable: bool = True,
) -> "tf.data.Dataset":
    """Build a tf.data.Dataset from a DatasetBag.

    See DatasetBag.as_tf_dataset docstring for full argument docs
    and spec §3 for the design contract.

    Args:
        bag: The source DatasetBag to build the dataset from.
        element_type: Name of the domain table whose rows become samples.
        sample_loader: Callable receiving ``(path, row_dict)``. For asset
            tables, ``path`` is the on-disk file path; for non-asset tables,
            ``path=None``. Required for asset-table element types.
        transform: Applied to the sample after ``sample_loader`` returns.
        targets: Feature names that supply labels. ``None`` yields unlabeled
            samples. ``list[str]`` yields one FeatureRecord per element.
            ``dict[str, selector]`` passes per-feature selectors.
        target_transform: Callable consuming the raw feature-record shape
            (FeatureRecord, dict, or None) and returning the final label.
        missing: Policy for elements with no feature value. ``"error"``
            raises at construction; ``"skip"`` drops unlabeled elements;
            ``"unknown"`` keeps them with target=None.
        output_signature: ``tf.TypeSpec`` (or nested structure of specs)
            describing the shape and dtype of each element yielded by the
            generator. When ``None`` (default), the first sample is consumed
            eagerly to infer the signature via ``tf.type_spec_from_value``,
            then the generator is re-wrapped so the first sample is not lost.
        reachable: ``True`` (default) enumerates every ``element_type`` row
            FK-reachable from this dataset — matching ``restructure_assets``,
            so an ``Image`` reachable only via ``Subject -> Observation ->
            Image`` is still included. ``False`` restricts to direct dataset
            members (``list_dataset_members(recurse=True)``), the opt-out.

    Returns:
        A ``tf.data.Dataset`` whose elements are:

        - ``(sample, rid)`` when ``targets=None`` (unlabeled).
        - ``(sample, target, rid)`` when ``targets`` is set (labeled).

        The element's RID is always the last positional value. It's
        passed through raw — never touched by ``transform`` or
        ``target_transform`` — because the RID is the row's catalog
        identity, not a feature value. The motivating use case is
        downstream code that records per-element predictions back to
        the catalog (the RID is the FK target for the feature row).

    Raises:
        ImportError: If TensorFlow is not installed.
        DerivaMLException: If element_type is not in the bag, if
            element_type is an asset table and sample_loader is None,
            if missing="error" and any element lacks a feature value, or
            if output_signature=None and the dataset is empty (cannot
            infer signature).
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(_TF_INSTALL_HINT) from e

    # Enumerate the element RIDs belonging to this dataset. reachable=True
    # (default) finds FK-reachable elements (e.g. Image via Subject -> ... ->
    # Image), matching restructure_assets / bag_info; reachable=False is the
    # direct-members-only opt-out. Raises if element_type is unresolvable.
    all_rids = resolve_element_rids(bag, element_type, reachable=reachable)

    # Validate sample_loader is provided for asset-table element types.
    is_asset = _bag_element_is_asset(bag, element_type)
    if is_asset and sample_loader is None:
        raise DerivaMLException(
            f"Element type {element_type!r} is an asset table and requires "
            f"a sample_loader. Common loaders:\n"
            f"    sample_loader=PIL.Image.open       # images\n"
            f"    sample_loader=nibabel.load         # NIfTI medical volumes\n"
            f"    sample_loader=h5py.File            # HDF5 arrays\n"
            f"Or pass sample_loader=lambda path, row: path.read_bytes() "
            f"if you want raw bytes."
        )

    # Resolve targets once at construction time.
    target_map = _resolve_targets(bag, element_type, targets=targets, missing=missing)

    # Filter the RID list — all reachable elements if targets=None, only
    # labeled elements if targets is set.
    if targets is None:
        rids = all_rids
    elif missing == "skip":
        rids = [rid for rid in all_rids if rid in target_map]
    else:
        rids = all_rids  # "error" already raised if incomplete; "unknown"
        # keeps all RIDs with None target for absent ones

    # Build the row lookup so sample_loader gets the full row dict.
    row_lookup = _build_row_lookup(bag, element_type)

    # The iteration body — a Python generator function.
    def _generate():
        for rid in rids:
            row = row_lookup.get(rid, {"RID": rid})

            if is_asset:
                path = _resolve_asset_path(bag, element_type, rid, row)
                sample = sample_loader(path, row)
            else:
                sample = sample_loader(None, row) if sample_loader is not None else row

            if transform is not None:
                sample = transform(sample)

            if targets is None:
                yield (sample, rid)
            else:
                target = target_map.get(rid)
                if target_transform is not None:
                    target = target_transform(target)
                yield (sample, target, rid)

    # output_signature handling.
    if output_signature is None:
        # Infer from first sample. Consume one iteration eagerly.
        gen = _generate()
        try:
            first = next(gen)
        except StopIteration:
            raise DerivaMLException(
                f"Cannot build tf.data.Dataset from empty generator "
                f"(element_type={element_type!r}, targets={targets!r}). "
                f"Either the bag has no members of this type, or all were "
                f"filtered by missing='skip'."
            )
        inferred = tf.nest.map_structure(tf.type_spec_from_value, first)

        # Re-wrap so we don't lose the first sample.
        def _generate_with_first():
            yield first
            yield from gen

        return tf.data.Dataset.from_generator(_generate_with_first, output_signature=inferred)
    else:
        return tf.data.Dataset.from_generator(_generate, output_signature=output_signature)


def _bag_element_is_asset(bag: "DatasetBag", element_type: str) -> bool:
    """Return True if element_type is an asset table in the bag."""
    # Try the direct is_asset method first (used in mock bags / tests).
    is_asset_method = getattr(bag, "is_asset", None)
    if is_asset_method is not None and callable(is_asset_method):
        try:
            return bool(is_asset_method(element_type))
        except Exception:
            pass
    # Fall back to model-level check for real DatasetBag instances.
    try:
        table = bag.model.name_to_table(element_type)
        return bool(bag.model.is_asset(table))
    except (AttributeError, KeyError, Exception):
        return False


def _build_row_lookup(bag: "DatasetBag", element_type: str) -> dict:
    """Build {RID: row_dict} lookup for the element table."""
    try:
        return {row["RID"]: row for row in bag.get_table_as_dict(element_type)}
    except Exception:
        return {}


def _resolve_asset_path(bag: "DatasetBag", element_type: str, rid: str, row: dict) -> Path:
    """Compute the on-disk path for an asset's file.

    Uses the canonical BDBag asset layout produced by the bag
    materializer: ``data/asset/{RID}/{element_type}/{Filename}``
    (singular ``asset/``; the RID comes before the element-type
    segment). See ``deriva_ml.dataset.restructure`` for the matching
    layout reference in the restructure path.
    """
    filename = row.get("Filename") or f"{rid}.bin"
    return bag.bag_path / "data" / "asset" / rid / element_type / filename
