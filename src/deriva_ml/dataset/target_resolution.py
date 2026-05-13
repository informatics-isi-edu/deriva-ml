"""Shared target-resolution logic for DatasetBag adapters and restructure.

Both ``DatasetBag.as_torch_dataset``, ``DatasetBag.as_tf_dataset``, and
``DatasetBag.restructure_assets`` need the same logic: given a bag, an
element-table name, and a target specification, walk ``bag.feature_values``
for each target, apply the requested missing-value policy, and return a
dict keyed by element RID mapping to the target-arity shape.

Keeping this logic in one module makes the "same semantics" claim across
the three public methods enforceable by shared code rather than parallel
implementation. See design spec §8.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

from deriva_ml.core.exceptions import DerivaMLException

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset_bag import DatasetBag
    from deriva_ml.feature import FeatureRecord

    FeatureSelector = Callable[[list["FeatureRecord"]], "FeatureRecord | None"]


# Maximum unlabeled-RID count to show in missing="error" message before
# truncating. Keeps error messages readable on very sparse datasets.
_MISSING_ERROR_RID_LIST_LIMIT = 20


def _validate_feature_vocabulary(
    bag: "DatasetBag",
    element_type: str,
    feature_names: list[str],
) -> None:
    """Validate that each named feature on ``element_type`` is vocabulary-based.

    A feature is vocabulary-based when it has at least one controlled-vocabulary
    term column. ``restructure_assets`` and other class-folder style export
    flows require this so that the resolved value is a stable directory name.

    Looks up each feature via ``bag.lookup_feature(element_type, name)`` and
    raises ``DerivaMLException`` if any feature exists but has no term columns.
    Missing features (where ``lookup_feature`` raises) are silently ignored
    here — ``_resolve_targets``'s feature_values walk handles that case via the
    ``missing`` policy.

    Args:
        bag: Source ``DatasetBag``.
        element_type: Table name to look features up on.
        feature_names: Feature names to validate.

    Raises:
        DerivaMLException: If a feature exists on ``element_type`` and has no
            controlled-vocabulary term columns.
    """
    for feature_name in feature_names:
        try:
            feat = bag.lookup_feature(element_type, feature_name)
        except Exception:
            # Not a feature on this table; nothing to validate here.
            continue
        if not list(feat.term_columns):
            raise DerivaMLException(
                f"Feature {feature_name!r} on table {element_type!r} has no "
                f"controlled vocabulary term columns. Only vocabulary-based "
                f"features can be used for class-folder grouping when "
                f"enforce_vocabulary=True. Set enforce_vocabulary=False to "
                f"allow non-vocabulary features."
            )


def _resolve_targets(
    bag: "DatasetBag",
    element_type: str,
    *,
    targets: "list[str] | dict[str, FeatureSelector] | None",
    missing: Literal["error", "skip", "unknown"],
    enforce_vocabulary: bool = False,
) -> "dict[str, Any]":
    """Resolve feature values into per-element target records.

    Walks ``bag.feature_values(element_type, feature_name, selector=...)``
    for each feature in ``targets``, groups records by their target-
    element RID, and applies the ``missing`` policy to elements whose
    feature value is absent.

    Args:
        bag: Source ``DatasetBag``.
        element_type: Domain table name whose rows are the elements
            (e.g., ``"Image"``).
        targets: Target specification per the aligned vocabulary (spec
            §3.7). ``None`` yields no labels (empty result). ``list[str]``
            yields one ``FeatureRecord`` per element for single-target,
            ``dict[feature_name, FeatureRecord]`` for multi-target.
            ``dict[str, FeatureSelector]`` passes per-feature selectors
            through to ``bag.feature_values``.
        missing: Policy for elements with no feature value.
        enforce_vocabulary: If True, every named feature on
            ``element_type`` must have at least one controlled-vocabulary
            term column. Required by class-folder style exporters
            (``restructure_assets``) where the resolved value must be a
            stable directory name. Defaults to False for adapter callers
            (``as_torch_dataset``, ``as_tf_dataset``) that consume the
            FeatureRecord directly.

    Returns:
        Dict keyed by element RID. Value shape:

        - `targets=None`: empty dict.
        - `targets=["A"]`: `FeatureRecord`.
        - `targets=["A", "B"]` or dict with 2+ keys: `dict[str, FeatureRecord]`.
        - Element absent under `missing="skip"`: key not in returned dict.
        - Element absent under `missing="unknown"`: value is `None`.

    Raises:
        DerivaMLException: If ``missing="error"`` and any element lacks a
            feature value (message lists up to 20 unlabeled RIDs), or if
            ``enforce_vocabulary=True`` and a named feature has no term
            columns.
    """
    if not targets:
        return {}

    # Normalize targets to (feature_name, selector) pairs. A list form has
    # selector=None for each feature; a dict form uses the mapped selector.
    if isinstance(targets, list):
        feature_specs: list[tuple[str, Any]] = [(name, None) for name in targets]
    else:
        feature_specs = list(targets.items())

    if enforce_vocabulary:
        _validate_feature_vocabulary(bag, element_type, [name for name, _ in feature_specs])

    # Walk features and collect records keyed by target-element RID.
    # The target column is the element_type's name (e.g., "Image" on an
    # Image-target FeatureRecord). FeatureRecord instances carry that
    # attribute; we pull it dynamically.
    per_feature_per_rid: dict[str, dict[str, "FeatureRecord"]] = {name: {} for name, _ in feature_specs}
    for feature_name, selector in feature_specs:
        for record in bag.feature_values(element_type, feature_name, selector=selector):
            rid = getattr(record, element_type)
            per_feature_per_rid[feature_name][rid] = record

    # Determine the universe of element RIDs. Start from the bag's full
    # membership list so we detect elements with no feature values at all
    # (they would otherwise be silently absent from the feature iterator).
    # Then union in any RIDs seen from feature_values in case list_dataset_members
    # is unavailable or returns a subset.
    members_by_type = bag.list_dataset_members(recurse=True)
    all_rids: set[str] = {m["RID"] for m in members_by_type.get(element_type, [])}
    for rid_map in per_feature_per_rid.values():
        all_rids.update(rid_map.keys())

    # Apply missing policy and build the result dict.
    unlabeled: list[str] = []
    result: dict[str, Any] = {}
    is_single_target = len(feature_specs) == 1
    single_feature_name = feature_specs[0][0] if is_single_target else None

    for rid in sorted(all_rids):
        feature_records = {name: per_feature_per_rid[name].get(rid) for name, _ in feature_specs}
        any_missing = any(v is None for v in feature_records.values())

        if any_missing:
            if missing == "error":
                unlabeled.append(rid)
                continue
            if missing == "skip":
                continue
            # missing == "unknown"
            result[rid] = None
            continue

        if is_single_target:
            result[rid] = feature_records[single_feature_name]
        else:
            result[rid] = feature_records

    if missing == "error" and unlabeled:
        preview = unlabeled[:_MISSING_ERROR_RID_LIST_LIMIT]
        suffix = f" (and {len(unlabeled) - len(preview)} more)" if len(unlabeled) > len(preview) else ""
        raise DerivaMLException(
            f"{len(unlabeled)} element(s) of type {element_type!r} have no "
            f"value for one or more targets in {targets!r}. "
            f"Unlabeled RIDs: {preview}{suffix}. "
            f"Pass missing='skip' to drop unlabeled elements, or "
            f"missing='unknown' to keep them with target=None."
        )

    return result
