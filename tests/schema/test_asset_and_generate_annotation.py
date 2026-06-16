"""Unit tests for ``asset_annotation`` and ``generate_annotation``.

Closes audit P1 F-13 (no unit test for ``asset_annotation``) and
F-14 (no unit test for ``generate_annotation``). Pure-Python
tests against ``MagicMock``-shaped ``Table`` / ``Model`` stand-ins;
no live catalog required.

These tests also serve as regression coverage for P1 F-01 (the
``"deriva-ml"`` literals that were hardcoded inside
``dataset_annotation``) and P1 F-08 (the set-iteration
non-determinism in ``asset_annotation``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva.core import tag as deriva_tags

from deriva_ml.schema.annotations import (
    asset_annotation,
    generate_annotation,
)


# ---------------------------------------------------------------------------
# asset_annotation — Table stand-in
# ---------------------------------------------------------------------------


def _make_asset_table(
    *,
    schema_name: str = "deriva-ml",
    table_name: str = "Execution_Asset",
    metadata_columns: tuple[str, ...] = (),
    fkey_columns: tuple[str, ...] = (),
) -> MagicMock:
    """Build a minimal ``Table``-shaped mock for asset_annotation tests.

    The function under test reads:
      - ``asset_table.schema.name`` — schema namespace for FK refs.
      - ``asset_table.name`` — used in FK constraint names.
      - ``asset_table.columns`` — iterable of objects with ``.name``;
        also subscriptable by name. Items NOT in ``DerivaAssetColumns``
        (``RID``, ``RCT``, ``RMT``, ``RCB``, ``RMB``, ``URL``,
        ``Filename``, ``Length``, ``MD5``, ``Description``) become
        the ``asset_metadata`` set the function iterates to build
        the ``visible_columns`` tail.
      - ``asset_table.foreign_keys`` — used by the ``fkey_column``
        helper. Tests that don't exercise FK detection pass an
        empty iterable.
      - ``asset_table.annotations`` — the dict the function
        mutates in place. We seed an empty dict and then read it
        back after the call.
      - ``asset_table.columns["URL"].annotations`` — read-modified
        with the file-preview annotation.
      - ``asset_table.schema.model.apply()`` — called once at the
        end; we leave it as a no-op MagicMock and assert it was
        invoked.
    """
    table = MagicMock()
    table.schema.name = schema_name
    table.name = table_name

    # ``columns`` needs to be iterable (yielding objects with .name)
    # AND subscriptable by name (returning column objects with their
    # own .annotations dict for the URL-column case). MagicMock's
    # default subscript-as-call doesn't match what the function
    # expects, so we use a real dict + an iterable-of-values trick.
    standard = (
        "RID",
        "RCT",
        "RMT",
        "RCB",
        "RMB",
        "URL",
        "Filename",
        "Length",
        "MD5",
        "Description",
    )
    all_col_names = list(standard) + list(metadata_columns)
    col_objs = {}
    for name in all_col_names:
        c = MagicMock()
        c.name = name
        c.annotations = {}
        col_objs[name] = c

    # Make ``columns`` both iterable (over the column objects, so
    # ``{c.name for c in asset_table.columns}`` works) and
    # subscriptable (so ``asset_table.columns["URL"]`` returns the
    # URL column object). A dict subclass that yields values on
    # iteration covers both.
    class _Columns(dict):
        def __iter__(self):
            return iter(self.values())

    table.columns = _Columns(col_objs)

    # ``foreign_keys`` — empty for the simple shape; tests that
    # want to exercise the fkey_column helper override this.
    table.foreign_keys = list(fkey_columns)

    # ``annotations`` starts empty; the function .update()s it.
    table.annotations = {}

    # ``schema.model.apply()`` is a no-op MagicMock we can assert
    # was called.
    return table


class TestAssetAnnotationShape:
    """Pin the top-level shape of the annotation dict produced by
    ``asset_annotation``.

    The function operates by side effect on the table; we inspect
    the after-state of ``asset_table.annotations``.
    """

    def test_writes_table_display_and_visible_columns(self):
        table = _make_asset_table()
        asset_annotation(table)

        # Both top-level Chaise tags are present.
        assert deriva_tags.table_display in table.annotations
        assert deriva_tags.visible_columns in table.annotations

    def test_table_display_row_name_uses_filename(self):
        """The row-display markdown pattern interpolates ``{{{Filename}}}``."""
        table = _make_asset_table()
        asset_annotation(table)
        td = table.annotations[deriva_tags.table_display]
        assert td == {"row_name": {"row_markdown_pattern": "{{{Filename}}}"}}

    def test_visible_columns_has_star_detailed_filter(self):
        """``visible_columns`` has all three Chaise context keys."""
        table = _make_asset_table()
        asset_annotation(table)
        vc = table.annotations[deriva_tags.visible_columns]
        assert set(vc.keys()) == {"*", "detailed", "filter"}

    def test_asset_type_source_inbound_outbound_present(self):
        """The Asset_Type inbound/outbound FK chain appears in `*`, `detailed`, `filter`."""
        table = _make_asset_table(table_name="Execution_Asset")
        asset_annotation(table)
        vc = table.annotations[deriva_tags.visible_columns]

        # Each context lists the Asset_Type association FK chain.
        expected_inbound_fkey = "Execution_Asset_Asset_Type_Execution_Asset_fkey"
        expected_outbound_fkey = "Execution_Asset_Asset_Type_Asset_Type_fkey"

        def _carries_asset_type(entries: list) -> bool:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                src = entry.get("source", [])
                if not (isinstance(src, list) and len(src) >= 2):
                    continue
                inbound = src[0].get("inbound", []) if isinstance(src[0], dict) else []
                outbound = src[1].get("outbound", []) if isinstance(src[1], dict) else []
                if expected_inbound_fkey in inbound and expected_outbound_fkey in outbound:
                    return True
            return False

        assert _carries_asset_type(vc["*"])
        assert _carries_asset_type(vc["detailed"])
        assert _carries_asset_type(vc["filter"]["and"])

    def test_url_column_gets_file_preview_annotation(self):
        """The ``URL`` column's ``asset`` annotation gets a file-preview display.

        Pins the application/octet-stream → text mapping that lets
        Chaise preview uv.lock / .toml / similar plain-text assets
        uploaded under the generic octet-stream content type.
        """
        table = _make_asset_table()
        asset_annotation(table)
        url_ann = table.columns["URL"].annotations[deriva_tags.asset]
        assert "display" in url_ann
        # Star-context display carries the file_preview mapping.
        star_display = url_ann["display"].get("*", {})
        file_preview = star_display.get("file_preview", {})
        mapping = file_preview.get("content_type_mapping", {})
        assert mapping.get("application/octet-stream") == "text"

    def test_apply_is_called_once(self):
        """``schema.model.apply()`` is called once at the end."""
        table = _make_asset_table()
        asset_annotation(table)
        table.schema.model.apply.assert_called_once()


class TestAssetAnnotationDeterministicOrdering:
    """Pin the F-08 fix: visible-columns order is deterministic
    across Python runs even though ``asset_metadata`` is a set.

    Pre-fix, ``[fkey_column(c) for c in asset_metadata]`` iterated
    the set's implementation-defined order; rebuilds produced
    jittering annotation output.
    """

    def test_visible_columns_star_metadata_tail_is_alphabetically_sorted(self):
        """The metadata-column tail of ``visible_columns["*"]`` is sorted by name."""
        # Pass metadata columns in deliberately-unsorted order; the
        # function should output them sorted.
        table = _make_asset_table(metadata_columns=("Zebra", "Alpha", "Mango", "Bravo"))
        asset_annotation(table)
        vc = table.annotations[deriva_tags.visible_columns]

        # Standard columns end with ``asset_type_source`` (a dict);
        # the metadata tail is the trailing string entries.
        star = vc["*"]
        trailing_strings = [e for e in star if isinstance(e, str) and e in {"Zebra", "Alpha", "Mango", "Bravo"}]
        assert trailing_strings == ["Alpha", "Bravo", "Mango", "Zebra"], (
            f"Expected metadata tail sorted alphabetically; got {trailing_strings}. F-08 regression."
        )

    def test_visible_columns_detailed_metadata_tail_is_alphabetically_sorted(self):
        """The metadata-column tail of ``visible_columns["detailed"]`` is sorted."""
        table = _make_asset_table(metadata_columns=("Zebra", "Alpha", "Mango", "Bravo"))
        asset_annotation(table)
        vc = table.annotations[deriva_tags.visible_columns]
        detailed = vc["detailed"]
        trailing = [e for e in detailed if isinstance(e, str) and e in {"Zebra", "Alpha", "Mango", "Bravo"}]
        assert trailing == ["Alpha", "Bravo", "Mango", "Zebra"]


# ---------------------------------------------------------------------------
# generate_annotation — Model stand-in
# ---------------------------------------------------------------------------


class TestGenerateAnnotationShape:
    """Pin the top-level shape of ``generate_annotation``'s return.

    The function takes a ``model`` (currently unused) and a
    ``schema`` name. We pass a stub ``MagicMock()`` for the model
    and a string for the schema.
    """

    def test_returns_dict_with_five_canonical_keys(self):
        """Five-key shape: workflow / dataset / execution / schema / dataset_version."""
        result = generate_annotation(MagicMock(), "deriva-ml")
        assert set(result.keys()) == {
            "workflow_annotation",
            "dataset_annotation",
            "execution_annotation",
            "schema_annotation",
            "dataset_version_annotation",
        }


class TestGenerateAnnotationSchemaParameterIsThreaded:
    """Pin F-01: every FK reference in ``dataset_annotation`` uses
    the ``schema`` parameter, NOT the hardcoded literal ``"deriva-ml"``.

    Pre-fix, six places inside ``dataset_annotation`` carried the
    literal string. A user calling
    ``create_ml_schema(schema_name="my_ml")`` got annotations
    referencing FKs in a schema that didn't exist.

    These tests pass a non-default schema name and assert no
    annotation value contains ``"deriva-ml"``.
    """

    def _flatten_to_strings(self, value) -> list[str]:
        """Recursively collect every string in a JSON-shaped value."""
        out: list[str] = []
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                out.extend(self._flatten_to_strings(v))
        elif isinstance(value, list):
            for item in value:
                out.extend(self._flatten_to_strings(item))
        return out

    def test_dataset_annotation_contains_no_deriva_ml_literal_under_custom_schema(self):
        """Under ``schema="my_custom"``, no string in ``dataset_annotation`` is ``"deriva-ml"``."""
        result = generate_annotation(MagicMock(), "my_custom")
        all_strings = self._flatten_to_strings(result["dataset_annotation"])
        leaked = [s for s in all_strings if s == "deriva-ml"]
        assert not leaked, (
            f"dataset_annotation leaked the literal 'deriva-ml' under "
            f"schema='my_custom'. F-01 regression — every FK reference "
            f"in this bundle must use the schema parameter, not the "
            f"literal. Pre-fix, six places carried the literal."
        )

    def test_dataset_annotation_references_custom_schema(self):
        """Under ``schema="my_custom"``, the bundle contains FK refs to ``"my_custom"``."""
        result = generate_annotation(MagicMock(), "my_custom")
        all_strings = self._flatten_to_strings(result["dataset_annotation"])
        assert "my_custom" in all_strings, (
            "dataset_annotation must reference the schema parameter; "
            "under schema='my_custom' at least one FK ref should "
            "namespace into 'my_custom'."
        )

    def test_workflow_annotation_contains_no_deriva_ml_literal_under_custom_schema(self):
        """The sibling bundles must also stay literal-free.

        These were already correct pre-fix (the bug was localized
        to ``dataset_annotation``), but the test makes the contract
        explicit so a future regression on the workflow / execution
        side fails this test, not silently.
        """
        result = generate_annotation(MagicMock(), "my_custom")
        for key in (
            "workflow_annotation",
            "execution_annotation",
            "dataset_version_annotation",
        ):
            leaked = [s for s in self._flatten_to_strings(result[key]) if s == "deriva-ml"]
            assert not leaked, f"{key} leaked the literal 'deriva-ml' under schema='my_custom'."
