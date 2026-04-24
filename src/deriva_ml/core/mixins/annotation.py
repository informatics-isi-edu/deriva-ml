"""Annotation management mixin for DerivaML.

This module provides the AnnotationMixin class which handles
Deriva catalog annotation operations for controlling how data
is displayed in the Chaise web interface.

Annotation Tags:
    - display: tag:isrd.isi.edu,2015:display
    - visible-columns: tag:isrd.isi.edu,2016:visible-columns
    - visible-foreign-keys: tag:isrd.isi.edu,2016:visible-foreign-keys
    - table-display: tag:isrd.isi.edu,2016:table-display
    - column-display: tag:isrd.isi.edu,2016:column-display
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
from typing import TYPE_CHECKING, Any, Callable

_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Column = _ermrest_model.Column
Table = _ermrest_model.Table

from pydantic import ConfigDict, validate_call

from deriva_ml.core.exceptions import DerivaMLException, DerivaMLTableTypeError

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


# Annotation tag URIs
DISPLAY_TAG = "tag:isrd.isi.edu,2015:display"
VISIBLE_COLUMNS_TAG = "tag:isrd.isi.edu,2016:visible-columns"
VISIBLE_FOREIGN_KEYS_TAG = "tag:isrd.isi.edu,2016:visible-foreign-keys"
TABLE_DISPLAY_TAG = "tag:isrd.isi.edu,2016:table-display"
COLUMN_DISPLAY_TAG = "tag:isrd.isi.edu,2016:column-display"
# Bug E.2: opt-in strict mode for pre-allocated RID inserts. When set
# to {"strict": true} on an asset table, deriva-py's uploader raises
# DerivaUploadCatalogCreateError on RID mismatch instead of silently
# adopting the existing row's RID. Use for tables whose rows are
# referenced by FK columns in the same upload batch.
STRICT_PREALLOCATED_RID_TAG = "tag:isrd.isi.edu,2026:strict-preallocated-rid"


class AnnotationMixin:
    """Mixin providing annotation management operations.

    This mixin requires the host class to have:
        - model: DerivaModel instance
        - pathBuilder(): method returning catalog path builder

    Methods:
        get_table_annotations: Get all display-related annotations for a table
        get_column_annotations: Get all display-related annotations for a column
        set_display_annotation: Set display annotation on table or column
        set_visible_columns: Set visible-columns annotation on a table
        set_visible_foreign_keys: Set visible-foreign-keys annotation on a table
        set_table_display: Set table-display annotation on a table
        set_column_display: Set column-display annotation on a column
        add_visible_column: Add a column to visible-columns list
        remove_visible_column: Remove a column from visible-columns list
        reorder_visible_columns: Reorder columns in visible-columns list
        add_visible_foreign_key: Add a foreign key to visible-foreign-keys list
        remove_visible_foreign_key: Remove a foreign key from visible-foreign-keys list
        reorder_visible_foreign_keys: Reorder foreign keys in visible-foreign-keys list
        apply_annotations: Apply staged annotation changes to the catalog
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    pathBuilder: Callable[[], Any]

    # =========================================================================
    # Core Annotation Operations
    # =========================================================================

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_table_annotations(self, table: str | Table) -> dict[str, Any]:
        """Get all Chaise display-related annotations for a table.

        Returns the current values of display, visible-columns,
        visible-foreign-keys, and table-display annotations. Missing
        annotations are represented as ``None`` in the returned dict.

        Args:
            table: Table name (str) or ``Table`` object.

        Returns:
            Dict with keys ``table`` (str), ``schema`` (str),
            ``display`` (dict | None), ``visible_columns`` (dict | None),
            ``visible_foreign_keys`` (dict | None), ``table_display``
            (dict | None).

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.

        Example:
            >>> anns = ml.get_table_annotations("Image")  # doctest: +SKIP
            >>> anns["visible_columns"]  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)
        return {
            "table": table_obj.name,
            "schema": table_obj.schema.name,
            "display": table_obj.annotations.get(DISPLAY_TAG),
            "visible_columns": table_obj.annotations.get(VISIBLE_COLUMNS_TAG),
            "visible_foreign_keys": table_obj.annotations.get(VISIBLE_FOREIGN_KEYS_TAG),
            "table_display": table_obj.annotations.get(TABLE_DISPLAY_TAG),
        }

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_column_annotations(self, table: str | Table, column_name: str) -> dict[str, Any]:
        """Get all Chaise display-related annotations for a column.

        Returns display and column-display annotations. Missing annotations
        are ``None``.

        Args:
            table: Table name (str) or ``Table`` object.
            column_name: Name of the column.

        Returns:
            Dict with keys ``table`` (str), ``schema`` (str),
            ``column`` (str), ``display`` (dict | None),
            ``column_display`` (dict | None).

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.
            DerivaMLException: If ``column_name`` is not a column of ``table``.

        Example:
            >>> anns = ml.get_column_annotations("Image", "Filename")  # doctest: +SKIP
            >>> anns["display"]  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)
        column = table_obj.columns[column_name]
        return {
            "table": table_obj.name,
            "column": column.name,
            "display": column.annotations.get(DISPLAY_TAG),
            "column_display": column.annotations.get(COLUMN_DISPLAY_TAG),
        }

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def set_display_annotation(
        self,
        table: str | Table,
        annotation: dict[str, Any] | None,
        column_name: str | None = None,
    ) -> str:
        """Set the Chaise display annotation on a table or column.

        The display annotation controls how the table or column is labeled in
        the Chaise web UI. The dict shape follows the Chaise display tag
        specification, e.g. ``{"name": "Human Readable Name"}``.
        Changes are staged locally until ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            annotation: Annotation dict, e.g. ``{"name": "My Table"}``.
                Set to ``None`` to remove the annotation.
            column_name: If provided, sets the annotation on that column;
                otherwise sets it on the table.

        Returns:
            Target identifier — table name (str) when setting on the table,
            or ``"Table.column"`` when setting on a column.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.

        Example:
            >>> ml.set_display_annotation("Image", {"name": "Scan Image"})  # doctest: +SKIP
            >>> ml.set_display_annotation("Image", {"name": "File Name"}, column_name="Filename")  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        if column_name:
            column = table_obj.columns[column_name]
            if annotation is None:
                column.annotations.pop(DISPLAY_TAG, None)
            else:
                column.annotations[DISPLAY_TAG] = annotation
            return f"{table_obj.name}.{column_name}"
        else:
            if annotation is None:
                table_obj.annotations.pop(DISPLAY_TAG, None)
            else:
                table_obj.annotations[DISPLAY_TAG] = annotation
            return table_obj.name

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def set_visible_columns(
        self,
        table: str | Table,
        annotation: dict[str, Any] | None,
    ) -> str:
        """Set the visible-columns annotation on a table.

        Controls which columns appear in different UI contexts and their order.
        The annotation is a dict mapping context names (e.g. ``"compact"``,
        ``"detailed"``, ``"entry"``) to lists of column specs. Each spec may
        be a plain column-name string, a foreign-key reference list
        ``[schema, constraint_name]``, or a pseudo-column dict per the Chaise
        visible-columns specification.
        Changes are staged locally until ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            annotation: The visible-columns annotation dict. Set to ``None``
                to remove the annotation.

        Returns:
            Table name.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.

        Example:
            >>> ml.set_visible_columns("Image", {  # doctest: +SKIP
            ...     "compact": ["RID", "Filename", "Subject"],
            ...     "detailed": ["RID", "Filename", "Subject", "Description"]
            ... })
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        if annotation is None:
            table_obj.annotations.pop(VISIBLE_COLUMNS_TAG, None)
        else:
            table_obj.annotations[VISIBLE_COLUMNS_TAG] = annotation

        return table_obj.name

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def set_visible_foreign_keys(
        self,
        table: str | Table,
        annotation: dict[str, Any] | None,
    ) -> str:
        """Set the visible-foreign-keys annotation on a table.

        Controls which related tables (via inbound foreign keys) appear in
        different UI contexts and their order. The annotation is a dict
        mapping context names to lists of FK specs. Each FK spec is a list
        ``[schema, constraint_name]`` referencing an inbound foreign key, or
        a pseudo-column dict per the Chaise visible-foreign-keys specification.
        Changes are staged locally until ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            annotation: The visible-foreign-keys annotation dict. Set to
                ``None`` to remove the annotation.

        Returns:
            Table name.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.

        Example:
            >>> ml.set_visible_foreign_keys("Subject", {  # doctest: +SKIP
            ...     "detailed": [
            ...         ["domain", "Image_Subject_fkey"],
            ...         ["domain", "Diagnosis_Subject_fkey"]
            ...     ]
            ... })
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        if annotation is None:
            table_obj.annotations.pop(VISIBLE_FOREIGN_KEYS_TAG, None)
        else:
            table_obj.annotations[VISIBLE_FOREIGN_KEYS_TAG] = annotation

        return table_obj.name

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def set_table_display(
        self,
        table: str | Table,
        annotation: dict[str, Any] | None,
    ) -> str:
        """Set the table-display annotation on a table.

        Controls table-level display options such as row-naming patterns,
        default page size, and sort order. The annotation dict follows the
        Chaise table-display tag specification, e.g.
        ``{"row_name": {"row_markdown_pattern": "{{{Name}}}"}}``.
        Changes are staged locally until ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            annotation: The table-display annotation dict. Set to ``None``
                to remove the annotation.

        Returns:
            Table name.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.

        Example:
            >>> ml.set_table_display("Subject", {  # doctest: +SKIP
            ...     "row_name": {
            ...         "row_markdown_pattern": "{{{Name}}} ({{{Species}}})"
            ...     }
            ... })
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        if annotation is None:
            table_obj.annotations.pop(TABLE_DISPLAY_TAG, None)
        else:
            table_obj.annotations[TABLE_DISPLAY_TAG] = annotation

        return table_obj.name

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def set_column_display(
        self,
        table: str | Table,
        column_name: str,
        annotation: dict[str, Any] | None,
    ) -> str:
        """Set the column-display annotation on a column.

        Controls how a column's values are rendered, including custom
        formatting and markdown patterns. The annotation dict follows the
        Chaise column-display tag specification, keyed by context name
        (or ``"*"`` for all contexts), e.g.
        ``{"*": {"pre_format": {"format": "%.2f"}}}``.
        Changes are staged locally until ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object containing the column.
            column_name: Name of the column.
            annotation: The column-display annotation dict. Set to ``None``
                to remove the annotation.

        Returns:
            Column identifier as ``"Table.column"`` (str).

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.

        Example:
            >>> ml.set_column_display("Measurement", "Value", {  # doctest: +SKIP
            ...     "*": {"pre_format": {"format": "%.2f"}}
            ... })
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)
        column = table_obj.columns[column_name]

        if annotation is None:
            column.annotations.pop(COLUMN_DISPLAY_TAG, None)
        else:
            column.annotations[COLUMN_DISPLAY_TAG] = annotation

        return f"{table_obj.name}.{column_name}"

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def set_strict_preallocated_rid(
        self,
        table: str | Table,
        strict: bool = True,
    ) -> str:
        """Mark or unmark an asset table as strict-preallocated-RID.

        When ``strict=True``, deriva-py's uploader raises
        ``DerivaUploadCatalogCreateError`` if an upload's caller-supplied
        pre-allocated RID differs from an existing catalog row's RID for
        the same MD5+Filename. When ``False`` (or the annotation is
        absent), the uploader silently adopts the existing row's RID
        (legacy behavior preserved for shared artifacts like
        ``Execution_Metadata`` configs).

        Use strict mode for tables whose rows are referenced by FK
        columns in the same upload batch — any unexpected RID
        reassignment would corrupt those references.

        Args:
            table: Asset table name or Table object.
            strict: If True, set the annotation to ``{"strict": true}``.
                If False, remove the annotation (equivalent to soft mode).

        Returns:
            The table's name.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not an asset table.

        Example:
            >>> ml.set_strict_preallocated_rid("ScanResult", strict=True)  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        if not self.model.is_asset(table):
            raise DerivaMLTableTypeError("asset table", str(table))
        table_obj = self.model.name_to_table(table)
        if strict:
            table_obj.annotations[STRICT_PREALLOCATED_RID_TAG] = {"strict": True}
        else:
            table_obj.annotations.pop(STRICT_PREALLOCATED_RID_TAG, None)
        return table_obj.name

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def is_strict_preallocated_rid(self, table: str | Table) -> bool:
        """Return True if the asset table has the strict-preallocated-RID annotation set.

        Checks for the ``tag:isrd.isi.edu,2026:strict-preallocated-rid``
        annotation. Returns True iff the annotation is present with
        ``{"strict": true}``.

        Args:
            table: Asset table name or Table object.

        Returns:
            True if strict mode is set on this table, False otherwise.
        """
        table_obj = self.model.name_to_table(table)
        anno = table_obj.annotations.get(STRICT_PREALLOCATED_RID_TAG, {})
        if not isinstance(anno, dict):
            return False
        return bool(anno.get("strict", False))

    def apply_annotations(self) -> None:
        """Apply all staged annotation changes to the catalog.

        Pushes any in-memory annotation changes to the live catalog. Must
        be called after any sequence of ``set_*`` or ``add_*/remove_*``
        annotation calls to make changes visible in Chaise.

        Raises:
            DerivaMLException: If the catalog is read-only or the apply
                call fails.

        Example:
            >>> ml.set_display_annotation("Image", {"name": "Scan"})  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        self.model.apply()

    # =========================================================================
    # Visible Columns Convenience Methods
    # =========================================================================

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_visible_column(
        self,
        table: str | Table,
        context: str,
        column: str | list[str] | dict[str, Any],
        position: int | None = None,
    ) -> list[Any]:
        """Add a column to the visible-columns list for a specific context.

        Convenience method for adding columns without replacing the entire
        visible-columns annotation. Changes are staged until
        ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            context: The context to modify (e.g., ``"compact"``,
                ``"detailed"``, ``"entry"``).
            column: Column to add. Can be:
                - str: column name (e.g., ``"Filename"``)
                - list: foreign key reference (e.g., ``["schema", "fkey_name"]``)
                - dict: pseudo-column definition
            position: Position to insert at (0-indexed). If ``None``, appends
                to the end.

        Returns:
            The updated column list for the context.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.
            DerivaMLException: If ``context`` references another context string
                rather than a list.

        Example:
            >>> ml.add_visible_column("Image", "compact", "Description")  # doctest: +SKIP
            >>> ml.add_visible_column("Image", "detailed", ["domain", "Image_Subject_fkey"], 1)  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        # Get or create visible_columns annotation
        visible_cols = table_obj.annotations.get(VISIBLE_COLUMNS_TAG, {})
        if visible_cols is None:
            visible_cols = {}

        # Get or create the context list
        context_list = visible_cols.get(context, [])
        if isinstance(context_list, str):
            raise DerivaMLException(
                f"Context '{context}' references another context '{context_list}'. "
                "Set it explicitly first with set_visible_columns()."
            )

        # Make a copy to avoid modifying in place
        context_list = list(context_list)

        # Insert at position or append
        if position is not None:
            context_list.insert(position, column)
        else:
            context_list.append(column)

        # Update the annotation
        visible_cols[context] = context_list
        table_obj.annotations[VISIBLE_COLUMNS_TAG] = visible_cols

        return context_list

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def remove_visible_column(
        self,
        table: str | Table,
        context: str,
        column: str | list[str] | int,
    ) -> list[Any]:
        """Remove a column from the visible-columns list for a specific context.

        Convenience method for removing columns without replacing the entire
        visible-columns annotation. Changes are staged until
        ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            context: The context to modify (e.g., ``"compact"``,
                ``"detailed"``).
            column: Column to remove. Can be:
                - str: column name to find and remove
                - list: foreign key reference ``[schema, constraint]`` to
                  find and remove
                - int: index position to remove (0-indexed)

        Returns:
            The updated column list for the context.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.
            DerivaMLException: If the annotation or context doesn't exist, or
                the column is not found.

        Example:
            >>> ml.remove_visible_column("Image", "compact", "Description")  # doctest: +SKIP
            >>> ml.remove_visible_column("Image", "compact", 0)  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        # Get visible_columns annotation
        visible_cols = table_obj.annotations.get(VISIBLE_COLUMNS_TAG, {})
        if not visible_cols:
            raise DerivaMLException(f"Table '{table_obj.name}' has no visible-columns annotation.")

        # Get the context list
        context_list = visible_cols.get(context)
        if context_list is None:
            raise DerivaMLException(f"Context '{context}' not found in visible-columns annotation.")
        if isinstance(context_list, str):
            raise DerivaMLException(
                f"Context '{context}' references another context '{context_list}'. "
                "Set it explicitly first with set_visible_columns()."
            )

        # Make a copy
        context_list = list(context_list)
        removed = None

        # Remove by index or by value
        if isinstance(column, int):
            if 0 <= column < len(context_list):
                removed = context_list.pop(column)
            else:
                raise DerivaMLException(
                    f"Index {column} out of range (list has {len(context_list)} items)."
                )
        else:
            # Find and remove the column
            for i, item in enumerate(context_list):
                if item == column:
                    removed = context_list.pop(i)
                    break
                # Also check if it's a pseudo-column with matching source
                if isinstance(item, dict) and isinstance(column, str):
                    if item.get("source") == column:
                        removed = context_list.pop(i)
                        break

            if removed is None:
                raise DerivaMLException(f"Column {column!r} not found in context '{context}'.")

        # Update the annotation
        visible_cols[context] = context_list
        table_obj.annotations[VISIBLE_COLUMNS_TAG] = visible_cols

        return context_list

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def reorder_visible_columns(
        self,
        table: str | Table,
        context: str,
        new_order: list[int] | list[str | list[str] | dict[str, Any]],
    ) -> list[Any]:
        """Reorder columns in the visible-columns list for a specific context.

        Convenience method for reordering columns without manually
        reconstructing the list. Changes are staged until
        ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            context: The context to modify (e.g., ``"compact"``,
                ``"detailed"``).
            new_order: The new order specification. Can be:
                - list of int: ``[2, 0, 1, 3]`` reorders by current positions
                - list of column specs: ``["Name", "RID", ...]`` specifies
                  the exact new order

        Returns:
            The reordered column list.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.
            DerivaMLException: If the annotation or context doesn't exist, or
                the index list is invalid.

        Example:
            >>> ml.reorder_visible_columns("Image", "compact", [2, 0, 1, 3, 4])  # doctest: +SKIP
            >>> ml.reorder_visible_columns("Image", "compact", ["Filename", "Subject", "RID"])  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        # Get visible_columns annotation
        visible_cols = table_obj.annotations.get(VISIBLE_COLUMNS_TAG, {})
        if not visible_cols:
            raise DerivaMLException(f"Table '{table_obj.name}' has no visible-columns annotation.")

        # Get the context list
        context_list = visible_cols.get(context)
        if context_list is None:
            raise DerivaMLException(f"Context '{context}' not found in visible-columns annotation.")
        if isinstance(context_list, str):
            raise DerivaMLException(
                f"Context '{context}' references another context '{context_list}'. "
                "Set it explicitly first with set_visible_columns()."
            )

        original_list = list(context_list)

        # Determine if new_order is indices or column specs
        if new_order and isinstance(new_order[0], int):
            # Reorder by indices
            if len(new_order) != len(original_list):
                raise DerivaMLException(
                    f"Index list length ({len(new_order)}) must match "
                    f"current list length ({len(original_list)})."
                )
            if set(new_order) != set(range(len(original_list))):
                raise DerivaMLException("Index list must contain each index exactly once.")
            new_list = [original_list[i] for i in new_order]
        else:
            # new_order is the exact new column list
            new_list = list(new_order)

        # Update the annotation
        visible_cols[context] = new_list
        table_obj.annotations[VISIBLE_COLUMNS_TAG] = visible_cols

        return new_list

    # =========================================================================
    # Visible Foreign Keys Convenience Methods
    # =========================================================================

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_visible_foreign_key(
        self,
        table: str | Table,
        context: str,
        foreign_key: list[str] | dict[str, Any],
        position: int | None = None,
    ) -> list[Any]:
        """Add a foreign key to the visible-foreign-keys list for a specific context.

        Convenience method for adding related tables without replacing the
        entire visible-foreign-keys annotation. Changes are staged until
        ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            context: The context to modify (e.g., ``"detailed"`` or ``"*"``).
            foreign_key: Foreign key to add. Can be:
                - list: inbound FK reference (e.g.,
                  ``["schema", "Other_Table_fkey"]``)
                - dict: pseudo-column definition for complex relationships
            position: Position to insert at (0-indexed). If ``None``, appends
                to the end.

        Returns:
            The updated foreign key list for the context.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.
            DerivaMLException: If ``context`` references another context string
                rather than a list.

        Example:
            >>> ml.add_visible_foreign_key("Subject", "detailed", ["domain", "Image_Subject_fkey"])  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        # Get or create visible_foreign_keys annotation
        visible_fkeys = table_obj.annotations.get(VISIBLE_FOREIGN_KEYS_TAG, {})
        if visible_fkeys is None:
            visible_fkeys = {}

        # Get or create the context list
        context_list = visible_fkeys.get(context, [])
        if isinstance(context_list, str):
            raise DerivaMLException(
                f"Context '{context}' references another context '{context_list}'. "
                "Set it explicitly first with set_visible_foreign_keys()."
            )

        # Make a copy to avoid modifying in place
        context_list = list(context_list)

        # Insert at position or append
        if position is not None:
            context_list.insert(position, foreign_key)
        else:
            context_list.append(foreign_key)

        # Update the annotation
        visible_fkeys[context] = context_list
        table_obj.annotations[VISIBLE_FOREIGN_KEYS_TAG] = visible_fkeys

        return context_list

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def remove_visible_foreign_key(
        self,
        table: str | Table,
        context: str,
        foreign_key: list[str] | int,
    ) -> list[Any]:
        """Remove a foreign key from the visible-foreign-keys list for a specific context.

        Convenience method for removing related tables without replacing the
        entire visible-foreign-keys annotation. Changes are staged until
        ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            context: The context to modify (e.g., ``"detailed"``, ``"*"``).
            foreign_key: Foreign key to remove. Can be:
                - list: FK reference ``[schema, constraint]`` to find and remove
                - int: index position to remove (0-indexed)

        Returns:
            The updated foreign key list for the context.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.
            DerivaMLException: If the annotation or context doesn't exist, or
                the foreign key is not found.

        Example:
            >>> ml.remove_visible_foreign_key("Subject", "detailed", ["domain", "Image_Subject_fkey"])  # doctest: +SKIP
            >>> ml.remove_visible_foreign_key("Subject", "detailed", 0)  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        # Get visible_foreign_keys annotation
        visible_fkeys = table_obj.annotations.get(VISIBLE_FOREIGN_KEYS_TAG, {})
        if not visible_fkeys:
            raise DerivaMLException(
                f"Table '{table_obj.name}' has no visible-foreign-keys annotation."
            )

        # Get the context list
        context_list = visible_fkeys.get(context)
        if context_list is None:
            raise DerivaMLException(
                f"Context '{context}' not found in visible-foreign-keys annotation."
            )
        if isinstance(context_list, str):
            raise DerivaMLException(
                f"Context '{context}' references another context '{context_list}'. "
                "Set it explicitly first with set_visible_foreign_keys()."
            )

        # Make a copy
        context_list = list(context_list)
        removed = None

        # Remove by index or by value
        if isinstance(foreign_key, int):
            if 0 <= foreign_key < len(context_list):
                removed = context_list.pop(foreign_key)
            else:
                raise DerivaMLException(
                    f"Index {foreign_key} out of range (list has {len(context_list)} items)."
                )
        else:
            # Find and remove the foreign key
            for i, item in enumerate(context_list):
                if item == foreign_key:
                    removed = context_list.pop(i)
                    break

            if removed is None:
                raise DerivaMLException(
                    f"Foreign key {foreign_key!r} not found in context '{context}'."
                )

        # Update the annotation
        visible_fkeys[context] = context_list
        table_obj.annotations[VISIBLE_FOREIGN_KEYS_TAG] = visible_fkeys

        return context_list

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def reorder_visible_foreign_keys(
        self,
        table: str | Table,
        context: str,
        new_order: list[int] | list[list[str] | dict[str, Any]],
    ) -> list[Any]:
        """Reorder foreign keys in the visible-foreign-keys list for a specific context.

        Convenience method for reordering related tables without manually
        reconstructing the list. Changes are staged until
        ``apply_annotations()`` is called.

        Args:
            table: Table name (str) or ``Table`` object.
            context: The context to modify (e.g., ``"detailed"``, ``"*"``).
            new_order: The new order specification. Can be:
                - list of int: ``[2, 0, 1]`` reorders by current positions
                - list of FK refs: ``[["schema", "fkey1"], ...]`` specifies
                  the exact new order

        Returns:
            The reordered foreign key list.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.
            DerivaMLException: If the annotation or context doesn't exist, or
                the index list is invalid.

        Example:
            >>> ml.reorder_visible_foreign_keys("Subject", "detailed", [2, 0, 1])  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)

        # Get visible_foreign_keys annotation
        visible_fkeys = table_obj.annotations.get(VISIBLE_FOREIGN_KEYS_TAG, {})
        if not visible_fkeys:
            raise DerivaMLException(
                f"Table '{table_obj.name}' has no visible-foreign-keys annotation."
            )

        # Get the context list
        context_list = visible_fkeys.get(context)
        if context_list is None:
            raise DerivaMLException(
                f"Context '{context}' not found in visible-foreign-keys annotation."
            )
        if isinstance(context_list, str):
            raise DerivaMLException(
                f"Context '{context}' references another context '{context_list}'. "
                "Set it explicitly first with set_visible_foreign_keys()."
            )

        original_list = list(context_list)

        # Determine if new_order is indices or foreign key specs
        if new_order and isinstance(new_order[0], int):
            # Reorder by indices
            if len(new_order) != len(original_list):
                raise DerivaMLException(
                    f"Index list length ({len(new_order)}) must match "
                    f"current list length ({len(original_list)})."
                )
            if set(new_order) != set(range(len(original_list))):
                raise DerivaMLException("Index list must contain each index exactly once.")
            new_list = [original_list[i] for i in new_order]
        else:
            # new_order is the exact new foreign key list
            new_list = list(new_order)

        # Update the annotation
        visible_fkeys[context] = new_list
        table_obj.annotations[VISIBLE_FOREIGN_KEYS_TAG] = visible_fkeys

        return new_list

    # =========================================================================
    # Template Helpers
    # =========================================================================

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_handlebars_template_variables(self, table: str | Table) -> dict[str, Any]:
        """Get all available template variables for a table.

        Returns the columns, foreign keys, and special variables that can be
        used in Handlebars templates (row_markdown_pattern, markdown_pattern, etc.)
        for the specified table.

        Args:
            table: Table name or Table object.

        Returns:
            Dictionary with columns, foreign_keys, special_variables, and helper_examples.

        Example:
            >>> vars = ml.get_handlebars_template_variables("Image")  # doctest: +SKIP
            >>> for col in vars["columns"]:  # doctest: +SKIP
            ...     print(f"{col['name']}: {col['template']}")
        """
        table_obj = self.model.name_to_table(table)

        # Get columns
        columns = []
        for col in table_obj.columns:
            columns.append({
                "name": col.name,
                "type": str(col.type.typename),
                "template": "{{{" + col.name + "}}}",
                "row_template": "{{{_row." + col.name + "}}}",
            })

        # Get foreign keys (outbound)
        foreign_keys = []
        for fkey in table_obj.foreign_keys:
            schema_name = fkey.constraint_schema.name
            constraint_name = fkey.constraint_name
            fk_path = f"$fkeys.{schema_name}.{constraint_name}"

            # Get columns from referenced table
            ref_columns = [col.name for col in fkey.pk_table.columns]

            foreign_keys.append({
                "constraint": [schema_name, constraint_name],
                "from_columns": [col.name for col in fkey.columns],
                "to_table": fkey.pk_table.name,
                "to_columns": ref_columns,
                "values_template": "{{{" + fk_path + ".values.COLUMN}}}",
                "row_name_template": "{{{" + fk_path + ".rowName}}}",
                "example_column_templates": [
                    "{{{" + fk_path + ".values." + c + "}}}"
                    for c in ref_columns[:3]  # Show first 3 as examples
                ]
            })

        return {
            "table": table_obj.name,
            "columns": columns,
            "foreign_keys": foreign_keys,
            "special_variables": {
                "_value": {
                    "description": "Current column value (in column_display)",
                    "template": "{{{_value}}}"
                },
                "_row": {
                    "description": "Object with all row columns",
                    "template": "{{{_row.column_name}}}"
                },
                "$catalog.id": {
                    "description": "Catalog ID",
                    "template": "{{{$catalog.id}}}"
                },
                "$catalog.snapshot": {
                    "description": "Current snapshot ID",
                    "template": "{{{$catalog.snapshot}}}"
                },
            },
            "helper_examples": {
                "conditional": "{{#if column}}...{{else}}...{{/if}}",
                "iteration": "{{#each array}}{{{this}}}{{/each}}",
                "comparison": "{{#ifCond val1 '==' val2}}...{{/ifCond}}",
                "date_format": "{{formatDate RCT 'YYYY-MM-DD'}}",
                "json_output": "{{toJSON object}}"
            }
        }
