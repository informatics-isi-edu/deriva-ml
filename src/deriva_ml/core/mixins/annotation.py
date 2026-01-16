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

from typing import TYPE_CHECKING, Any, Callable

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Column = _ermrest_model.Column
Table = _ermrest_model.Table

from pydantic import ConfigDict, validate_call

from deriva_ml.core.exceptions import DerivaMLException

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


# Annotation tag URIs
DISPLAY_TAG = "tag:isrd.isi.edu,2015:display"
VISIBLE_COLUMNS_TAG = "tag:isrd.isi.edu,2016:visible-columns"
VISIBLE_FOREIGN_KEYS_TAG = "tag:isrd.isi.edu,2016:visible-foreign-keys"
TABLE_DISPLAY_TAG = "tag:isrd.isi.edu,2016:table-display"
COLUMN_DISPLAY_TAG = "tag:isrd.isi.edu,2016:column-display"


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
        list_foreign_keys: List all foreign keys related to a table
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
        """Get all display-related annotations for a table.

        Returns the current values of display, visible-columns, visible-foreign-keys,
        and table-display annotations for the specified table.

        Args:
            table: Table name or Table object.

        Returns:
            Dictionary with keys: table, schema, display, visible_columns,
            visible_foreign_keys, table_display. Missing annotations are None.

        Example:
            >>> annotations = ml.get_table_annotations("Image")
            >>> print(annotations["visible_columns"])
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
        """Get all display-related annotations for a column.

        Returns the current values of display and column-display annotations
        for the specified column.

        Args:
            table: Table name or Table object containing the column.
            column_name: Name of the column.

        Returns:
            Dictionary with keys: table, column, display, column_display.
            Missing annotations are None.

        Example:
            >>> annotations = ml.get_column_annotations("Image", "Filename")
            >>> print(annotations["display"])
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
        """Set the display annotation on a table or column.

        The display annotation controls basic naming and display options.
        Changes are staged locally until apply_annotations() is called.

        Args:
            table: Table name or Table object.
            annotation: The display annotation value. Set to None to remove.
            column_name: If provided, sets annotation on the column; otherwise on the table.

        Returns:
            Target identifier (table name or table.column).

        Example:
            >>> ml.set_display_annotation("Image", {"name": "Images"})
            >>> ml.set_display_annotation("Image", {"name": "File Name"}, column_name="Filename")
            >>> ml.apply_annotations()  # Commit changes
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
        Changes are staged locally until apply_annotations() is called.

        Args:
            table: Table name or Table object.
            annotation: The visible-columns annotation value. Set to None to remove.

        Returns:
            Table name.

        Example:
            >>> ml.set_visible_columns("Image", {
            ...     "compact": ["RID", "Filename", "Subject"],
            ...     "detailed": ["RID", "Filename", "Subject", "Description"]
            ... })
            >>> ml.apply_annotations()
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
        different UI contexts and their order.
        Changes are staged locally until apply_annotations() is called.

        Args:
            table: Table name or Table object.
            annotation: The visible-foreign-keys annotation value. Set to None to remove.

        Returns:
            Table name.

        Example:
            >>> ml.set_visible_foreign_keys("Subject", {
            ...     "detailed": [
            ...         ["domain", "Image_Subject_fkey"],
            ...         ["domain", "Diagnosis_Subject_fkey"]
            ...     ]
            ... })
            >>> ml.apply_annotations()
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

        Controls table-level display options like row naming patterns,
        page size, and row ordering.
        Changes are staged locally until apply_annotations() is called.

        Args:
            table: Table name or Table object.
            annotation: The table-display annotation value. Set to None to remove.

        Returns:
            Table name.

        Example:
            >>> ml.set_table_display("Subject", {
            ...     "row_name": {
            ...         "row_markdown_pattern": "{{{Name}}} ({{{Species}}})"
            ...     }
            ... })
            >>> ml.apply_annotations()
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
        formatting and markdown patterns.
        Changes are staged locally until apply_annotations() is called.

        Args:
            table: Table name or Table object containing the column.
            column_name: Name of the column.
            annotation: The column-display annotation value. Set to None to remove.

        Returns:
            Column identifier (table.column).

        Example:
            >>> ml.set_column_display("Measurement", "Value", {
            ...     "*": {"pre_format": {"format": "%.2f"}}
            ... })
            >>> ml.apply_annotations()
        """
        table_obj = self.model.name_to_table(table)
        column = table_obj.columns[column_name]

        if annotation is None:
            column.annotations.pop(COLUMN_DISPLAY_TAG, None)
        else:
            column.annotations[COLUMN_DISPLAY_TAG] = annotation

        return f"{table_obj.name}.{column_name}"

    def apply_annotations(self) -> None:
        """Apply all staged annotation changes to the catalog.

        Commits any annotation changes made via set_display_annotation,
        set_visible_columns, set_visible_foreign_keys, set_table_display,
        or set_column_display to the remote catalog.

        Example:
            >>> ml.set_display_annotation("Image", {"name": "Images"})
            >>> ml.set_visible_columns("Image", {"compact": ["RID", "Filename"]})
            >>> ml.apply_annotations()  # Commit all changes
        """
        self.model.apply()

    # =========================================================================
    # Foreign Key Information
    # =========================================================================

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_foreign_keys(self, table: str | Table) -> dict[str, Any]:
        """List all foreign keys related to a table.

        Returns both outbound foreign keys (from this table to others) and
        inbound foreign keys (from other tables to this one). Useful for
        determining valid constraint names for visible-columns and
        visible-foreign-keys annotations.

        Args:
            table: Table name or Table object.

        Returns:
            Dictionary with:
            - table: Table name
            - outbound: List of outbound foreign keys
            - inbound: List of inbound foreign keys
            Each foreign key contains constraint_name, from_table, from_columns,
            to_table, to_columns.

        Example:
            >>> fkeys = ml.list_foreign_keys("Image")
            >>> for fk in fkeys["outbound"]:
            ...     print(f"{fk['constraint_name']} -> {fk['to_table']}")
        """
        table_obj = self.model.name_to_table(table)

        outbound = []
        for fkey in table_obj.foreign_keys:
            outbound.append({
                "constraint_name": [fkey.constraint_schema.name, fkey.constraint_name],
                "from_table": table_obj.name,
                "from_columns": [col.name for col in fkey.columns],
                "to_table": fkey.pk_table.name,
                "to_columns": [col.name for col in fkey.referenced_columns],
            })

        inbound = []
        for fkey in table_obj.referenced_by:
            inbound.append({
                "constraint_name": [fkey.constraint_schema.name, fkey.constraint_name],
                "from_table": fkey.table.name,
                "from_columns": [col.name for col in fkey.columns],
                "to_table": table_obj.name,
                "to_columns": [col.name for col in fkey.referenced_columns],
            })

        return {
            "table": table_obj.name,
            "outbound": outbound,
            "inbound": inbound,
        }

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
        visible-columns annotation. Changes are staged until apply_annotations()
        is called.

        Args:
            table: Table name or Table object.
            context: The context to modify (e.g., "compact", "detailed", "entry").
            column: Column to add. Can be:
                - String: column name (e.g., "Filename")
                - List: foreign key reference (e.g., ["schema", "fkey_name"])
                - Dict: pseudo-column definition
            position: Position to insert at (0-indexed). If None, appends to end.

        Returns:
            The updated column list for the context.

        Raises:
            DerivaMLException: If context references another context.

        Example:
            >>> ml.add_visible_column("Image", "compact", "Description")
            >>> ml.add_visible_column("Image", "detailed", ["domain", "Image_Subject_fkey"], 1)
            >>> ml.apply_annotations()
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
        visible-columns annotation. Changes are staged until apply_annotations()
        is called.

        Args:
            table: Table name or Table object.
            context: The context to modify (e.g., "compact", "detailed").
            column: Column to remove. Can be:
                - String: column name to find and remove
                - List: foreign key reference [schema, constraint] to find and remove
                - Integer: index position to remove (0-indexed)

        Returns:
            The updated column list for the context.

        Raises:
            DerivaMLException: If annotation or context doesn't exist, or column not found.

        Example:
            >>> ml.remove_visible_column("Image", "compact", "Description")
            >>> ml.remove_visible_column("Image", "compact", 0)  # Remove first column
            >>> ml.apply_annotations()
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

        Convenience method for reordering columns without manually reconstructing
        the list. Changes are staged until apply_annotations() is called.

        Args:
            table: Table name or Table object.
            context: The context to modify (e.g., "compact", "detailed").
            new_order: The new order specification. Can be:
                - List of indices: [2, 0, 1, 3] reorders by current positions
                - List of column specs: ["Name", "RID", ...] specifies exact order

        Returns:
            The reordered column list.

        Raises:
            DerivaMLException: If annotation or context doesn't exist, or invalid order.

        Example:
            >>> ml.reorder_visible_columns("Image", "compact", [2, 0, 1, 3, 4])
            >>> ml.reorder_visible_columns("Image", "compact", ["Filename", "Subject", "RID"])
            >>> ml.apply_annotations()
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

        Convenience method for adding related tables without replacing the entire
        visible-foreign-keys annotation. Changes are staged until apply_annotations()
        is called.

        Args:
            table: Table name or Table object.
            context: The context to modify (typically "detailed" or "*").
            foreign_key: Foreign key to add. Can be:
                - List: inbound foreign key reference (e.g., ["schema", "Other_Table_fkey"])
                - Dict: pseudo-column definition for complex relationships
            position: Position to insert at (0-indexed). If None, appends to end.

        Returns:
            The updated foreign key list for the context.

        Raises:
            DerivaMLException: If context references another context.

        Example:
            >>> ml.add_visible_foreign_key("Subject", "detailed", ["domain", "Image_Subject_fkey"])
            >>> ml.apply_annotations()
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

        Convenience method for removing related tables without replacing the entire
        visible-foreign-keys annotation. Changes are staged until apply_annotations()
        is called.

        Args:
            table: Table name or Table object.
            context: The context to modify (e.g., "detailed", "*").
            foreign_key: Foreign key to remove. Can be:
                - List: foreign key reference [schema, constraint] to find and remove
                - Integer: index position to remove (0-indexed)

        Returns:
            The updated foreign key list for the context.

        Raises:
            DerivaMLException: If annotation or context doesn't exist, or foreign key not found.

        Example:
            >>> ml.remove_visible_foreign_key("Subject", "detailed", ["domain", "Image_Subject_fkey"])
            >>> ml.remove_visible_foreign_key("Subject", "detailed", 0)  # Remove first
            >>> ml.apply_annotations()
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
        reconstructing the list. Changes are staged until apply_annotations()
        is called.

        Args:
            table: Table name or Table object.
            context: The context to modify (e.g., "detailed", "*").
            new_order: The new order specification. Can be:
                - List of indices: [2, 0, 1] reorders by current positions
                - List of foreign key refs: [["schema", "fkey1"], ...] specifies exact order

        Returns:
            The reordered foreign key list.

        Raises:
            DerivaMLException: If annotation or context doesn't exist, or invalid order.

        Example:
            >>> ml.reorder_visible_foreign_keys("Subject", "detailed", [2, 0, 1])
            >>> ml.apply_annotations()
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
            >>> vars = ml.get_handlebars_template_variables("Image")
            >>> for col in vars["columns"]:
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
