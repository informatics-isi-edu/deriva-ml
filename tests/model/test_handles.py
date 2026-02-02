"""Tests for TableHandle and ColumnHandle wrapper classes."""

import pytest

from deriva_ml.core.enums import BuiltinTypes
from deriva_ml.model.handles import ColumnHandle, TableHandle


class TestColumnHandle:
    """Tests for ColumnHandle wrapper class."""

    def test_column_handle_creation(self, test_ml):
        """Test creating a ColumnHandle from an existing column."""
        ml = test_ml
        # Get an existing table
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        col = table.columns["Filename"]

        handle = ColumnHandle(col)
        assert handle.name == "Filename"
        assert handle._column is col

    def test_column_handle_delegation(self, test_ml):
        """Test that ColumnHandle delegates to underlying Column."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        col = table.columns["Filename"]
        handle = ColumnHandle(col)

        # Delegated properties should work
        assert handle.name == col.name
        assert handle.nullok == col.nullok
        assert handle.table == col.table

    def test_column_handle_description(self, test_ml):
        """Test getting and setting column description."""
        ml = test_ml
        ml.create_vocabulary("CV_ColDesc", "Test column description")
        table = ml.model.schemas[ml.default_schema].tables["CV_ColDesc"]

        handle = ColumnHandle(table.columns["Name"])

        # Initial description may be set by vocab definition
        original = handle.description

        # Set new description
        handle.description = "Test description for Name column"
        assert handle.description == "Test description for Name column"

        # Verify it persisted (the alter() call applies immediately)
        fresh_col = ml.model.schemas[ml.default_schema].tables["CV_ColDesc"].columns["Name"]
        assert fresh_col.comment == "Test description for Name column"

    def test_column_handle_column_type(self, test_ml):
        """Test getting column type as BuiltinTypes enum."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = ColumnHandle(table.columns["Filename"])

        col_type = handle.column_type
        assert isinstance(col_type, BuiltinTypes)
        assert col_type == BuiltinTypes.text

    def test_column_handle_is_system_column(self, test_ml):
        """Test identifying system columns."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]

        rid_handle = ColumnHandle(table.columns["RID"])
        assert rid_handle.is_system_column is True

        filename_handle = ColumnHandle(table.columns["Filename"])
        assert filename_handle.is_system_column is False

    def test_column_handle_display_name(self, test_ml):
        """Test setting and getting column display name."""
        ml = test_ml
        ml.create_vocabulary("CV_ColDisp", "Test column display")
        table = ml.model.schemas[ml.default_schema].tables["CV_ColDisp"]
        handle = ColumnHandle(table.columns["Name"])

        # Set display name
        handle.set_display_name("Friendly Name")

        # Get display name
        assert handle.get_display_name() == "Friendly Name"

    def test_column_handle_repr(self, test_ml):
        """Test ColumnHandle string representation."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = ColumnHandle(table.columns["Filename"])

        repr_str = repr(handle)
        assert "ColumnHandle" in repr_str
        assert "Filename" in repr_str


class TestTableHandle:
    """Tests for TableHandle wrapper class."""

    def test_table_handle_creation(self, test_ml):
        """Test creating a TableHandle from an existing table."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]

        handle = TableHandle(table)
        assert handle.name == "Image"
        assert handle._table is table

    def test_table_handle_delegation(self, test_ml):
        """Test that TableHandle delegates to underlying Table."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = TableHandle(table)

        # Delegated properties should work
        assert handle.name == table.name
        assert handle.schema == table.schema
        assert handle.foreign_keys == table.foreign_keys

    def test_table_handle_description(self, test_ml):
        """Test getting and setting table description."""
        ml = test_ml
        ml.create_vocabulary("CV_TblDesc", "Original description")
        table = ml.model.schemas[ml.default_schema].tables["CV_TblDesc"]
        handle = TableHandle(table)

        # Set new description
        handle.description = "Updated table description"
        assert handle.description == "Updated table description"

    def test_table_handle_get_column(self, test_ml):
        """Test getting a column by name."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = TableHandle(table)

        # Get existing column
        col = handle.get_column("Filename")
        assert col is not None
        assert isinstance(col, ColumnHandle)
        assert col.name == "Filename"

        # Get non-existent column
        col = handle.get_column("NonExistent")
        assert col is None

    def test_table_handle_column(self, test_ml):
        """Test getting a column with KeyError on missing."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = TableHandle(table)

        # Get existing column
        col = handle.column("Filename")
        assert col.name == "Filename"

        # Get non-existent column raises KeyError
        with pytest.raises(KeyError):
            handle.column("NonExistent")

    def test_table_handle_all_columns(self, test_ml):
        """Test iterating over all columns."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = TableHandle(table)

        columns = list(handle.all_columns)
        assert len(columns) > 0
        assert all(isinstance(c, ColumnHandle) for c in columns)

        # Should include system columns
        names = [c.name for c in columns]
        assert "RID" in names

    def test_table_handle_user_columns(self, test_ml):
        """Test getting non-system columns."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = TableHandle(table)

        user_cols = handle.user_columns
        names = [c.name for c in user_cols]

        # Should not include system columns
        assert "RID" not in names
        assert "RCT" not in names
        assert "RMT" not in names
        assert "RCB" not in names
        assert "RMB" not in names

        # Should include user columns
        assert "Filename" in names

    def test_table_handle_column_names(self, test_ml):
        """Test getting list of column names."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = TableHandle(table)

        names = handle.column_names
        assert isinstance(names, list)
        assert "Filename" in names
        assert "RID" in names

    def test_table_handle_has_column(self, test_ml):
        """Test checking if column exists."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = TableHandle(table)

        assert handle.has_column("Filename") is True
        assert handle.has_column("NonExistent") is False

    def test_table_handle_add_column(self, test_ml):
        """Test adding a new column to a table."""
        ml = test_ml
        ml.create_vocabulary("CV_AddCol", "Test add column")
        table = ml.model.schemas[ml.default_schema].tables["CV_AddCol"]
        handle = TableHandle(table)

        # Add a new column
        new_col = handle.add_column(
            name="TestColumn",
            column_type=BuiltinTypes.text,
            nullok=True,
            comment="A test column",
        )

        assert isinstance(new_col, ColumnHandle)
        assert new_col.name == "TestColumn"
        assert handle.has_column("TestColumn")

        # Adding duplicate should raise
        with pytest.raises(ValueError):
            handle.add_column("TestColumn", BuiltinTypes.text)

    def test_table_handle_display_name(self, test_ml):
        """Test setting and getting table display name."""
        ml = test_ml
        ml.create_vocabulary("CV_TblDisp", "Test table display")
        table = ml.model.schemas[ml.default_schema].tables["CV_TblDisp"]
        handle = TableHandle(table)

        # Set display name
        handle.set_display_name("Friendly Table Name")

        # Get display name
        assert handle.get_display_name() == "Friendly Table Name"

    def test_table_handle_row_name_pattern(self, test_ml):
        """Test setting and getting row name pattern."""
        ml = test_ml
        ml.create_vocabulary("CV_RowName", "Test row name")
        table = ml.model.schemas[ml.default_schema].tables["CV_RowName"]
        handle = TableHandle(table)

        # Set row name pattern
        handle.set_row_name_pattern("{{{Name}}}")

        # Get row name pattern
        assert handle.get_row_name_pattern() == "{{{Name}}}"

    def test_table_handle_visible_columns(self, test_ml):
        """Test setting and getting visible columns."""
        ml = test_ml
        ml.create_vocabulary("CV_VisCols", "Test visible columns")
        table = ml.model.schemas[ml.default_schema].tables["CV_VisCols"]
        handle = TableHandle(table)

        # Set visible columns
        handle.set_visible_columns(["RID", "Name", "Description"])

        # Get visible columns
        visible = handle.get_visible_columns()
        assert visible == ["RID", "Name", "Description"]

    def test_table_handle_add_remove_visible_column(self, test_ml):
        """Test adding and removing individual visible columns."""
        ml = test_ml
        ml.create_vocabulary("CV_AddVisCols", "Test add/remove visible columns")
        table = ml.model.schemas[ml.default_schema].tables["CV_AddVisCols"]
        handle = TableHandle(table)

        # Set initial visible columns
        handle.set_visible_columns(["RID", "Name"])

        # Add a column
        handle.add_visible_column("Description")
        visible = handle.get_visible_columns()
        assert "Description" in visible

        # Remove a column
        handle.remove_visible_column("RID")
        visible = handle.get_visible_columns()
        assert "RID" not in visible
        assert "Name" in visible
        assert "Description" in visible

    def test_table_handle_presence_annotations(self, test_ml):
        """Test is_generated, is_immutable, is_non_deletable properties."""
        ml = test_ml
        ml.create_vocabulary("CV_Presence", "Test presence annotations")
        table = ml.model.schemas[ml.default_schema].tables["CV_Presence"]
        handle = TableHandle(table)

        # Initially should be false
        assert handle.is_generated is False
        assert handle.is_immutable is False
        assert handle.is_non_deletable is False

        # Set to true
        handle.is_generated = True
        assert handle.is_generated is True

        handle.is_immutable = True
        assert handle.is_immutable is True

        handle.is_non_deletable = True
        assert handle.is_non_deletable is True

        # Set back to false
        handle.is_generated = False
        assert handle.is_generated is False

    def test_table_handle_repr(self, test_ml):
        """Test TableHandle string representation."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = TableHandle(table)

        repr_str = repr(handle)
        assert "TableHandle" in repr_str
        assert "Image" in repr_str

    def test_table_handle_column_caching(self, test_ml):
        """Test that ColumnHandle objects are cached."""
        ml = test_ml
        table = ml.model.schemas[ml.default_schema].tables["Image"]
        handle = TableHandle(table)

        # Get same column twice
        col1 = handle.get_column("Filename")
        col2 = handle.get_column("Filename")

        # Should be the same object
        assert col1 is col2


class TestTableColumnIntegration:
    """Integration tests for TableHandle and ColumnHandle working together."""

    def test_modify_column_via_table_handle(self, test_ml):
        """Test modifying a column obtained from TableHandle."""
        ml = test_ml
        ml.create_vocabulary("CV_Integration", "Test integration")
        table = ml.model.schemas[ml.default_schema].tables["CV_Integration"]
        handle = TableHandle(table)

        # Get column and modify
        col = handle.column("Name")
        col.description = "Modified via handle"

        # Verify change persisted
        fresh_handle = TableHandle(ml.model.schemas[ml.default_schema].tables["CV_Integration"])
        fresh_col = fresh_handle.column("Name")
        assert fresh_col.description == "Modified via handle"

    def test_iterate_and_modify_columns(self, test_ml):
        """Test iterating over columns and modifying them."""
        ml = test_ml
        ml.create_vocabulary("CV_IterMod", "Test iterate and modify")
        table = ml.model.schemas[ml.default_schema].tables["CV_IterMod"]
        handle = TableHandle(table)

        # Set display names for all user columns
        for col in handle.user_columns:
            col.set_display_name(f"Display: {col.name}")

        # Verify
        for col in handle.user_columns:
            assert col.get_display_name() == f"Display: {col.name}"

    def test_add_column_and_configure(self, test_ml):
        """Test adding a column and immediately configuring it."""
        ml = test_ml
        ml.create_vocabulary("CV_AddConfig", "Test add and configure")
        table = ml.model.schemas[ml.default_schema].tables["CV_AddConfig"]
        handle = TableHandle(table)

        # Add column
        col = handle.add_column(
            name="Status",
            column_type=BuiltinTypes.text,
            comment="Status of the item",
        )

        # Configure it
        col.set_display_name("Item Status")
        col.is_immutable = True

        # Verify
        assert col.get_display_name() == "Item Status"
        assert col.is_immutable is True
