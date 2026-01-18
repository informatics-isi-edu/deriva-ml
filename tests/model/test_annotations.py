"""Tests for annotation builder classes."""

import pytest

from deriva_ml.model.annotations import (
    # Builders
    Display,
    VisibleColumns,
    VisibleForeignKeys,
    TableDisplay,
    TableDisplayOptions,
    ColumnDisplay,
    ColumnDisplayOptions,
    PreFormat,
    PseudoColumn,
    PseudoColumnDisplay,
    Facet,
    FacetList,
    FacetRange,
    SortKey,
    NameStyle,
    # FK helpers
    InboundFK,
    OutboundFK,
    fk_constraint,
    # Enums
    TemplateEngine,
    Aggregate,
    ArrayUxMode,
    FacetUxMode,
    # Context constants
    CONTEXT_DEFAULT,
    CONTEXT_COMPACT,
    CONTEXT_DETAILED,
    CONTEXT_ENTRY,
    # Tags
    TAG_DISPLAY,
    TAG_VISIBLE_COLUMNS,
    TAG_VISIBLE_FOREIGN_KEYS,
    TAG_TABLE_DISPLAY,
    TAG_COLUMN_DISPLAY,
)


class TestDisplay:
    """Tests for Display annotation builder."""

    def test_simple_name(self):
        """Test display with simple name."""
        display = Display(name="My Table")
        assert display.tag == TAG_DISPLAY
        assert display.to_dict() == {"name": "My Table"}

    def test_markdown_name(self):
        """Test display with markdown name."""
        display = Display(markdown_name="**Bold** Name")
        assert display.to_dict() == {"markdown_name": "**Bold** Name"}

    def test_name_and_markdown_mutually_exclusive(self):
        """Test that name and markdown_name cannot both be set."""
        with pytest.raises(ValueError):
            Display(name="Name", markdown_name="**Name**")

    def test_with_comment(self):
        """Test display with comment."""
        display = Display(name="Table", comment="Description text")
        result = display.to_dict()
        assert result["name"] == "Table"
        assert result["comment"] == "Description text"

    def test_with_name_style(self):
        """Test display with name style."""
        display = Display(
            name_style=NameStyle(underline_space=True, title_case=True)
        )
        result = display.to_dict()
        assert result["name_style"] == {
            "underline_space": True,
            "title_case": True,
        }

    def test_with_show_null(self):
        """Test display with show_null options."""
        display = Display(
            name="Table",
            show_null={CONTEXT_COMPACT: False, CONTEXT_DETAILED: True}
        )
        result = display.to_dict()
        assert result["show_null"] == {
            "compact": False,
            "detailed": True,
        }

    def test_empty_display(self):
        """Test display with no options set."""
        display = Display()
        assert display.to_dict() == {}


class TestSortKey:
    """Tests for SortKey helper."""

    def test_ascending(self):
        """Test ascending sort key (returns string)."""
        key = SortKey("Name")
        assert key.to_dict() == "Name"

    def test_descending(self):
        """Test descending sort key (returns dict)."""
        key = SortKey("Created", descending=True)
        assert key.to_dict() == {"column": "Created", "descending": True}


class TestForeignKeyHelpers:
    """Tests for FK path helpers."""

    def test_inbound_fk(self):
        """Test inbound FK path step."""
        fk = InboundFK("domain", "Image_Subject_fkey")
        assert fk.to_dict() == {"inbound": ["domain", "Image_Subject_fkey"]}

    def test_outbound_fk(self):
        """Test outbound FK path step."""
        fk = OutboundFK("domain", "Subject_Species_fkey")
        assert fk.to_dict() == {"outbound": ["domain", "Subject_Species_fkey"]}

    def test_fk_constraint(self):
        """Test FK constraint reference."""
        ref = fk_constraint("domain", "Image_Subject_fkey")
        assert ref == ["domain", "Image_Subject_fkey"]


class TestPseudoColumn:
    """Tests for PseudoColumn builder."""

    def test_simple_source(self):
        """Test pseudo-column with simple source."""
        pc = PseudoColumn(source="Name", markdown_name="Subject Name")
        result = pc.to_dict()
        assert result["source"] == "Name"
        assert result["markdown_name"] == "Subject Name"

    def test_fk_path(self):
        """Test pseudo-column with FK traversal."""
        pc = PseudoColumn(
            source=[OutboundFK("domain", "Image_Subject_fkey"), "Name"],
            markdown_name="Subject"
        )
        result = pc.to_dict()
        assert result["source"] == [
            {"outbound": ["domain", "Image_Subject_fkey"]},
            "Name"
        ]
        assert result["markdown_name"] == "Subject"

    def test_with_aggregate(self):
        """Test pseudo-column with aggregate."""
        pc = PseudoColumn(
            source=[InboundFK("domain", "Image_Subject_fkey"), "RID"],
            aggregate=Aggregate.CNT,
            markdown_name="Image Count"
        )
        result = pc.to_dict()
        assert result["aggregate"] == "cnt"

    def test_source_and_sourcekey_mutually_exclusive(self):
        """Test that source and sourcekey cannot both be set."""
        with pytest.raises(ValueError):
            PseudoColumn(source="Name", sourcekey="my_source")

    def test_with_display_options(self):
        """Test pseudo-column with display options."""
        pc = PseudoColumn(
            source="URL",
            display=PseudoColumnDisplay(
                markdown_pattern="[Link]({{{_value}}})",
                show_foreign_key_link=False
            )
        )
        result = pc.to_dict()
        assert result["display"]["markdown_pattern"] == "[Link]({{{_value}}})"
        assert result["display"]["show_foreign_key_link"] is False


class TestVisibleColumns:
    """Tests for VisibleColumns builder."""

    def test_basic_contexts(self):
        """Test setting basic contexts."""
        vc = VisibleColumns()
        vc.compact(["RID", "Name"])
        vc.detailed(["RID", "Name", "Description"])

        result = vc.to_dict()
        assert vc.tag == TAG_VISIBLE_COLUMNS
        assert result["compact"] == ["RID", "Name"]
        assert result["detailed"] == ["RID", "Name", "Description"]

    def test_default_context(self):
        """Test setting default context."""
        vc = VisibleColumns()
        vc.default(["RID", "Name"])

        result = vc.to_dict()
        assert result["*"] == ["RID", "Name"]

    def test_chaining(self):
        """Test method chaining."""
        vc = (
            VisibleColumns()
            .compact(["RID", "Name"])
            .detailed(["RID", "Name", "Description"])
            .entry(["Name", "Description"])
        )

        result = vc.to_dict()
        assert "compact" in result
        assert "detailed" in result
        assert "entry" in result

    def test_with_pseudo_columns(self):
        """Test with pseudo-column entries."""
        vc = VisibleColumns()
        vc.compact([
            "RID",
            PseudoColumn(source="Name", markdown_name="Subject Name"),
            fk_constraint("domain", "Image_Subject_fkey"),
        ])

        result = vc.to_dict()
        assert result["compact"][0] == "RID"
        assert result["compact"][1]["source"] == "Name"
        assert result["compact"][2] == ["domain", "Image_Subject_fkey"]

    def test_context_reference(self):
        """Test referencing another context."""
        vc = VisibleColumns()
        vc.set_context("compact", ["RID", "Name"])
        vc.set_context("compact/brief", "compact")  # Reference

        result = vc.to_dict()
        assert result["compact/brief"] == "compact"


class TestVisibleForeignKeys:
    """Tests for VisibleForeignKeys builder."""

    def test_basic_usage(self):
        """Test basic FK visibility."""
        vfk = VisibleForeignKeys()
        vfk.detailed([
            fk_constraint("domain", "Image_Subject_fkey"),
            fk_constraint("domain", "Diagnosis_Subject_fkey"),
        ])

        result = vfk.to_dict()
        assert vfk.tag == TAG_VISIBLE_FOREIGN_KEYS
        assert len(result["detailed"]) == 2
        assert result["detailed"][0] == ["domain", "Image_Subject_fkey"]

    def test_default(self):
        """Test default context."""
        vfk = VisibleForeignKeys()
        vfk.default([fk_constraint("domain", "Related_fkey")])

        result = vfk.to_dict()
        assert result["*"] == [["domain", "Related_fkey"]]


class TestTableDisplay:
    """Tests for TableDisplay builder."""

    def test_row_name_pattern(self):
        """Test setting row name pattern."""
        td = TableDisplay()
        td.row_name("{{{Name}}} ({{{Species}}})")

        result = td.to_dict()
        assert td.tag == TAG_TABLE_DISPLAY
        assert result["row_name"]["row_markdown_pattern"] == "{{{Name}}} ({{{Species}}})"

    def test_compact_options(self):
        """Test compact view options."""
        td = TableDisplay()
        td.compact(TableDisplayOptions(
            row_order=[SortKey("Name"), SortKey("Created", descending=True)],
            page_size=25
        ))

        result = td.to_dict()
        assert result["compact"]["page_size"] == 25
        assert result["compact"]["row_order"] == [
            "Name",
            {"column": "Created", "descending": True}
        ]

    def test_with_template_engine(self):
        """Test setting template engine."""
        td = TableDisplay()
        td.row_name(
            "{{{Name}}}",
            template_engine=TemplateEngine.HANDLEBARS
        )

        result = td.to_dict()
        assert result["row_name"]["template_engine"] == "handlebars"


class TestColumnDisplay:
    """Tests for ColumnDisplay builder."""

    def test_pre_format(self):
        """Test pre-formatting options."""
        cd = ColumnDisplay()
        cd.default(ColumnDisplayOptions(
            pre_format=PreFormat(format="%.2f")
        ))

        result = cd.to_dict()
        assert cd.tag == TAG_COLUMN_DISPLAY
        assert result["*"]["pre_format"]["format"] == "%.2f"

    def test_boolean_format(self):
        """Test boolean formatting."""
        cd = ColumnDisplay()
        cd.default(ColumnDisplayOptions(
            pre_format=PreFormat(
                bool_true_value="Yes",
                bool_false_value="No"
            )
        ))

        result = cd.to_dict()
        assert result["*"]["pre_format"]["bool_true_value"] == "Yes"
        assert result["*"]["pre_format"]["bool_false_value"] == "No"

    def test_markdown_pattern(self):
        """Test markdown pattern."""
        cd = ColumnDisplay()
        cd.default(ColumnDisplayOptions(
            markdown_pattern="[Link]({{{_value}}})"
        ))

        result = cd.to_dict()
        assert result["*"]["markdown_pattern"] == "[Link]({{{_value}}})"


class TestFacet:
    """Tests for Facet builder."""

    def test_simple_facet(self):
        """Test simple facet."""
        facet = Facet(source="Species", open=True)
        result = facet.to_dict()
        assert result["source"] == "Species"
        assert result["open"] is True

    def test_facet_with_ux_mode(self):
        """Test facet with UX mode."""
        facet = Facet(
            source="Age",
            ux_mode=FacetUxMode.RANGES,
            ranges=[
                FacetRange(min=0, max=18),
                FacetRange(min=18, max=65),
                FacetRange(min=65),
            ]
        )
        result = facet.to_dict()
        assert result["ux_mode"] == "ranges"
        assert len(result["ranges"]) == 3

    def test_facet_with_choices(self):
        """Test facet with preset choices."""
        facet = Facet(
            source="Status",
            ux_mode=FacetUxMode.CHOICES,
            choices=["Active", "Inactive", "Pending"]
        )
        result = facet.to_dict()
        assert result["choices"] == ["Active", "Inactive", "Pending"]

    def test_facet_with_fk_path(self):
        """Test facet with FK traversal."""
        facet = Facet(
            source=[OutboundFK("domain", "Image_Subject_fkey"), "Species"],
            markdown_name="Species"
        )
        result = facet.to_dict()
        assert result["source"][0] == {"outbound": ["domain", "Image_Subject_fkey"]}


class TestFacetList:
    """Tests for FacetList builder."""

    def test_facet_list(self):
        """Test creating a facet list."""
        facets = FacetList([
            Facet(source="Species", open=True),
            Facet(source="Age", ux_mode=FacetUxMode.RANGES),
        ])

        result = facets.to_dict()
        assert "and" in result
        assert len(result["and"]) == 2

    def test_add_facet(self):
        """Test adding facets via add method."""
        facets = FacetList()
        facets.add(Facet(source="Name"))
        facets.add(Facet(source="Status"))

        result = facets.to_dict()
        assert len(result["and"]) == 2


class TestEnums:
    """Tests for enum values."""

    def test_template_engine(self):
        """Test template engine values."""
        assert TemplateEngine.HANDLEBARS.value == "handlebars"
        assert TemplateEngine.MUSTACHE.value == "mustache"

    def test_aggregate(self):
        """Test aggregate values."""
        assert Aggregate.CNT.value == "cnt"
        assert Aggregate.ARRAY.value == "array"
        assert Aggregate.CNT_D.value == "cnt_d"

    def test_array_ux_mode(self):
        """Test array UX mode values."""
        assert ArrayUxMode.CSV.value == "csv"
        assert ArrayUxMode.OLIST.value == "olist"

    def test_facet_ux_mode(self):
        """Test facet UX mode values."""
        assert FacetUxMode.CHOICES.value == "choices"
        assert FacetUxMode.RANGES.value == "ranges"
        assert FacetUxMode.CHECK_PRESENCE.value == "check_presence"


class TestContextConstants:
    """Tests for context constants."""

    def test_context_values(self):
        """Test context constant values."""
        assert CONTEXT_DEFAULT == "*"
        assert CONTEXT_COMPACT == "compact"
        assert CONTEXT_DETAILED == "detailed"
        assert CONTEXT_ENTRY == "entry"


class TestComplexScenarios:
    """Integration tests for complex annotation scenarios."""

    def test_full_visible_columns_config(self):
        """Test a complete visible columns configuration."""
        vc = VisibleColumns()

        # Compact view: basic columns
        vc.compact([
            "RID",
            "Name",
            fk_constraint("domain", "Image_Subject_fkey"),
        ])

        # Detailed view: more columns with pseudo-column
        vc.detailed([
            "RID",
            "Name",
            PseudoColumn(
                source=[OutboundFK("domain", "Image_Subject_fkey"), "Name"],
                markdown_name="Subject Name"
            ),
            "Description",
        ])

        # Entry view: editable columns
        vc.entry(["Name", "Description"])

        result = vc.to_dict()
        assert len(result) == 3
        assert isinstance(result["detailed"][2], dict)
        assert result["detailed"][2]["markdown_name"] == "Subject Name"

    def test_full_table_display_config(self):
        """Test a complete table display configuration."""
        td = TableDisplay()

        # Row name pattern
        td.row_name("{{{Name}}} - {{{RID}}}")

        # Compact view options
        td.compact(TableDisplayOptions(
            row_order=[SortKey("Name")],
            page_size=50,
        ))

        # Detailed view options
        td.detailed(TableDisplayOptions(
            collapse_toc_panel=True,
        ))

        result = td.to_dict()
        assert "row_name" in result
        assert result["compact"]["page_size"] == 50
        assert result["detailed"]["collapse_toc_panel"] is True
