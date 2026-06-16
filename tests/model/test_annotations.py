"""Tests for annotation builder classes."""

import pytest
from deriva.core import tag as deriva_tag

from deriva_ml.core.mixins.annotation import DISPLAY_TAG
from deriva_ml.model.annotations import (
    CONTEXT_COMPACT,
    # Context constants
    CONTEXT_DEFAULT,
    CONTEXT_DETAILED,
    CONTEXT_ENTRY,
    TAG_COLUMN_DISPLAY,
    # Tags
    TAG_DISPLAY,
    TAG_TABLE_DISPLAY,
    TAG_VISIBLE_COLUMNS,
    TAG_VISIBLE_FOREIGN_KEYS,
    Aggregate,
    ArrayUxMode,
    ColumnDisplay,
    ColumnDisplayOptions,
    # Builders
    Display,
    Facet,
    FacetList,
    FacetRange,
    FacetUxMode,
    # FK helpers
    InboundFK,
    NameStyle,
    OutboundFK,
    PreFormat,
    PseudoColumn,
    PseudoColumnDisplay,
    SortKey,
    TableDisplay,
    TableDisplayOptions,
    # Enums
    TemplateEngine,
    VisibleColumns,
    VisibleForeignKeys,
    fk_constraint,
)


class TestDisplay:
    """Tests for Display annotation builder."""

    def test_simple_name(self):
        """Test display with simple name."""
        display = Display(name="My Table")
        # Pin against deriva-py's authoritative tag, not the local constant
        # (asserting against TAG_DISPLAY here would be a tautology).
        assert display.tag == deriva_tag.display
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
        display = Display(name_style=NameStyle(underline_space=True, title_case=True))
        result = display.to_dict()
        assert result["name_style"] == {
            "underline_space": True,
            "title_case": True,
        }

    def test_with_show_null(self):
        """Test display with show_null options."""
        display = Display(name="Table", show_null={CONTEXT_COMPACT: False, CONTEXT_DETAILED: True})
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
        pc = PseudoColumn(source=[OutboundFK("domain", "Image_Subject_fkey"), "Name"], markdown_name="Subject")
        result = pc.to_dict()
        assert result["source"] == [{"outbound": ["domain", "Image_Subject_fkey"]}, "Name"]
        assert result["markdown_name"] == "Subject"

    def test_with_aggregate(self):
        """Test pseudo-column with aggregate."""
        pc = PseudoColumn(
            source=[InboundFK("domain", "Image_Subject_fkey"), "RID"],
            aggregate=Aggregate.CNT,
            markdown_name="Image Count",
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
            display=PseudoColumnDisplay(markdown_pattern="[Link]({{{_value}}})", show_foreign_key_link=False),
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
        vc.compact(
            [
                "RID",
                PseudoColumn(source="Name", markdown_name="Subject Name"),
                fk_constraint("domain", "Image_Subject_fkey"),
            ]
        )

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
        vfk.detailed(
            [
                fk_constraint("domain", "Image_Subject_fkey"),
                fk_constraint("domain", "Diagnosis_Subject_fkey"),
            ]
        )

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
        td.compact(TableDisplayOptions(row_order=[SortKey("Name"), SortKey("Created", descending=True)], page_size=25))

        result = td.to_dict()
        assert result["compact"]["page_size"] == 25
        assert result["compact"]["row_order"] == ["Name", {"column": "Created", "descending": True}]

    def test_with_template_engine(self):
        """Test setting template engine."""
        td = TableDisplay()
        td.row_name("{{{Name}}}", template_engine=TemplateEngine.HANDLEBARS)

        result = td.to_dict()
        assert result["row_name"]["template_engine"] == "handlebars"


class TestColumnDisplay:
    """Tests for ColumnDisplay builder."""

    def test_pre_format(self):
        """Test pre-formatting options."""
        cd = ColumnDisplay()
        cd.default(ColumnDisplayOptions(pre_format=PreFormat(format="%.2f")))

        result = cd.to_dict()
        assert cd.tag == TAG_COLUMN_DISPLAY
        assert result["*"]["pre_format"]["format"] == "%.2f"

    def test_boolean_format(self):
        """Test boolean formatting."""
        cd = ColumnDisplay()
        cd.default(ColumnDisplayOptions(pre_format=PreFormat(bool_true_value="Yes", bool_false_value="No")))

        result = cd.to_dict()
        assert result["*"]["pre_format"]["bool_true_value"] == "Yes"
        assert result["*"]["pre_format"]["bool_false_value"] == "No"

    def test_markdown_pattern(self):
        """Test markdown pattern."""
        cd = ColumnDisplay()
        cd.default(ColumnDisplayOptions(markdown_pattern="[Link]({{{_value}}})"))

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
            ],
        )
        result = facet.to_dict()
        assert result["ux_mode"] == "ranges"
        assert len(result["ranges"]) == 3

    def test_facet_with_choices(self):
        """Test facet with preset choices."""
        facet = Facet(source="Status", ux_mode=FacetUxMode.CHOICES, choices=["Active", "Inactive", "Pending"])
        result = facet.to_dict()
        assert result["choices"] == ["Active", "Inactive", "Pending"]

    def test_facet_with_fk_path(self):
        """Test facet with FK traversal."""
        facet = Facet(source=[OutboundFK("domain", "Image_Subject_fkey"), "Species"], markdown_name="Species")
        result = facet.to_dict()
        assert result["source"][0] == {"outbound": ["domain", "Image_Subject_fkey"]}


class TestFacetList:
    """Tests for FacetList builder."""

    def test_facet_list(self):
        """Test creating a facet list."""
        facets = FacetList(
            [
                Facet(source="Species", open=True),
                Facet(source="Age", ux_mode=FacetUxMode.RANGES),
            ]
        )

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
        vc.compact(
            [
                "RID",
                "Name",
                fk_constraint("domain", "Image_Subject_fkey"),
            ]
        )

        # Detailed view: more columns with pseudo-column
        vc.detailed(
            [
                "RID",
                "Name",
                PseudoColumn(source=[OutboundFK("domain", "Image_Subject_fkey"), "Name"], markdown_name="Subject Name"),
                "Description",
            ]
        )

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
        td.compact(
            TableDisplayOptions(
                row_order=[SortKey("Name")],
                page_size=50,
            )
        )

        # Detailed view options
        td.detailed(
            TableDisplayOptions(
                collapse_toc_panel=True,
            )
        )

        result = td.to_dict()
        assert "row_name" in result
        assert result["compact"]["page_size"] == 50
        assert result["detailed"]["collapse_toc_panel"] is True


class TestExternalConsumerContract:
    """End-to-end coverage of the ``deriva-skills/use-annotation-builders`` pattern.

    The annotation builders in :mod:`deriva_ml.model.annotations` are an
    **externally-consumed public API**, used by the
    ``deriva-skills/use-annotation-builders`` Claude Code skill. The skill's
    canonical pattern (see ``SKILL.md`` in the deriva-skills plugin) is::

        from deriva_ml.model.annotations import Display, VisibleColumns
        display = Display(name="Images", markdown_name=None)
        table.annotations[Display.tag] = display.to_dict()
        ml.apply_annotations()

    These tests pin that contract: every builder we export has a class-level
    ``tag`` attribute and an instance-level ``to_dict()`` method, the result
    is JSON-serializable, and the produced dict is shaped so the
    ``table.annotations[Builder.tag] = builder.to_dict()`` line works.

    No live catalog is needed — we exercise the contract against a plain dict
    that mirrors what the deriva-py Table's ``annotations`` mapping accepts.
    """

    # Every annotation builder that ``model/__init__.py`` re-exports under
    # the ``# Annotation builders`` block. If the public surface grows or
    # shrinks, this list must move with it — that mismatch is the test's
    # whole point.
    EXPORTED_BUILDERS: tuple[type, ...] = (
        Display,
        VisibleColumns,
        VisibleForeignKeys,
        TableDisplay,
        ColumnDisplay,
    )

    def test_every_builder_has_a_tag(self):
        """Class-level ``.tag`` is the dict key for ``table.annotations``."""
        for cls in self.EXPORTED_BUILDERS:
            assert hasattr(cls, "tag"), f"{cls.__name__}.tag missing"
            tag = cls.tag
            assert isinstance(tag, str) and tag, f"{cls.__name__}.tag is not a usable string"

    def test_every_builder_has_to_dict(self):
        """Instance-level ``.to_dict()`` is what gets assigned to the annotation."""
        # Use a trivial constructor call that should succeed for each.
        instances = [
            Display(name="X"),
            VisibleColumns(),
            VisibleForeignKeys(),
            TableDisplay(),
            ColumnDisplay(),
        ]
        for instance in instances:
            assert hasattr(instance, "to_dict"), f"{type(instance).__name__}.to_dict missing"
            payload = instance.to_dict()
            assert isinstance(payload, dict), (
                f"{type(instance).__name__}.to_dict() returned {type(payload).__name__}, expected dict"
            )

    def test_skill_application_pattern_with_dict_mock(self):
        """Apply each builder via the ``annotations[tag] = to_dict()`` pattern.

        Mirrors the SKILL.md flow against a plain dict (no live catalog).
        The annotations dict ends up with one entry per builder, and each
        entry is JSON-serializable — i.e. equivalent to what deriva-py
        would push over the wire on ``ml.apply_annotations()``.
        """
        import json

        # Stand-in for ``table.annotations`` — a real ``Table.annotations``
        # is a dict subclass, so a bare dict matches the contract surface.
        annotations: dict[str, dict] = {}

        display = Display(name="Image")
        annotations[Display.tag] = display.to_dict()

        vc = VisibleColumns()
        vc.compact(["RID", "Filename"])
        annotations[VisibleColumns.tag] = vc.to_dict()

        vfk = VisibleForeignKeys()
        annotations[VisibleForeignKeys.tag] = vfk.to_dict()

        # Each entry must round-trip JSON — that's the wire format Chaise consumes.
        for tag, payload in annotations.items():
            roundtripped = json.loads(json.dumps(payload))
            assert roundtripped == payload, f"Annotation {tag!r} payload is not JSON-stable"

        # And the keys are exactly the builder tags.
        assert Display.tag in annotations
        assert VisibleColumns.tag in annotations
        assert VisibleForeignKeys.tag in annotations

    def test_apply_annotations_method_exists_on_derivaml(self):
        """SKILL.md's apply step is ``ml.apply_annotations()`` — verify the entry point.

        This is a smoke test on the API surface — we don't construct a live
        DerivaML (no catalog), but we do verify the method exists and has
        the expected signature shape so a future refactor doesn't silently
        drop the apply entry point the skill relies on.
        """
        from deriva_ml import DerivaML

        assert hasattr(DerivaML, "apply_annotations"), (
            "DerivaML.apply_annotations is the documented skill apply step; do not remove it."
        )
        # No required positional args beyond self.
        import inspect

        sig = inspect.signature(DerivaML.apply_annotations)
        positional_required = [
            p
            for p in sig.parameters.values()
            if p.name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        assert positional_required == [], (
            f"DerivaML.apply_annotations grew required positional args: {positional_required}. "
            "The deriva-skills/use-annotation-builders skill calls it as ml.apply_annotations() "
            "with no arguments — any required arg here is a breaking change for that contract."
        )


class TestDisplayTagAuthority:
    """Pin deriva-ml's 2015 display tag to deriva-py's canonical value.

    The 2015 ``display`` annotation is grandfathered under the historical
    ``misd`` namespace; Chaise reads display annotations from
    ``tag:misd.isi.edu,2015:display``. deriva-ml previously hardcoded the
    ``isrd`` namespace, so display annotations were written under a tag
    Chaise ignores. These assertions import the authoritative value from
    ``deriva.core.tag`` (re-exported from
    ``deriva.core.utils.core_utils``) so the constants can never drift
    from deriva-py again — and so the tautology that let the original bug
    survive (asserting a builder's tag against the constant it is set
    from) cannot reoccur.
    """

    def test_authority_value_is_misd(self):
        """deriva-py's canonical display tag is in the misd namespace."""
        assert deriva_tag.display == "tag:misd.isi.edu,2015:display"

    def test_display_builder_tag_matches_deriva_py(self):
        """Display builder writes under deriva-py's authoritative tag."""
        assert Display.tag == deriva_tag.display

    def test_tag_display_constant_matches_deriva_py(self):
        """annotations.TAG_DISPLAY matches deriva-py's authoritative tag."""
        assert TAG_DISPLAY == deriva_tag.display

    def test_display_tag_mixin_constant_matches_deriva_py(self):
        """mixins.annotation.DISPLAY_TAG matches deriva-py's authoritative tag."""
        assert DISPLAY_TAG == deriva_tag.display
