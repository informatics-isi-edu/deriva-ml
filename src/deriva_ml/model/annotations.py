"""Annotation helper classes for DerivaML.

This module provides lightweight helper classes for building common annotation
structures. These are NOT comprehensive Pydantic models of the full annotation
schema - instead they are convenience builders for the most common use cases.

The classes help with:
- IDE autocompletion and type hints
- Reducing typos in annotation keys
- Making code more readable
- Documenting common patterns

For full schema details, see the Deriva annotation documentation.

Example:
    >>> from deriva_ml.model.annotations import Display, VisibleColumns, RowNamePattern
    >>>
    >>> # Build display annotation
    >>> display = Display(name="Friendly Name")
    >>> table_handle.set_annotation(display)
    >>>
    >>> # Build visible columns with contexts
    >>> vis_cols = VisibleColumns()
    >>> vis_cols.compact(["RID", "Name", "Description"])
    >>> vis_cols.detailed(["RID", "Name", "Description", "Created"])
    >>> table_handle.set_annotation(vis_cols)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# =============================================================================
# Enums for constrained values
# =============================================================================

class TemplateEngine(str, Enum):
    """Template engine for markdown patterns."""
    HANDLEBARS = "handlebars"
    MUSTACHE = "mustache"


class Aggregate(str, Enum):
    """Aggregation functions for pseudo-columns."""
    MIN = "min"
    MAX = "max"
    CNT = "cnt"
    CNT_D = "cnt_d"
    ARRAY = "array"
    ARRAY_D = "array_d"


class ArrayUxMode(str, Enum):
    """Display modes for array values."""
    RAW = "raw"
    CSV = "csv"
    OLIST = "olist"
    ULIST = "ulist"


class FacetUxMode(str, Enum):
    """UX modes for facet filters."""
    CHOICES = "choices"
    RANGES = "ranges"
    CHECK_PRESENCE = "check_presence"


# =============================================================================
# Context Names
# =============================================================================

# Standard context names
CONTEXT_DEFAULT = "*"
CONTEXT_COMPACT = "compact"
CONTEXT_COMPACT_BRIEF = "compact/brief"
CONTEXT_COMPACT_BRIEF_INLINE = "compact/brief/inline"
CONTEXT_COMPACT_SELECT = "compact/select"
CONTEXT_DETAILED = "detailed"
CONTEXT_ENTRY = "entry"
CONTEXT_ENTRY_CREATE = "entry/create"
CONTEXT_ENTRY_EDIT = "entry/edit"
CONTEXT_EXPORT = "export"
CONTEXT_FILTER = "filter"
CONTEXT_ROW_NAME = "row_name"


# =============================================================================
# Annotation Tag URIs
# =============================================================================

TAG_DISPLAY = "tag:isrd.isi.edu,2015:display"
TAG_VISIBLE_COLUMNS = "tag:isrd.isi.edu,2016:visible-columns"
TAG_VISIBLE_FOREIGN_KEYS = "tag:isrd.isi.edu,2016:visible-foreign-keys"
TAG_TABLE_DISPLAY = "tag:isrd.isi.edu,2016:table-display"
TAG_COLUMN_DISPLAY = "tag:isrd.isi.edu,2016:column-display"
TAG_SOURCE_DEFINITIONS = "tag:isrd.isi.edu,2019:source-definitions"


# =============================================================================
# Base Protocol for Annotations
# =============================================================================

class AnnotationBuilder:
    """Base class for annotation builders.

    Subclasses must implement:
    - tag: The annotation tag URI
    - to_dict(): Convert to dictionary representation
    """

    tag: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for catalog annotation."""
        raise NotImplementedError


# =============================================================================
# Display Annotation
# =============================================================================

@dataclass
class NameStyle:
    """Styling options for display names.

    Args:
        underline_space: Replace underscores with spaces
        title_case: Apply title case formatting
        markdown: Render as markdown
    """
    underline_space: bool | None = None
    title_case: bool | None = None
    markdown: bool | None = None

    def to_dict(self) -> dict[str, bool]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.underline_space is not None:
            result["underline_space"] = self.underline_space
        if self.title_case is not None:
            result["title_case"] = self.title_case
        if self.markdown is not None:
            result["markdown"] = self.markdown
        return result


@dataclass
class Display(AnnotationBuilder):
    """Display annotation for tables and columns.

    Controls basic naming and display options.

    Args:
        name: Display name (mutually exclusive with markdown_name)
        markdown_name: Markdown-formatted display name
        name_style: Styling options for the name
        comment: Description/tooltip text
        show_null: How to show null values (by context)
        show_foreign_key_link: Show FK as link (by context)

    Example:
        >>> display = Display(name="Friendly Name")
        >>> display = Display(
        ...     markdown_name="**Bold** Name",
        ...     comment="Description text",
        ...     show_null={CONTEXT_COMPACT: False}
        ... )
    """
    tag = TAG_DISPLAY

    name: str | None = None
    markdown_name: str | None = None
    name_style: NameStyle | None = None
    comment: str | None = None
    show_null: dict[str, bool | str] | None = None
    show_foreign_key_link: dict[str, bool] | None = None

    def __post_init__(self):
        if self.name and self.markdown_name:
            raise ValueError("name and markdown_name are mutually exclusive")

    def to_dict(self) -> dict[str, Any]:
        result = {}
        if self.name is not None:
            result["name"] = self.name
        if self.markdown_name is not None:
            result["markdown_name"] = self.markdown_name
        if self.name_style is not None:
            style_dict = self.name_style.to_dict()
            if style_dict:
                result["name_style"] = style_dict
        if self.comment is not None:
            result["comment"] = self.comment
        if self.show_null is not None:
            result["show_null"] = self.show_null
        if self.show_foreign_key_link is not None:
            result["show_foreign_key_link"] = self.show_foreign_key_link
        return result


# =============================================================================
# Sort Key
# =============================================================================

@dataclass
class SortKey:
    """A sort key for row ordering.

    Args:
        column: Column name to sort by
        descending: Sort in descending order (default False)

    Example:
        >>> SortKey("Name")  # Ascending
        >>> SortKey("Created", descending=True)  # Descending
    """
    column: str
    descending: bool = False

    def to_dict(self) -> dict[str, Any] | str:
        """Convert to dict or string (if ascending)."""
        if self.descending:
            return {"column": self.column, "descending": True}
        return self.column


# =============================================================================
# Foreign Key Path Components
# =============================================================================

@dataclass
class InboundFK:
    """An inbound foreign key path step.

    Args:
        schema: Schema name containing the FK
        constraint: Constraint name
    """
    schema: str
    constraint: str

    def to_dict(self) -> dict[str, list[str]]:
        return {"inbound": [self.schema, self.constraint]}


@dataclass
class OutboundFK:
    """An outbound foreign key path step.

    Args:
        schema: Schema name containing the FK
        constraint: Constraint name
    """
    schema: str
    constraint: str

    def to_dict(self) -> dict[str, list[str]]:
        return {"outbound": [self.schema, self.constraint]}


def fk_constraint(schema: str, constraint: str) -> list[str]:
    """Create a foreign key constraint reference.

    Args:
        schema: Schema name
        constraint: Constraint name

    Returns:
        [schema, constraint] list suitable for annotations

    Example:
        >>> fk_constraint("domain", "Image_Subject_fkey")
        ['domain', 'Image_Subject_fkey']
    """
    return [schema, constraint]


# =============================================================================
# Pseudo-Column Display Options
# =============================================================================

@dataclass
class PseudoColumnDisplay:
    """Display options for a pseudo-column.

    Args:
        markdown_pattern: Handlebars/mustache template
        template_engine: Template engine to use
        show_foreign_key_link: Show as clickable link
        array_ux_mode: How to render array values
        column_order: Sort order for the column, or False to disable
        wait_for: Template variables to wait for before rendering
    """
    markdown_pattern: str | None = None
    template_engine: TemplateEngine | None = None
    show_foreign_key_link: bool | None = None
    array_ux_mode: ArrayUxMode | None = None
    column_order: list[SortKey] | Literal[False] | None = None
    wait_for: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {}
        if self.markdown_pattern is not None:
            result["markdown_pattern"] = self.markdown_pattern
        if self.template_engine is not None:
            result["template_engine"] = self.template_engine.value
        if self.show_foreign_key_link is not None:
            result["show_foreign_key_link"] = self.show_foreign_key_link
        if self.array_ux_mode is not None:
            result["array_ux_mode"] = self.array_ux_mode.value
        if self.column_order is not None:
            if self.column_order is False:
                result["column_order"] = False
            else:
                result["column_order"] = [
                    k.to_dict() if isinstance(k, SortKey) else k
                    for k in self.column_order
                ]
        if self.wait_for is not None:
            result["wait_for"] = self.wait_for
        return result


# =============================================================================
# Pseudo-Column (for visible-columns and visible-foreign-keys)
# =============================================================================

@dataclass
class PseudoColumn:
    """A pseudo-column definition for visible columns/foreign keys.

    Pseudo-columns allow displaying computed values, values from related tables,
    or custom markdown patterns in table views.

    Args:
        source: Path to source data (column name, or list with FK paths)
        sourcekey: Reference to a named source in source-definitions
        markdown_name: Display name (supports markdown)
        comment: Description/tooltip
        entity: Whether this represents an entity
        aggregate: Aggregation function for array sources
        self_link: Make row a self-link
        display: Display options
        array_options: Options for array aggregates

    Note: source and sourcekey are mutually exclusive.

    Example:
        >>> # Simple column reference
        >>> PseudoColumn(source="Name", markdown_name="Subject Name")
        >>>
        >>> # FK traversal
        >>> PseudoColumn(
        ...     source=[OutboundFK("domain", "Image_Subject_fkey"), "Name"],
        ...     markdown_name="Subject"
        ... )
        >>>
        >>> # With aggregate
        >>> PseudoColumn(
        ...     source=[InboundFK("domain", "Image_Subject_fkey"), "RID"],
        ...     aggregate=Aggregate.CNT,
        ...     markdown_name="Image Count"
        ... )
    """
    source: str | list[str | InboundFK | OutboundFK] | None = None
    sourcekey: str | None = None
    markdown_name: str | None = None
    comment: str | Literal[False] | None = None
    entity: bool | None = None
    aggregate: Aggregate | None = None
    self_link: bool | None = None
    display: PseudoColumnDisplay | None = None
    array_options: dict[str, Any] | None = None  # Can be complex

    def __post_init__(self):
        if self.source is not None and self.sourcekey is not None:
            raise ValueError("source and sourcekey are mutually exclusive")

    def to_dict(self) -> dict[str, Any]:
        result = {}

        if self.source is not None:
            if isinstance(self.source, str):
                result["source"] = self.source
            else:
                # Convert path elements
                result["source"] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in self.source
                ]

        if self.sourcekey is not None:
            result["sourcekey"] = self.sourcekey
        if self.markdown_name is not None:
            result["markdown_name"] = self.markdown_name
        if self.comment is not None:
            result["comment"] = self.comment
        if self.entity is not None:
            result["entity"] = self.entity
        if self.aggregate is not None:
            result["aggregate"] = self.aggregate.value
        if self.self_link is not None:
            result["self_link"] = self.self_link
        if self.display is not None:
            result["display"] = self.display.to_dict()
        if self.array_options is not None:
            result["array_options"] = self.array_options

        return result


# =============================================================================
# Visible Columns Annotation
# =============================================================================

# Type for a single column entry
ColumnEntry = str | list[str] | PseudoColumn


@dataclass
class VisibleColumns(AnnotationBuilder):
    """Visible-columns annotation builder.

    Controls which columns appear in different UI contexts.

    Example:
        >>> vc = VisibleColumns()
        >>> vc.set_context(CONTEXT_COMPACT, ["RID", "Name"])
        >>> vc.set_context(CONTEXT_DETAILED, ["RID", "Name", "Description"])
        >>>
        >>> # Or use convenience methods
        >>> vc = VisibleColumns()
        >>> vc.compact(["RID", "Name"])
        >>> vc.detailed(["RID", "Name", "Description"])
    """
    tag = TAG_VISIBLE_COLUMNS

    _contexts: dict[str, list[ColumnEntry] | str] = field(default_factory=dict)

    def set_context(
        self,
        context: str,
        columns: list[ColumnEntry] | str
    ) -> "VisibleColumns":
        """Set columns for a context.

        Args:
            context: Context name (e.g., "compact", "detailed", "*")
            columns: List of columns, or string referencing another context

        Returns:
            Self for chaining
        """
        self._contexts[context] = columns
        return self

    def compact(self, columns: list[ColumnEntry]) -> "VisibleColumns":
        """Set columns for compact (list) view."""
        return self.set_context(CONTEXT_COMPACT, columns)

    def detailed(self, columns: list[ColumnEntry]) -> "VisibleColumns":
        """Set columns for detailed (record) view."""
        return self.set_context(CONTEXT_DETAILED, columns)

    def entry(self, columns: list[ColumnEntry]) -> "VisibleColumns":
        """Set columns for entry (create/edit) forms."""
        return self.set_context(CONTEXT_ENTRY, columns)

    def entry_create(self, columns: list[ColumnEntry]) -> "VisibleColumns":
        """Set columns for create form only."""
        return self.set_context(CONTEXT_ENTRY_CREATE, columns)

    def entry_edit(self, columns: list[ColumnEntry]) -> "VisibleColumns":
        """Set columns for edit form only."""
        return self.set_context(CONTEXT_ENTRY_EDIT, columns)

    def default(self, columns: list[ColumnEntry]) -> "VisibleColumns":
        """Set default columns for all contexts."""
        return self.set_context(CONTEXT_DEFAULT, columns)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for context, columns in self._contexts.items():
            if isinstance(columns, str):
                result[context] = columns
            else:
                result[context] = [
                    c.to_dict() if isinstance(c, PseudoColumn) else c
                    for c in columns
                ]
        return result


# =============================================================================
# Visible Foreign Keys Annotation
# =============================================================================

# Type for a single FK entry
ForeignKeyEntry = list[str] | PseudoColumn


@dataclass
class VisibleForeignKeys(AnnotationBuilder):
    """Visible-foreign-keys annotation builder.

    Controls which related tables appear in the UI via inbound foreign keys.

    Example:
        >>> vfk = VisibleForeignKeys()
        >>> vfk.detailed([
        ...     fk_constraint("domain", "Image_Subject_fkey"),
        ...     fk_constraint("domain", "Diagnosis_Subject_fkey")
        ... ])
    """
    tag = TAG_VISIBLE_FOREIGN_KEYS

    _contexts: dict[str, list[ForeignKeyEntry] | str] = field(default_factory=dict)

    def set_context(
        self,
        context: str,
        foreign_keys: list[ForeignKeyEntry] | str
    ) -> "VisibleForeignKeys":
        """Set foreign keys for a context."""
        self._contexts[context] = foreign_keys
        return self

    def detailed(self, foreign_keys: list[ForeignKeyEntry]) -> "VisibleForeignKeys":
        """Set foreign keys for detailed view."""
        return self.set_context(CONTEXT_DETAILED, foreign_keys)

    def default(self, foreign_keys: list[ForeignKeyEntry]) -> "VisibleForeignKeys":
        """Set default foreign keys for all contexts."""
        return self.set_context(CONTEXT_DEFAULT, foreign_keys)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for context, fkeys in self._contexts.items():
            if isinstance(fkeys, str):
                result[context] = fkeys
            else:
                result[context] = [
                    fk.to_dict() if isinstance(fk, PseudoColumn) else fk
                    for fk in fkeys
                ]
        return result


# =============================================================================
# Table Display Annotation
# =============================================================================

@dataclass
class TableDisplayOptions:
    """Options for a single table display context.

    Args:
        row_order: Sort order for rows
        page_size: Number of rows per page
        row_markdown_pattern: Template for row names
        page_markdown_pattern: Template for page header
        separator_markdown: Template between rows
        prefix_markdown: Template before rows
        suffix_markdown: Template after rows
        template_engine: Template engine for patterns
        collapse_toc_panel: Collapse TOC panel
        hide_column_headers: Hide column headers
    """
    row_order: list[SortKey] | None = None
    page_size: int | None = None
    row_markdown_pattern: str | None = None
    page_markdown_pattern: str | None = None
    separator_markdown: str | None = None
    prefix_markdown: str | None = None
    suffix_markdown: str | None = None
    template_engine: TemplateEngine | None = None
    collapse_toc_panel: bool | None = None
    hide_column_headers: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {}
        if self.row_order is not None:
            result["row_order"] = [
                k.to_dict() if isinstance(k, SortKey) else k
                for k in self.row_order
            ]
        if self.page_size is not None:
            result["page_size"] = self.page_size
        if self.row_markdown_pattern is not None:
            result["row_markdown_pattern"] = self.row_markdown_pattern
        if self.page_markdown_pattern is not None:
            result["page_markdown_pattern"] = self.page_markdown_pattern
        if self.separator_markdown is not None:
            result["separator_markdown"] = self.separator_markdown
        if self.prefix_markdown is not None:
            result["prefix_markdown"] = self.prefix_markdown
        if self.suffix_markdown is not None:
            result["suffix_markdown"] = self.suffix_markdown
        if self.template_engine is not None:
            result["template_engine"] = self.template_engine.value
        if self.collapse_toc_panel is not None:
            result["collapse_toc_panel"] = self.collapse_toc_panel
        if self.hide_column_headers is not None:
            result["hide_column_headers"] = self.hide_column_headers
        return result


@dataclass
class TableDisplay(AnnotationBuilder):
    """Table-display annotation builder.

    Controls table-level display options like row naming and ordering.

    Example:
        >>> td = TableDisplay()
        >>> td.row_name(row_markdown_pattern="{{{Name}}} ({{{Species}}})")
        >>> td.compact(row_order=[SortKey("Name")])
    """
    tag = TAG_TABLE_DISPLAY

    _contexts: dict[str, TableDisplayOptions | str | None] = field(default_factory=dict)

    def set_context(
        self,
        context: str,
        options: TableDisplayOptions | str | None
    ) -> "TableDisplay":
        """Set options for a context."""
        self._contexts[context] = options
        return self

    def row_name(
        self,
        row_markdown_pattern: str,
        template_engine: TemplateEngine | None = None
    ) -> "TableDisplay":
        """Set row name pattern (used in foreign key dropdowns, etc.)."""
        return self.set_context(
            CONTEXT_ROW_NAME,
            TableDisplayOptions(
                row_markdown_pattern=row_markdown_pattern,
                template_engine=template_engine
            )
        )

    def compact(self, options: TableDisplayOptions) -> "TableDisplay":
        """Set options for compact (list) view."""
        return self.set_context(CONTEXT_COMPACT, options)

    def detailed(self, options: TableDisplayOptions) -> "TableDisplay":
        """Set options for detailed (record) view."""
        return self.set_context(CONTEXT_DETAILED, options)

    def default(self, options: TableDisplayOptions) -> "TableDisplay":
        """Set default options."""
        return self.set_context(CONTEXT_DEFAULT, options)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for context, options in self._contexts.items():
            if options is None:
                result[context] = None
            elif isinstance(options, str):
                result[context] = options
            else:
                result[context] = options.to_dict()
        return result


# =============================================================================
# Column Display Annotation
# =============================================================================

@dataclass
class PreFormat:
    """Pre-formatting options for column values.

    Args:
        format: Printf-style format string (e.g., "%.2f")
        bool_true_value: Display value for True
        bool_false_value: Display value for False
    """
    format: str | None = None
    bool_true_value: str | None = None
    bool_false_value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {}
        if self.format is not None:
            result["format"] = self.format
        if self.bool_true_value is not None:
            result["bool_true_value"] = self.bool_true_value
        if self.bool_false_value is not None:
            result["bool_false_value"] = self.bool_false_value
        return result


@dataclass
class ColumnDisplayOptions:
    """Options for displaying a column in a specific context.

    Args:
        pre_format: Pre-formatting options
        markdown_pattern: Template for rendering
        template_engine: Template engine to use
        column_order: Sort order, or False to disable
    """
    pre_format: PreFormat | None = None
    markdown_pattern: str | None = None
    template_engine: TemplateEngine | None = None
    column_order: list[SortKey] | Literal[False] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {}
        if self.pre_format is not None:
            result["pre_format"] = self.pre_format.to_dict()
        if self.markdown_pattern is not None:
            result["markdown_pattern"] = self.markdown_pattern
        if self.template_engine is not None:
            result["template_engine"] = self.template_engine.value
        if self.column_order is not None:
            if self.column_order is False:
                result["column_order"] = False
            else:
                result["column_order"] = [
                    k.to_dict() if isinstance(k, SortKey) else k
                    for k in self.column_order
                ]
        return result


@dataclass
class ColumnDisplay(AnnotationBuilder):
    """Column-display annotation builder.

    Controls how column values are rendered.

    Example:
        >>> cd = ColumnDisplay()
        >>> cd.default(ColumnDisplayOptions(
        ...     pre_format=PreFormat(format="%.2f")
        ... ))
        >>>
        >>> # Markdown link
        >>> cd = ColumnDisplay()
        >>> cd.default(ColumnDisplayOptions(
        ...     markdown_pattern="[Link]({{{_value}}})"
        ... ))
    """
    tag = TAG_COLUMN_DISPLAY

    _contexts: dict[str, ColumnDisplayOptions | str] = field(default_factory=dict)

    def set_context(
        self,
        context: str,
        options: ColumnDisplayOptions | str
    ) -> "ColumnDisplay":
        """Set options for a context."""
        self._contexts[context] = options
        return self

    def default(self, options: ColumnDisplayOptions) -> "ColumnDisplay":
        """Set default options."""
        return self.set_context(CONTEXT_DEFAULT, options)

    def compact(self, options: ColumnDisplayOptions) -> "ColumnDisplay":
        """Set options for compact view."""
        return self.set_context(CONTEXT_COMPACT, options)

    def detailed(self, options: ColumnDisplayOptions) -> "ColumnDisplay":
        """Set options for detailed view."""
        return self.set_context(CONTEXT_DETAILED, options)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for context, options in self._contexts.items():
            if isinstance(options, str):
                result[context] = options
            else:
                result[context] = options.to_dict()
        return result


# =============================================================================
# Facet Entry (for filter context)
# =============================================================================

@dataclass
class FacetRange:
    """A range for facet filtering.

    Args:
        min: Minimum value
        max: Maximum value
        min_exclusive: Exclude min value
        max_exclusive: Exclude max value
    """
    min: float | None = None
    max: float | None = None
    min_exclusive: bool | None = None
    max_exclusive: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {}
        if self.min is not None:
            result["min"] = self.min
        if self.max is not None:
            result["max"] = self.max
        if self.min_exclusive is not None:
            result["min_exclusive"] = self.min_exclusive
        if self.max_exclusive is not None:
            result["max_exclusive"] = self.max_exclusive
        return result


@dataclass
class Facet:
    """A facet definition for filtering.

    Args:
        source: Path to source data
        sourcekey: Reference to named source
        markdown_name: Display name
        comment: Description
        entity: Whether this is an entity facet
        open: Start expanded
        ux_mode: UI mode (choices, ranges, check_presence)
        bar_plot: Show bar plot
        choices: Preset choice values
        ranges: Preset range values
        not_null: Filter to non-null values
        hide_null_choice: Hide "null" option
        hide_not_null_choice: Hide "not null" option
        n_bins: Number of bins for histogram
    """
    source: str | list[str | InboundFK | OutboundFK] | None = None
    sourcekey: str | None = None
    markdown_name: str | None = None
    comment: str | None = None
    entity: bool | None = None
    open: bool | None = None
    ux_mode: FacetUxMode | None = None
    bar_plot: bool | None = None
    choices: list[Any] | None = None
    ranges: list[FacetRange] | None = None
    not_null: bool | None = None
    hide_null_choice: bool | None = None
    hide_not_null_choice: bool | None = None
    n_bins: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {}

        if self.source is not None:
            if isinstance(self.source, str):
                result["source"] = self.source
            else:
                result["source"] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in self.source
                ]

        if self.sourcekey is not None:
            result["sourcekey"] = self.sourcekey
        if self.markdown_name is not None:
            result["markdown_name"] = self.markdown_name
        if self.comment is not None:
            result["comment"] = self.comment
        if self.entity is not None:
            result["entity"] = self.entity
        if self.open is not None:
            result["open"] = self.open
        if self.ux_mode is not None:
            result["ux_mode"] = self.ux_mode.value
        if self.bar_plot is not None:
            result["bar_plot"] = self.bar_plot
        if self.choices is not None:
            result["choices"] = self.choices
        if self.ranges is not None:
            result["ranges"] = [r.to_dict() for r in self.ranges]
        if self.not_null is not None:
            result["not_null"] = self.not_null
        if self.hide_null_choice is not None:
            result["hide_null_choice"] = self.hide_null_choice
        if self.hide_not_null_choice is not None:
            result["hide_not_null_choice"] = self.hide_not_null_choice
        if self.n_bins is not None:
            result["n_bins"] = self.n_bins

        return result


@dataclass
class FacetList:
    """A list of facets for filtering (visible_columns.filter).

    Example:
        >>> facets = FacetList([
        ...     Facet(source="Species", open=True),
        ...     Facet(source="Age", ux_mode=FacetUxMode.RANGES)
        ... ])
    """
    facets: list[Facet] = field(default_factory=list)

    def add(self, facet: Facet) -> "FacetList":
        """Add a facet to the list."""
        self.facets.append(facet)
        return self

    def to_dict(self) -> dict[str, list[dict]]:
        return {"and": [f.to_dict() for f in self.facets]}
