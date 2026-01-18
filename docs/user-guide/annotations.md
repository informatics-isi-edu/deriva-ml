# Catalog Annotations

Deriva catalogs use **annotations** to control how data is displayed in the Chaise web interface. Annotations configure things like:

- Display names for tables and columns
- Which columns appear in list vs. detail views
- How related tables are shown
- Row ordering and pagination
- Custom formatting and markdown templates
- Faceted search configuration

DerivaML provides **annotation builder classes** that make it easier to create and manage annotations with IDE autocompletion, type safety, and validation.

## Quick Start

```python
from deriva_ml.model import (
    TableHandle, Display, VisibleColumns, TableDisplay,
    PseudoColumn, OutboundFK, fk_constraint, SortKey,
    CONTEXT_COMPACT, CONTEXT_DETAILED
)

# Get a table handle
table = ml.model.name_to_table("Subject")
handle = TableHandle(table)

# Set display name and description
handle.set_annotation(Display(
    name="Research Subjects",
    comment="Individuals enrolled in the study"
))

# Configure visible columns
vc = VisibleColumns()
vc.compact(["RID", "Name", "Species", "Age"])
vc.detailed(["RID", "Name", "Species", "Age", "Enrollment_Date", "Notes"])
handle.set_annotation(vc)

# Set row name pattern (how rows appear in dropdowns)
td = TableDisplay()
td.row_name("{{{Name}}} ({{{Species}}})")
handle.set_annotation(td)
```

## Understanding Contexts

Annotations often apply to specific **contexts** - different places where data appears in the UI:

| Context | Description | Example Use |
|---------|-------------|-------------|
| `*` | Default for all contexts | Fallback settings |
| `compact` | Table/list views | Search results, data browser |
| `compact/brief` | Abbreviated previews | Tooltips, inline references |
| `compact/select` | Selection dropdowns | Foreign key pickers |
| `detailed` | Full record view | Single record page |
| `entry` | Create/edit forms | Data entry |
| `entry/create` | Create form only | New record form |
| `entry/edit` | Edit form only | Existing record form |
| `filter` | Faceted search | Search sidebar |

DerivaML provides constants for common contexts:

```python
from deriva_ml.model import (
    CONTEXT_DEFAULT,    # "*"
    CONTEXT_COMPACT,    # "compact"
    CONTEXT_DETAILED,   # "detailed"
    CONTEXT_ENTRY,      # "entry"
    CONTEXT_FILTER,     # "filter"
)
```

## Annotation Builders

### Display Annotation

Controls basic display properties for tables and columns.

```python
from deriva_ml.model import Display, NameStyle

# Simple display name
display = Display(name="Friendly Name")

# With markdown name (mutually exclusive with name)
display = Display(markdown_name="**Bold** Name")

# With description/tooltip
display = Display(
    name="Subjects",
    comment="Research subjects enrolled in the study"
)

# With name styling
display = Display(
    name_style=NameStyle(
        underline_space=True,  # Convert underscores to spaces
        title_case=True        # Apply title case
    )
)

# Context-specific options
display = Display(
    name="Value",
    show_null={
        CONTEXT_COMPACT: False,      # Hide nulls in lists
        CONTEXT_DETAILED: '"N/A"'    # Show "N/A" in detail view
    }
)

# Apply to table
handle.set_annotation(display)
```

### Visible Columns

Controls which columns appear in different UI contexts.

```python
from deriva_ml.model import VisibleColumns, PseudoColumn, fk_constraint

vc = VisibleColumns()

# Simple column lists
vc.compact(["RID", "Name", "Status"])
vc.detailed(["RID", "Name", "Status", "Description", "Created"])
vc.entry(["Name", "Status", "Description"])

# Include foreign key references (shows related data)
vc.compact([
    "RID",
    "Name",
    fk_constraint("domain", "Subject_Species_fkey"),  # FK reference
])

# Include pseudo-columns (computed/derived values)
vc.detailed([
    "RID",
    "Name",
    PseudoColumn(
        source="Description",
        markdown_name="Notes"  # Custom display name
    ),
])

# Set default for all contexts
vc.default(["RID", "Name"])

# Reference another context (inherit its settings)
vc.set_context("compact/brief", "compact")

handle.set_annotation(vc)
```

### Visible Foreign Keys

Controls which related tables appear in the detail view.

```python
from deriva_ml.model import VisibleForeignKeys, fk_constraint, PseudoColumn

vfk = VisibleForeignKeys()

# Show inbound foreign keys (tables that reference this one)
vfk.detailed([
    fk_constraint("domain", "Image_Subject_fkey"),
    fk_constraint("domain", "Diagnosis_Subject_fkey"),
])

# Set for all contexts
vfk.default([fk_constraint("domain", "Sample_Subject_fkey")])

handle.set_annotation(vfk)
```

### Table Display

Controls table-level display options like row naming and ordering.

```python
from deriva_ml.model import TableDisplay, TableDisplayOptions, SortKey, TemplateEngine

td = TableDisplay()

# Row name pattern (used in dropdowns, references)
td.row_name("{{{Name}}}")

# More complex pattern with multiple columns
td.row_name("{{{Name}}} - {{{RID}}}")

# With explicit template engine
td.row_name(
    "{{{Name}}} ({{{Species}}})",
    template_engine=TemplateEngine.HANDLEBARS
)

# Configure compact view options
td.compact(TableDisplayOptions(
    row_order=[
        SortKey("Name"),                      # Ascending
        SortKey("Created", descending=True),  # Descending
    ],
    page_size=50
))

# Configure detailed view options
td.detailed(TableDisplayOptions(
    collapse_toc_panel=True,
    hide_column_headers=False
))

handle.set_annotation(td)
```

### Column Display

Controls how column values are rendered.

```python
from deriva_ml.model import ColumnDisplay, ColumnDisplayOptions, PreFormat

cd = ColumnDisplay()

# Number formatting
cd.default(ColumnDisplayOptions(
    pre_format=PreFormat(format="%.2f")  # Two decimal places
))

# Boolean formatting
cd.default(ColumnDisplayOptions(
    pre_format=PreFormat(
        bool_true_value="Yes",
        bool_false_value="No"
    )
))

# Markdown pattern (make URLs clickable)
cd.default(ColumnDisplayOptions(
    markdown_pattern="[Link]({{{_value}}})"
))

# Context-specific formatting
cd.compact(ColumnDisplayOptions(
    markdown_pattern="[{{{_value}}}]({{{_value}}})"
))
cd.detailed(ColumnDisplayOptions(
    markdown_pattern="**URL**: [{{{_value}}}]({{{_value}}})"
))

# Apply to a column
col_handle = handle.column("URL")
col_handle.annotations[ColumnDisplay.tag] = cd.to_dict()
col_handle.apply()
```

## Pseudo-Columns

Pseudo-columns display computed values, values from related tables, or custom markdown patterns.

### Basic Pseudo-Column

```python
from deriva_ml.model import PseudoColumn

# Simple column with custom name
pc = PseudoColumn(
    source="Internal_ID",
    markdown_name="ID"
)
```

### Foreign Key Traversal

Display values from related tables by traversing foreign keys.

```python
from deriva_ml.model import PseudoColumn, OutboundFK, InboundFK

# Outbound: Follow FK from this table to another
# Example: Image -> Subject (get Subject name)
pc = PseudoColumn(
    source=[
        OutboundFK("domain", "Image_Subject_fkey"),
        "Name"  # Column in the referenced table
    ],
    markdown_name="Subject Name"
)

# Inbound: Follow FK from another table to this one
# Example: Subject <- Images (count images)
pc = PseudoColumn(
    source=[
        InboundFK("domain", "Image_Subject_fkey"),
        "RID"
    ],
    aggregate=Aggregate.CNT,  # Count the related records
    markdown_name="Image Count"
)

# Multi-hop: Chain multiple FKs
# Example: Image -> Subject -> Species
pc = PseudoColumn(
    source=[
        OutboundFK("domain", "Image_Subject_fkey"),
        OutboundFK("domain", "Subject_Species_fkey"),
        "Name"
    ],
    markdown_name="Species"
)
```

### Aggregates

Aggregate values from related tables.

```python
from deriva_ml.model import PseudoColumn, InboundFK, Aggregate

# Count related records
pc = PseudoColumn(
    source=[InboundFK("domain", "Sample_Subject_fkey"), "RID"],
    aggregate=Aggregate.CNT,
    markdown_name="Sample Count"
)

# Count distinct values
pc = PseudoColumn(
    source=[InboundFK("domain", "Sample_Subject_fkey"), "Type"],
    aggregate=Aggregate.CNT_D,
    markdown_name="Unique Types"
)

# Min/max values
pc = PseudoColumn(
    source=[InboundFK("domain", "Measurement_Subject_fkey"), "Value"],
    aggregate=Aggregate.MAX,
    markdown_name="Max Value"
)

# Array of values
pc = PseudoColumn(
    source=[InboundFK("domain", "Tag_Subject_fkey"), "Name"],
    aggregate=Aggregate.ARRAY,
    markdown_name="Tags"
)
```

### Display Options

```python
from deriva_ml.model import PseudoColumn, PseudoColumnDisplay, ArrayUxMode

pc = PseudoColumn(
    source="URL",
    display=PseudoColumnDisplay(
        markdown_pattern="[Download]({{{_value}}})",
        show_foreign_key_link=False
    )
)

# Array display options
pc = PseudoColumn(
    source=[InboundFK("domain", "Tag_Subject_fkey"), "Name"],
    aggregate=Aggregate.ARRAY,
    display=PseudoColumnDisplay(
        array_ux_mode=ArrayUxMode.CSV  # Show as comma-separated
    )
)
```

## Faceted Search

Configure the filter panel in the Chaise data browser.

```python
from deriva_ml.model import (
    VisibleColumns, Facet, FacetList, FacetRange,
    FacetUxMode, OutboundFK
)

# Create facet list
facets = FacetList()

# Simple choice facet
facets.add(Facet(
    source="Status",
    open=True,  # Start expanded
    markdown_name="Status"
))

# FK-based facet
facets.add(Facet(
    source=[OutboundFK("domain", "Subject_Species_fkey"), "Name"],
    markdown_name="Species",
    open=True
))

# Range facet for numeric values
facets.add(Facet(
    source="Age",
    ux_mode=FacetUxMode.RANGES,
    ranges=[
        FacetRange(min=0, max=18),
        FacetRange(min=18, max=65),
        FacetRange(min=65)  # 65+
    ],
    markdown_name="Age Group"
))

# Facet with preset choices
facets.add(Facet(
    source="Priority",
    ux_mode=FacetUxMode.CHOICES,
    choices=["High", "Medium", "Low"],
    hide_null_choice=True
))

# Check presence facet (has value / no value)
facets.add(Facet(
    source="Notes",
    ux_mode=FacetUxMode.CHECK_PRESENCE,
    markdown_name="Has Notes"
))

# Apply to visible columns
vc = VisibleColumns()
vc.compact(["RID", "Name", "Status"])
vc._contexts["filter"] = facets.to_dict()

handle.set_annotation(vc)
```

## Handlebars Templates

Many annotations support Handlebars templates for custom formatting.

### Available Variables

```python
# Get template variables for a table
vars = ml.get_handlebars_template_variables("Subject")

# Direct column access
# {{{ColumnName}}}

# Row context (all columns)
# {{{_row.ColumnName}}}

# Current value (in column_display)
# {{{_value}}}

# Foreign key values
# {{{$fkeys.schema.constraint_name.values.ColumnName}}}
# {{{$fkeys.schema.constraint_name.rowName}}}

# Catalog info
# {{{$catalog.id}}}
# {{{$catalog.snapshot}}}
```

### Template Examples

```python
# Row name with multiple fields
"{{{Name}}} ({{{Species}}})"

# Conditional display
"{{#if Notes}}{{{Notes}}}{{else}}No notes{{/if}}"

# URL link
"[{{{Filename}}}]({{{URL}}})"

# Image preview
"[![{{{Filename}}}]({{{URL}}})]({{{URL}}})"

# Formatted date
"{{formatDate RCT 'YYYY-MM-DD'}}"

# FK value
"{{{$fkeys.domain.Subject_Species_fkey.values.Name}}}"
```

## Complete Example

Here's a complete example configuring annotations for a Subject table:

```python
from deriva_ml import DerivaML
from deriva_ml.model import (
    TableHandle, Display, VisibleColumns, VisibleForeignKeys,
    TableDisplay, TableDisplayOptions, ColumnDisplay, ColumnDisplayOptions,
    PseudoColumn, OutboundFK, InboundFK, Facet, FacetList,
    fk_constraint, SortKey, Aggregate, FacetUxMode, PreFormat,
    CONTEXT_COMPACT, CONTEXT_DETAILED
)

# Connect to catalog
ml = DerivaML(hostname="example.org", catalog_id="1")

# Get table handle
table = ml.model.name_to_table("Subject")
handle = TableHandle(table)

# 1. Display annotation - friendly name and description
handle.set_annotation(Display(
    name="Research Subjects",
    comment="Individuals enrolled in research studies"
))

# 2. Table display - row naming and ordering
td = TableDisplay()
td.row_name("{{{Name}}} ({{{Subject_ID}}})")
td.compact(TableDisplayOptions(
    row_order=[SortKey("Name")],
    page_size=25
))
handle.set_annotation(td)

# 3. Visible columns - what appears in each view
vc = VisibleColumns()

# Compact view - essential columns
vc.compact([
    "RID",
    "Subject_ID",
    "Name",
    fk_constraint("domain", "Subject_Species_fkey"),
    "Age",
    PseudoColumn(
        source=[InboundFK("domain", "Sample_Subject_fkey"), "RID"],
        aggregate=Aggregate.CNT,
        markdown_name="Samples"
    ),
])

# Detailed view - all columns plus computed fields
vc.detailed([
    "RID",
    "Subject_ID",
    "Name",
    fk_constraint("domain", "Subject_Species_fkey"),
    "Age",
    "Sex",
    "Enrollment_Date",
    "Notes",
    PseudoColumn(
        source=[InboundFK("domain", "Sample_Subject_fkey"), "RID"],
        aggregate=Aggregate.CNT,
        markdown_name="Sample Count"
    ),
])

# Entry view - editable fields
vc.entry(["Subject_ID", "Name", "Species", "Age", "Sex", "Notes"])

# Filter configuration
facets = FacetList()
facets.add(Facet(
    source=[OutboundFK("domain", "Subject_Species_fkey"), "Name"],
    markdown_name="Species",
    open=True
))
facets.add(Facet(
    source="Sex",
    open=True
))
facets.add(Facet(
    source="Age",
    ux_mode=FacetUxMode.RANGES
))
vc._contexts["filter"] = facets.to_dict()

handle.set_annotation(vc)

# 4. Visible foreign keys - related tables in detail view
vfk = VisibleForeignKeys()
vfk.detailed([
    fk_constraint("domain", "Sample_Subject_fkey"),
    fk_constraint("domain", "Diagnosis_Subject_fkey"),
    fk_constraint("domain", "Image_Subject_fkey"),
])
handle.set_annotation(vfk)

# 5. Column-specific annotations
age_col = handle.column("Age")
age_col.description = "Age in years at enrollment"
age_col.set_display_name("Age (years)")

notes_col = handle.column("Notes")
notes_col.description = "Additional observations or comments"

print("Annotations configured successfully!")
```

## Using Raw Dictionaries

The annotation builders are optional. You can always use raw dictionaries for complex cases or when the builders don't cover your needs:

```python
# Raw dictionary approach
table.annotations["tag:isrd.isi.edu,2016:visible-columns"] = {
    "compact": ["RID", "Name", "Description"],
    "detailed": ["RID", "Name", "Description", "Created"],
    "filter": {
        "and": [
            {"source": "Name", "open": True},
            {"source": "Status"}
        ]
    }
}
table.apply()
```

## Best Practices

1. **Always provide descriptions** - Use `comment` for tables and columns to make the catalog self-documenting.

2. **Set row name patterns** - Makes foreign key dropdowns and references readable.

3. **Configure visible columns** - Show relevant columns in compact view, more detail in detailed view.

4. **Use faceted search** - Configure filters for commonly-searched fields.

5. **Order columns logically** - Put identifiers first, then key attributes, then metadata.

6. **Test in Chaise** - Use `ml.get_chaise_url("TableName")` to view your changes.

## API Reference

For complete API documentation, see:

- [`deriva_ml.model.annotations`](../code-docs/annotations.md) - Annotation builder classes
- [`TableHandle`](../code-docs/handles.md) - Table wrapper with annotation methods
- [`ColumnHandle`](../code-docs/handles.md) - Column wrapper with annotation methods
