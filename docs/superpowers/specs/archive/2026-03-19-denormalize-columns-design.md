# Design: `denormalize_columns()` and Unified Column Naming

## Problem

1. **No column preview**: Users must run the full data query to discover what columns denormalization will produce. Debugging wrong `include_tables` requires expensive joins.

2. **Inconsistent column naming**: The catalog side uses underscore separators (`Image_Acquisition_Date`) while the bag side uses dots (`Image.Acquisition_Date`). The underscore convention is ambiguous when column names contain underscores.

3. **No schema qualification**: Multi-schema catalogs have no way to distinguish `schema1.Image.Name` from `schema2.Image.Name`.

## Solution

Three changes in one PR:

1. **Add `denormalize_columns()`** — new method on `DatasetLike` returning column names and types without fetching data.
2. **Unify column naming to dot notation** — both catalog and bag use `Table.Column` format.
3. **Schema prefix when needed** — include schema name only when the catalog has multiple domain schemas.

## Column Naming Convention

**Single domain schema** (common case):
```
Image.RID
Image.Filename
Image.Acquisition_Date
Subject.Name
Image_Classification.Image_Class
```

**Multiple domain schemas**:
```
test-schema.Image.RID
test-schema.Image.Filename
other-schema.Sample.Name
```

Detection: check if the set of schemas across `include_tables` has more than one domain schema. If so, prefix all columns with schema.

## API: `denormalize_columns()`

```python
def denormalize_columns(
    self,
    include_tables: list[str],
    **kwargs: Any,
) -> list[tuple[str, str]]:
    """Return the columns that denormalize would produce, without fetching data.

    Performs the same validation as denormalize_as_dataframe (table existence,
    FK path resolution, ambiguity detection) but stops before data queries.

    Args:
        include_tables: List of table names to include.
        **kwargs: Additional arguments (ignored, for protocol compatibility).

    Returns:
        List of (prefixed_column_name, column_type) tuples.
        Column names use dot notation: "Table.Column" (single schema)
        or "Schema.Table.Column" (multi-schema).
        Type strings use ermrest type names (text, int4, float8, etc.).

    Raises:
        DerivaMLException: If tables don't exist or FK paths are ambiguous.
    """
```

### Example Usage

```python
# Preview columns before running expensive query
cols = dataset.denormalize_columns(["Image", "Subject", "Image_Classification"])
for name, dtype in cols:
    print(f"  {name}: {dtype}")

# Output (single schema):
#   Image.RID: ermrest_rid
#   Image.Filename: text
#   Image.URL: text
#   Subject.RID: ermrest_rid
#   Subject.Name: text
#   Image_Classification.Image_Class: text
```

## Implementation

### Column Name Builder (shared helper)

A helper function computes the column prefix given a table name and the set of schemas involved:

```python
def _column_prefix(table_name: str, schema_name: str, multi_schema: bool) -> str:
    if multi_schema:
        return f"{schema_name}.{table_name}"
    return table_name
```

Used by both `denormalize_columns()`, `denormalize_as_dataframe()`, `denormalize_as_dict()`, and `_denormalize_datapath()` to ensure consistent naming.

### Protocol (`interfaces.py`)

Add `denormalize_columns` to `DatasetLike` protocol alongside existing methods.

### Catalog Implementation (`dataset.py`)

1. Call `_prepare_wide_table()` for FK path validation and ambiguity detection
2. Determine if multi-schema (check schemas of all tables in `include_tables`)
3. For each `(table_name, col_name)` in `_prepare_wide_table`'s `denormalized_columns`:
   - Get type from ermrest column: `column.type.typename`
   - Build prefixed name: `Table.Column` or `Schema.Table.Column`
4. Return ordered by table appearance in `include_tables`, then column order within table

Also update `_denormalize_datapath()` to use dot notation instead of underscore.

### Bag Implementation (`dataset_bag.py`)

1. Call `_prepare_wide_table()` for path validation (returns `denormalized_columns`)
2. Determine if multi-schema
3. For each `(table_name, col_name)`:
   - Get type from SQLAlchemy column, mapped to ermrest type names
   - Build prefixed name with same convention

Also update `_denormalize()` SQL labels to use same naming convention.

### SQLAlchemy → ERMrest Type Mapping

The bag side needs to map SQLAlchemy types to ermrest names for consistency:

| SQLAlchemy | ermrest |
|-----------|---------|
| `String`/`Text` | `text` |
| `Integer` | `int4` |
| `BigInteger` | `int8` |
| `SmallInteger` | `int2` |
| `Float`/`Numeric` | `float8` |
| `Boolean` | `boolean` |
| `Date` | `date` |
| `DateTime` | `timestamptz` |
| `LargeBinary` | `bytea` |

Fallback: `str(col.type).lower()` for unmapped types.

### Column Ordering

Columns are ordered by:
1. Table appearance order in `include_tables` (preserves user intent)
2. Within each table, column definition order from the schema

Both implementations iterate `include_tables` in order (not through `_prepare_wide_table`'s set conversion) to guarantee deterministic ordering.

### split_dataset Fix (`split.py`)

Update `split_dataset` to use dot notation:

- Line 771: `f"{element_table}_RID"` → `f"{element_table}.RID"`
- `stratify_by_column` parameter: document new format `Image_Classification.Image_Class`
- Add backward compatibility: if column not found with dot, try underscore fallback with deprecation warning

### Association Table Filtering

Both implementations skip association tables in the column output, matching `_prepare_wide_table`'s existing filter (line 732: `if not self.is_association(table_name)`).

## Cost

Near zero for `denormalize_columns()`. The naming convention change is a one-line change per column formatting site.

## Breaking Changes

- `denormalize_as_dataframe()` column names change from `Table_Column` to `Table.Column` (catalog side)
- `stratify_by_column` parameter format changes from `Table_Column` to `Table.Column`
- Backward compatibility: `split_dataset` falls back to underscore format with deprecation warning

## Files Modified

| File | Change |
|------|--------|
| `src/deriva_ml/interfaces.py` | Add `denormalize_columns` to `DatasetLike` protocol |
| `src/deriva_ml/dataset/dataset.py` | Implement `denormalize_columns`; switch to dot notation |
| `src/deriva_ml/dataset/dataset_bag.py` | Implement `denormalize_columns`; type mapping |
| `src/deriva_ml/dataset/split.py` | Update RID column lookup and stratify_by_column |
| `src/deriva_ml/model/catalog.py` | Add `_column_prefix` helper |
| `tests/dataset/test_denormalize.py` | Add column preview tests |
| `tests/dataset/test_split.py` | Update column name expectations |
| `docs/user-guide/denormalization.md` | New section on denormalized tables |

## Testing

1. Catalog `denormalize_columns`: verify names and types match actual `denormalize_as_dataframe().columns`
2. Bag `denormalize_columns`: verify names and types match actual `denormalize_as_dataframe().columns`
3. Cross-implementation: verify catalog and bag return same column names (modulo schema prefix) for same schema
4. Ambiguity detection: verify same errors as full denormalize
5. Multi-schema: verify schema prefix appears only when needed
6. split_dataset: verify stratified split works with new dot notation
7. split_dataset: verify backward compat with underscore notation (deprecation warning)
