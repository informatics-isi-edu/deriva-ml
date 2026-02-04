"""Schema validation for DerivaML catalogs.

This module provides functionality to validate that a catalog's ML schema matches
the expected structure created by create_schema.py. It can check for:
- Required tables and their columns
- Required vocabulary tables and their initial terms
- Foreign key relationships
- Extra tables/columns (in strict mode)

Usage:
    from deriva_ml import DerivaML

    ml = DerivaML('localhost', 'my_catalog')
    report = ml.validate_schema(strict=False)
    if not report.is_valid:
        print(report.to_text())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML

from deriva_ml.core.definitions import MLTable, MLVocab


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Schema is invalid, will cause failures
    WARNING = "warning"  # Schema may work but has issues
    INFO = "info"  # Informational (e.g., extra items in non-strict mode)


@dataclass
class ValidationIssue:
    """A single validation issue found during schema inspection."""

    severity: ValidationSeverity
    category: str  # e.g., "table", "column", "vocabulary", "foreign_key"
    message: str
    table: str | None = None
    column: str | None = None
    expected: Any = None
    actual: Any = None

    def __str__(self) -> str:
        location = ""
        if self.table:
            location = f"[{self.table}"
            if self.column:
                location += f".{self.column}"
            location += "] "
        return f"{self.severity.value.upper()}: {location}{self.message}"


@dataclass
class SchemaValidationReport:
    """Complete validation report for a DerivaML catalog schema."""

    schema_name: str
    strict_mode: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Returns True if no errors were found."""
        return not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Returns only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Returns only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def info(self) -> list[ValidationIssue]:
        """Returns only info-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def add_error(self, category: str, message: str, **kwargs) -> None:
        """Add an error-level issue."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category=category,
            message=message,
            **kwargs
        ))

    def add_warning(self, category: str, message: str, **kwargs) -> None:
        """Add a warning-level issue."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category=category,
            message=message,
            **kwargs
        ))

    def add_info(self, category: str, message: str, **kwargs) -> None:
        """Add an info-level issue."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            category=category,
            message=message,
            **kwargs
        ))

    def to_text(self) -> str:
        """Generate a human-readable text report."""
        lines = [
            f"Schema Validation Report for '{self.schema_name}'",
            f"Mode: {'Strict' if self.strict_mode else 'Non-strict'}",
            f"Status: {'VALID' if self.is_valid else 'INVALID'}",
            "",
        ]

        if not self.issues:
            lines.append("No issues found.")
            return "\n".join(lines)

        # Summary
        lines.append(f"Summary: {len(self.errors)} errors, {len(self.warnings)} warnings, {len(self.info)} info")
        lines.append("")

        # Group by category
        by_category: dict[str, list[ValidationIssue]] = {}
        for issue in self.issues:
            by_category.setdefault(issue.category, []).append(issue)

        for category, category_issues in sorted(by_category.items()):
            lines.append(f"## {category.replace('_', ' ').title()}")
            for issue in category_issues:
                lines.append(f"  - {issue}")
                if issue.expected is not None or issue.actual is not None:
                    if issue.expected is not None:
                        lines.append(f"      Expected: {issue.expected}")
                    if issue.actual is not None:
                        lines.append(f"      Actual: {issue.actual}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format for JSON serialization.

        Returns:
            Dictionary with schema validation results suitable for JSON encoding.
            Structure:
            {
                "schema_name": str,
                "strict_mode": bool,
                "is_valid": bool,
                "summary": {"errors": int, "warnings": int, "info": int},
                "issues": [
                    {
                        "severity": "error"|"warning"|"info",
                        "category": str,
                        "message": str,
                        "table": str|null,
                        "column": str|null,
                        "expected": any|null,
                        "actual": any|null
                    },
                    ...
                ]
            }
        """
        return {
            "schema_name": self.schema_name,
            "strict_mode": self.strict_mode,
            "is_valid": self.is_valid,
            "summary": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.info),
            },
            "issues": [
                {
                    "severity": i.severity.value,
                    "category": i.category,
                    "message": i.message,
                    "table": i.table,
                    "column": i.column,
                    "expected": i.expected,
                    "actual": i.actual,
                }
                for i in self.issues
            ],
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Convert report to JSON string format.

        Args:
            indent: Number of spaces for indentation, or None for compact output.

        Returns:
            JSON string representation of the validation report.

        Example:
            >>> report = ml.validate_schema()
            >>> print(report.to_json())
            {
              "schema_name": "deriva-ml",
              "strict_mode": false,
              "is_valid": true,
              ...
            }
        """
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# Expected Schema Structure
# =============================================================================

# Expected columns for each table in the ML schema
# Format: {table_name: {column_name: (type_name, nullok)}}
EXPECTED_TABLE_COLUMNS: dict[str, dict[str, tuple[str, bool]]] = {
    MLTable.dataset: {
        "Description": ("markdown", True),
        "Deleted": ("boolean", True),
        "Version": ("text", True),  # FK column
    },
    MLTable.dataset_version: {
        "Version": ("text", True),
        "Description": ("markdown", True),
        "Dataset": ("text", True),
        "Execution": ("text", True),
        "Minid": ("text", True),
        "Snapshot": ("text", True),
    },
    MLTable.workflow: {
        "Name": ("text", True),
        "Description": ("markdown", True),
        "URL": ("ermrest_uri", True),
        "Checksum": ("text", True),
        "Version": ("text", True),
        "Workflow_Type": ("text", True),  # FK column
    },
    MLTable.execution: {
        "Workflow": ("text", True),
        "Description": ("markdown", True),
        "Duration": ("text", True),
        "Status": ("text", True),
        "Status_Detail": ("text", True),
    },
    MLTable.execution_metadata: {
        # Asset table columns
        "URL": ("text", False),
        "Filename": ("text", True),
        "Description": ("markdown", True),
        "Length": ("int8", False),
        "MD5": ("text", False),
    },
    MLTable.execution_asset: {
        # Asset table columns
        "URL": ("text", False),
        "Filename": ("text", True),
        "Description": ("markdown", True),
        "Length": ("int8", False),
        "MD5": ("text", False),
    },
    MLTable.file: {
        # Asset table columns
        "URL": ("text", False),
        "Filename": ("text", True),
        "Description": ("markdown", True),
        "Length": ("int8", False),
        "MD5": ("text", False),
    },
}

# Expected vocabulary tables
EXPECTED_VOCABULARY_TABLES: list[str] = [
    MLVocab.dataset_type,
    MLVocab.workflow_type,
    MLVocab.asset_type,
    MLVocab.asset_role,
    MLVocab.feature_name,
]

# Expected vocabulary columns (all vocab tables have these)
EXPECTED_VOCABULARY_COLUMNS: dict[str, tuple[str, bool]] = {
    "Name": ("text", False),
    "Description": ("markdown", False),
    "Synonyms": ("text[]", True),
    "ID": ("ermrest_curie", False),
    "URI": ("ermrest_uri", False),
}

# Expected initial terms for each vocabulary table
# Format: {vocab_table: [term_names]}
EXPECTED_VOCABULARY_TERMS: dict[str, list[str]] = {
    MLVocab.asset_type: [
        "Execution_Config",
        "Runtime_Env",
        "Hydra_Config",
        "Deriva_Config",
        "Execution_Metadata",
        "Execution_Asset",
        "File",
        "Input_File",
        "Output_File",
        "Model_File",
        "Notebook_Output",
    ],
    MLVocab.asset_role: [
        "Input",
        "Output",
    ],
    MLVocab.dataset_type: [
        "File",
    ],
}

# Expected association tables
EXPECTED_ASSOCIATION_TABLES: list[str] = [
    "Dataset_Dataset_Type",
    "Dataset_Dataset",  # Nested datasets
    "Dataset_Execution",
    "Dataset_File",
    "Execution_Execution",  # Nested executions
    "Execution_Metadata_Asset_Type",
    "Execution_Metadata_Execution",
    "Execution_Asset_Asset_Type",
    "Execution_Asset_Execution",
    "File_Asset_Type",
    "File_Execution",
]

# System columns present in all tables
SYSTEM_COLUMNS = {"RID", "RCT", "RMT", "RCB", "RMB"}


class SchemaValidator:
    """Validates a DerivaML catalog schema against expected structure."""

    def __init__(self, ml: "DerivaML"):
        """Initialize the validator.

        Args:
            ml: DerivaML instance connected to the catalog to validate.
        """
        self.ml = ml
        self.model = ml.model
        self.ml_schema_name = ml.ml_schema

    def validate(self, strict: bool = False) -> SchemaValidationReport:
        """Validate the ML schema structure.

        Args:
            strict: If True, report extra tables/columns as errors.
                   If False, report them as info only.

        Returns:
            SchemaValidationReport with all validation results.
        """
        report = SchemaValidationReport(
            schema_name=self.ml_schema_name,
            strict_mode=strict,
        )

        # Check that ML schema exists
        if self.ml_schema_name not in self.model.model.schemas:
            report.add_error(
                "schema",
                f"ML schema '{self.ml_schema_name}' does not exist",
            )
            return report

        schema = self.model.model.schemas[self.ml_schema_name]

        # Validate core tables
        self._validate_core_tables(schema, report, strict)

        # Validate vocabulary tables
        self._validate_vocabulary_tables(schema, report, strict)

        # Validate association tables
        self._validate_association_tables(schema, report, strict)

        # Validate vocabulary terms
        self._validate_vocabulary_terms(report)

        # Check for extra tables (in strict mode)
        if strict:
            self._check_extra_tables(schema, report)

        return report

    def _validate_core_tables(
        self,
        schema,
        report: SchemaValidationReport,
        strict: bool,
    ) -> None:
        """Validate that all core tables exist with required columns."""
        for table_name, expected_columns in EXPECTED_TABLE_COLUMNS.items():
            if table_name not in schema.tables:
                report.add_error(
                    "table",
                    f"Missing required table '{table_name}'",
                    table=table_name,
                )
                continue

            table = schema.tables[table_name]
            self._validate_table_columns(
                table, table_name, expected_columns, report, strict
            )

    def _validate_vocabulary_tables(
        self,
        schema,
        report: SchemaValidationReport,
        strict: bool,
    ) -> None:
        """Validate that all vocabulary tables exist with required columns."""
        for table_name in EXPECTED_VOCABULARY_TABLES:
            if table_name not in schema.tables:
                report.add_error(
                    "vocabulary_table",
                    f"Missing required vocabulary table '{table_name}'",
                    table=table_name,
                )
                continue

            table = schema.tables[table_name]
            self._validate_table_columns(
                table, table_name, EXPECTED_VOCABULARY_COLUMNS, report, strict
            )

    def _validate_association_tables(
        self,
        schema,
        report: SchemaValidationReport,
        strict: bool,
    ) -> None:
        """Validate that all association tables exist."""
        for table_name in EXPECTED_ASSOCIATION_TABLES:
            if table_name not in schema.tables:
                report.add_error(
                    "association_table",
                    f"Missing required association table '{table_name}'",
                    table=table_name,
                )

    def _validate_table_columns(
        self,
        table,
        table_name: str,
        expected_columns: dict[str, tuple[str, bool]],
        report: SchemaValidationReport,
        strict: bool,
    ) -> None:
        """Validate columns of a specific table."""
        actual_columns = {col.name: col for col in table.columns}

        # Check for missing columns
        for col_name, (expected_type, expected_nullok) in expected_columns.items():
            if col_name not in actual_columns:
                report.add_error(
                    "column",
                    f"Missing required column '{col_name}'",
                    table=table_name,
                    column=col_name,
                )
                continue

            col = actual_columns[col_name]
            actual_type = col.type.typename

            # Check type (allow some flexibility for domain types)
            if not self._types_compatible(expected_type, actual_type):
                report.add_warning(
                    "column_type",
                    f"Column '{col_name}' has unexpected type",
                    table=table_name,
                    column=col_name,
                    expected=expected_type,
                    actual=actual_type,
                )

            # Check nullok (only warn, don't error)
            if col.nullok != expected_nullok:
                report.add_info(
                    "column_nullok",
                    f"Column '{col_name}' has different nullok setting",
                    table=table_name,
                    column=col_name,
                    expected=expected_nullok,
                    actual=col.nullok,
                )

        # Check for extra columns in strict mode
        if strict:
            expected_col_names = set(expected_columns.keys()) | SYSTEM_COLUMNS
            for col_name in actual_columns:
                if col_name not in expected_col_names:
                    report.add_error(
                        "extra_column",
                        f"Unexpected column '{col_name}' (strict mode)",
                        table=table_name,
                        column=col_name,
                    )

    def _types_compatible(self, expected: str, actual: str) -> bool:
        """Check if two type names are compatible.

        Allows for domain type variations (e.g., markdown is text-based).
        """
        if expected == actual:
            return True

        # Handle domain types that are based on text
        text_based = {"text", "markdown", "longtext", "ermrest_curie", "ermrest_uri"}
        if expected in text_based and actual in text_based:
            return True

        # Handle timestamp variations
        timestamp_types = {"timestamp", "timestamptz", "ermrest_rct", "ermrest_rmt"}
        if expected in timestamp_types and actual in timestamp_types:
            return True

        return False

    def _validate_vocabulary_terms(self, report: SchemaValidationReport) -> None:
        """Validate that required vocabulary terms exist."""
        for vocab_table, expected_terms in EXPECTED_VOCABULARY_TERMS.items():
            try:
                actual_terms = self.ml.list_vocabulary_terms(vocab_table)
                actual_term_names = {term.name for term in actual_terms}

                for term_name in expected_terms:
                    if term_name not in actual_term_names:
                        report.add_error(
                            "vocabulary_term",
                            f"Missing required term '{term_name}'",
                            table=vocab_table,
                            expected=term_name,
                        )
            except Exception as e:
                report.add_error(
                    "vocabulary_term",
                    f"Could not validate terms: {e}",
                    table=vocab_table,
                )

    def _check_extra_tables(
        self,
        schema,
        report: SchemaValidationReport,
    ) -> None:
        """Check for extra tables not in the expected schema."""
        expected_tables = (
            set(EXPECTED_TABLE_COLUMNS.keys())
            | set(EXPECTED_VOCABULARY_TABLES)
            | set(EXPECTED_ASSOCIATION_TABLES)
        )

        for table_name in schema.tables:
            if table_name not in expected_tables:
                report.add_error(
                    "extra_table",
                    f"Unexpected table '{table_name}' (strict mode)",
                    table=table_name,
                )


def validate_ml_schema(ml: "DerivaML", strict: bool = False) -> SchemaValidationReport:
    """Validate the ML schema of a DerivaML catalog.

    This is a convenience function that creates a validator and runs validation.

    Args:
        ml: DerivaML instance connected to the catalog to validate.
        strict: If True, report extra tables/columns as errors.
               If False, report them as info only.

    Returns:
        SchemaValidationReport with all validation results.

    Example:
        >>> from deriva_ml import DerivaML
        >>> ml = DerivaML('localhost', 'my_catalog')
        >>> report = validate_ml_schema(ml, strict=False)
        >>> if not report.is_valid:
        ...     print(report.to_text())
    """
    validator = SchemaValidator(ml)
    return validator.validate(strict=strict)
