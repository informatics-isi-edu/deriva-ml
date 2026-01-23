"""Tests for the schema validation module."""

from __future__ import annotations

import pytest

from deriva_ml.schema.validation import (
    EXPECTED_ASSOCIATION_TABLES,
    EXPECTED_TABLE_COLUMNS,
    EXPECTED_VOCABULARY_TABLES,
    EXPECTED_VOCABULARY_TERMS,
    SchemaValidationReport,
    SchemaValidator,
    ValidationIssue,
    ValidationSeverity,
    validate_ml_schema,
)


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_str_with_table_and_column(self):
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category="column",
            message="Missing required column",
            table="Dataset",
            column="Description",
        )
        assert "[Dataset.Description]" in str(issue)
        assert "ERROR" in str(issue)
        assert "Missing required column" in str(issue)

    def test_str_with_table_only(self):
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category="table",
            message="Table has issues",
            table="Dataset",
        )
        assert "[Dataset]" in str(issue)
        assert "WARNING" in str(issue)

    def test_str_without_location(self):
        issue = ValidationIssue(
            severity=ValidationSeverity.INFO,
            category="schema",
            message="Schema info message",
        )
        assert "INFO" in str(issue)
        assert "[" not in str(issue)


class TestSchemaValidationReport:
    """Tests for SchemaValidationReport dataclass."""

    def test_empty_report_is_valid(self):
        report = SchemaValidationReport(schema_name="deriva-ml", strict_mode=False)
        assert report.is_valid
        assert len(report.errors) == 0
        assert len(report.warnings) == 0
        assert len(report.info) == 0

    def test_report_with_error_is_invalid(self):
        report = SchemaValidationReport(schema_name="deriva-ml", strict_mode=False)
        report.add_error("table", "Missing table", table="Dataset")
        assert not report.is_valid
        assert len(report.errors) == 1

    def test_report_with_only_warnings_is_valid(self):
        report = SchemaValidationReport(schema_name="deriva-ml", strict_mode=False)
        report.add_warning("column_type", "Type mismatch", table="Dataset", column="Desc")
        assert report.is_valid
        assert len(report.warnings) == 1

    def test_add_methods(self):
        report = SchemaValidationReport(schema_name="test", strict_mode=True)
        report.add_error("cat1", "error msg", table="T1")
        report.add_warning("cat2", "warning msg", column="C1")
        report.add_info("cat3", "info msg", expected="a", actual="b")

        assert len(report.issues) == 3
        assert report.issues[0].severity == ValidationSeverity.ERROR
        assert report.issues[1].severity == ValidationSeverity.WARNING
        assert report.issues[2].severity == ValidationSeverity.INFO

    def test_to_text_empty(self):
        report = SchemaValidationReport(schema_name="deriva-ml", strict_mode=False)
        text = report.to_text()
        assert "VALID" in text
        assert "No issues found" in text

    def test_to_text_with_issues(self):
        report = SchemaValidationReport(schema_name="deriva-ml", strict_mode=True)
        report.add_error("table", "Missing table Dataset", table="Dataset")
        report.add_warning("column", "Wrong type", table="Workflow", column="Name")

        text = report.to_text()
        assert "INVALID" in text
        assert "1 errors" in text
        assert "1 warnings" in text
        assert "Dataset" in text
        assert "Workflow" in text

    def test_to_dict(self):
        report = SchemaValidationReport(schema_name="deriva-ml", strict_mode=False)
        report.add_error("table", "Missing table", table="Dataset")

        d = report.to_dict()
        assert d["schema_name"] == "deriva-ml"
        assert d["strict_mode"] is False
        assert d["is_valid"] is False
        assert d["summary"]["errors"] == 1
        assert len(d["issues"]) == 1
        assert d["issues"][0]["category"] == "table"


class TestSchemaValidatorIntegration:
    """Integration tests for SchemaValidator against a real catalog."""

    def test_validate_valid_schema(self, test_ml):
        """Test validation of a properly created ML schema."""
        report = test_ml.validate_schema(strict=False)

        # The catalog should be valid
        assert report.is_valid, f"Schema should be valid:\n{report.to_text()}"
        assert len(report.errors) == 0

    def test_validate_vocabulary_terms_exist(self, test_ml):
        """Test that required vocabulary terms are present."""
        report = test_ml.validate_schema(strict=False)

        # Check no vocabulary term errors
        vocab_errors = [
            i for i in report.errors if i.category == "vocabulary_term"
        ]
        assert len(vocab_errors) == 0, f"Missing vocabulary terms: {vocab_errors}"

    def test_validate_core_tables_exist(self, test_ml):
        """Test that all core tables exist."""
        report = test_ml.validate_schema(strict=False)

        # Check no table errors
        table_errors = [i for i in report.errors if i.category == "table"]
        assert len(table_errors) == 0, f"Missing tables: {table_errors}"

    def test_validate_association_tables_exist(self, test_ml):
        """Test that all association tables exist."""
        report = test_ml.validate_schema(strict=False)

        # Check no association table errors
        assoc_errors = [
            i for i in report.errors if i.category == "association_table"
        ]
        assert len(assoc_errors) == 0, f"Missing association tables: {assoc_errors}"

    def test_validate_convenience_function(self, test_ml):
        """Test the validate_ml_schema convenience function."""
        report = validate_ml_schema(test_ml, strict=False)
        assert isinstance(report, SchemaValidationReport)
        assert report.is_valid

    def test_strict_mode_with_extra_table(self, test_ml):
        """Test that strict mode reports extra tables.

        We create an extra table in the ML schema and verify
        strict mode reports it as an error.
        """
        from deriva_ml import BuiltinTypes, ColumnDefinition, TableDefinition

        # Create an extra table directly in the ML schema
        table_def = TableDefinition(
            name="ExtraTestTable",
            columns=[
                ColumnDefinition(name="TestColumn", type=BuiltinTypes.text),
            ],
        )
        # Create in ML schema directly
        ml_schema = test_ml.model.model.schemas[test_ml.ml_schema]
        ml_schema.create_table(table_def.to_dict())

        # In non-strict mode, extra tables should be allowed
        report_non_strict = test_ml.validate_schema(strict=False)
        # May have info about extra table but not errors
        extra_table_errors = [
            i for i in report_non_strict.errors if i.category == "extra_table"
        ]
        assert len(extra_table_errors) == 0

        # In strict mode, extra tables should be reported as errors
        report_strict = test_ml.validate_schema(strict=True)
        extra_table_errors_strict = [
            i for i in report_strict.errors if i.category == "extra_table"
        ]
        assert len(extra_table_errors_strict) >= 1, (
            f"Strict mode should report extra tables: {report_strict.to_text()}"
        )

    def test_validate_column_types(self, test_ml):
        """Test that column type validation works."""
        report = test_ml.validate_schema(strict=False)

        # Check no column type errors (warnings are OK)
        column_errors = [i for i in report.errors if i.category == "column"]
        assert len(column_errors) == 0, f"Column errors: {column_errors}"


class TestExpectedSchemaStructure:
    """Tests to verify the expected schema structure constants are correct."""

    def test_expected_tables_not_empty(self):
        """Verify expected table columns dict is populated."""
        assert len(EXPECTED_TABLE_COLUMNS) > 0
        assert "Dataset" in EXPECTED_TABLE_COLUMNS
        assert "Workflow" in EXPECTED_TABLE_COLUMNS
        assert "Execution" in EXPECTED_TABLE_COLUMNS

    def test_expected_vocabulary_tables_not_empty(self):
        """Verify expected vocabulary tables list is populated."""
        assert len(EXPECTED_VOCABULARY_TABLES) > 0
        assert "Dataset_Type" in EXPECTED_VOCABULARY_TABLES
        assert "Workflow_Type" in EXPECTED_VOCABULARY_TABLES
        assert "Asset_Type" in EXPECTED_VOCABULARY_TABLES

    def test_expected_vocabulary_terms_not_empty(self):
        """Verify expected vocabulary terms dict is populated."""
        assert len(EXPECTED_VOCABULARY_TERMS) > 0
        assert "Asset_Type" in EXPECTED_VOCABULARY_TERMS
        assert len(EXPECTED_VOCABULARY_TERMS["Asset_Type"]) > 0

    def test_expected_association_tables_not_empty(self):
        """Verify expected association tables list is populated."""
        assert len(EXPECTED_ASSOCIATION_TABLES) > 0
        assert "Dataset_Dataset_Type" in EXPECTED_ASSOCIATION_TABLES
        assert "Dataset_Dataset" in EXPECTED_ASSOCIATION_TABLES


class TestValidatorTypeCompatibility:
    """Tests for the type compatibility checking logic."""

    def test_types_compatible_same_type(self, test_ml):
        """Test that same types are compatible."""
        validator = SchemaValidator(test_ml)
        assert validator._types_compatible("text", "text")
        assert validator._types_compatible("int4", "int4")

    def test_types_compatible_text_based(self, test_ml):
        """Test that text-based types are compatible with each other."""
        validator = SchemaValidator(test_ml)
        assert validator._types_compatible("text", "markdown")
        assert validator._types_compatible("markdown", "text")
        assert validator._types_compatible("text", "ermrest_curie")

    def test_types_compatible_timestamp_based(self, test_ml):
        """Test that timestamp types are compatible with each other."""
        validator = SchemaValidator(test_ml)
        assert validator._types_compatible("timestamp", "timestamptz")
        assert validator._types_compatible("timestamptz", "ermrest_rct")

    def test_types_not_compatible_different(self, test_ml):
        """Test that incompatible types are rejected."""
        validator = SchemaValidator(test_ml)
        assert not validator._types_compatible("text", "int4")
        assert not validator._types_compatible("boolean", "text")
