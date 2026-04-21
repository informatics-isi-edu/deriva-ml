"""Guards for the core/enums.py modernization to StrEnum.

After this refactor, every *string-valued* enum class in core/enums.py
uses enum.StrEnum as its base. str(MemberX) returns its value string
(not 'ClassName.MemberX'). Membership semantics and value lookup
behavior are unchanged. UploadState keeps plain Enum because its
values are ints, not strings.
"""

from __future__ import annotations

from enum import StrEnum


def test_base_str_enum_is_gone():
    """The legacy BaseStrEnum helper is removed — StrEnum is used directly."""
    from deriva_ml.core import enums
    assert not hasattr(enums, "BaseStrEnum"), (
        "BaseStrEnum should be removed in A5; use enum.StrEnum directly."
    )


def test_execution_status_is_str_enum():
    """Phase 2 Subsystem 1a replaced legacy core.enums.Status with
    deriva_ml.execution.state_store.ExecutionStatus (title-case values).
    """
    from deriva_ml.execution.state_store import ExecutionStatus
    assert issubclass(ExecutionStatus, StrEnum)


def test_mlvocab_is_str_enum():
    from deriva_ml.core.enums import MLVocab
    assert issubclass(MLVocab, StrEnum)


def test_mlasset_is_str_enum():
    from deriva_ml.core.enums import MLAsset
    assert issubclass(MLAsset, StrEnum)


def test_mltable_is_str_enum():
    from deriva_ml.core.enums import MLTable
    assert issubclass(MLTable, StrEnum)


def test_execmetadatatype_is_str_enum():
    from deriva_ml.core.enums import ExecMetadataType
    assert issubclass(ExecMetadataType, StrEnum)


def test_execassettype_is_str_enum():
    from deriva_ml.core.enums import ExecAssetType
    assert issubclass(ExecAssetType, StrEnum)


def test_str_returns_value_not_dotted_name():
    """StrEnum.__str__ returns the value, not 'ClassName.MEMBER'."""
    from deriva_ml.core.enums import MLVocab
    from deriva_ml.execution.state_store import ExecutionStatus
    s = next(iter(ExecutionStatus))
    v = next(iter(MLVocab))
    assert str(s) == s.value
    assert str(v) == v.value


def test_value_lookup_still_works():
    """Member-by-value lookup is unchanged."""
    from deriva_ml.execution.state_store import ExecutionStatus
    s = next(iter(ExecutionStatus))
    assert ExecutionStatus(s.value) is s
