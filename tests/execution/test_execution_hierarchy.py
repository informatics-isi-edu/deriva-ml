"""Tests for ExecutionRecord.list_execution_parents /
list_execution_children (renamed from list_parent_executions /
list_nested_executions)."""

from __future__ import annotations


def test_list_execution_parents_symbol():
    """New name exists; old name is gone (R5.1 hard cutover)."""
    from deriva_ml.execution.execution_record import ExecutionRecord as LiveER

    assert hasattr(LiveER, "list_execution_parents"), \
        "list_execution_parents should exist"
    assert hasattr(LiveER, "list_execution_children"), \
        "list_execution_children should exist"
    assert not hasattr(LiveER, "list_parent_executions"), \
        "list_parent_executions should be removed (R5.1 hard cutover)"
    assert not hasattr(LiveER, "list_nested_executions"), \
        "list_nested_executions should be removed (R5.1 hard cutover)"
