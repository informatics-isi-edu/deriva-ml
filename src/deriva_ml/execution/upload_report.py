"""``UploadReport`` — the public return type of :meth:`DerivaML.upload_pending`.

:meth:`upload_pending` drives ``Execution._bag_commit_upload`` per
execution; the report aggregates those per-execution outcomes.

The shape is intentionally narrow:

* ``execution_rids``: which executions the call attempted.
* ``total_uploaded`` / ``total_failed``: aggregate counts across
  every execution drained.
* ``per_table``: ``"{schema}:{table}" -> {"uploaded": int, "failed": int}``
  rolled up across executions. Useful for "did Image rows actually
  land" diagnostics without inspecting individual bag load reports.
* ``errors``: human-readable error lines, one per failed execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UploadReport:
    """Result of a :meth:`DerivaML.upload_pending` call.

    Attributes:
        execution_rids: Executions the call attempted to drain.
        total_uploaded: Sum of rows that landed at the destination
            across all drained executions. Equal to the sum of
            :attr:`deriva.bag.catalog_loader.LoadReport.total_rows_inserted`
            for each successful bag-commit.
        total_failed: Number of executions whose bag-commit raised.
            Per-row failures inside a successful bag-commit count as
            ``uploaded`` from the report's perspective — the load
            either committed or didn't.
        per_table: Per-(schema, table) roll-up across executions.
            ``{"schema:table": {"uploaded": N, "failed": M}}``.
        errors: One error line per failed execution. Empty when all
            executions drained successfully.
    """

    execution_rids: list[str]
    total_uploaded: int
    total_failed: int
    per_table: dict[str, dict[str, int]]
    errors: list[str] = field(default_factory=list)


__all__ = ["UploadReport"]
