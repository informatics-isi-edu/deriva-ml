"""``UploadReport`` — the public return type of the commit-output-assets surface.

Returned by :meth:`Execution.commit_output_assets` (per-execution)
and :meth:`DerivaML.commit_pending_executions` (batch). Both go
through the same lifecycle bracket and yield the same report shape.

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
    """Result of a commit-output-assets call.

    Returned by both :meth:`Execution.commit_output_assets` (with a
    single-element ``execution_rids``) and
    :meth:`DerivaML.commit_pending_executions` (with one entry per
    drained execution).

    Attributes:
        execution_rids: Executions the call attempted to drain.
        total_uploaded: Sum of asset rows that landed at the
            destination across all drained executions.
        total_failed: Number of executions whose commit raised.
            Per-row failures inside a successful commit count as
            ``uploaded`` from the report's perspective — the commit
            either succeeded or didn't.
        per_table: Per-(schema, table) roll-up across executions.
            ``{"schema/table": {"uploaded": N, "failed": M}}``.
        errors: One error line per failed execution. Empty when all
            executions drained successfully.
    """

    execution_rids: list[str]
    total_uploaded: int
    total_failed: int
    per_table: dict[str, dict[str, int]]
    errors: list[str] = field(default_factory=list)


__all__ = ["UploadReport"]
