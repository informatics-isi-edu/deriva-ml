# Documentation for Lineage models in DerivaML

The `lookup_lineage()` method on `DerivaML` returns a tree of provenance
information for any artifact RID (Dataset, Asset, Feature value, or
Execution). The Pydantic models that shape the response are defined in
`deriva_ml.execution.lineage`.

For the user-guide walkthrough — including common patterns, depth
control, cycle handling, and the data-flow-vs-orchestration distinction
(see ADR-0001) — see
[Running an experiment — How to trace an artifact's lineage](../user-guide/executions.md#how-to-trace-an-artifacts-lineage).

The method itself is documented on the `DerivaML` class:
[DerivaML — lookup_lineage](deriva_ml_base.md).

::: deriva_ml.execution.lineage
    handler: python
