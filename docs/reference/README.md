# Reference docs

Formal, code-grounded reference manuals for the operations behind
dataset bags and ML-frame production. Each states exact behavior with
numbered rules (tagged `[engine: deriva-py]` vs `[deriva-ml]`) and
demo-catalog-verified worked examples. They complement the
example-led tutorials in `../user-guide/`.

## fk-traversal.md

The FK-traversal engine: the `FKTraversalPolicy` fields (walk-phase vs
load-phase), the bidirectional walker, terminal tables, the vocabulary
leaf rule, and `max_depth`. The foundation the other two build on.

## bag-export.md

How `DatasetBagBuilder` drives the traversal engine to produce a
dataset bag: nested-descendant anchors, the empty-association pruning
rule, the provenance-terminal-tables guard, and the server-side export
spec. Includes the "same engine, different consumer" entries for
size-estimate, clone, and drift.

## denormalization.md

The formal rules for the wide-table (`get_denormalized_*`) operations
on a downloaded `DatasetBag` — `row_per` / `via` / selector semantics
and exact return shapes. Formal counterpart to the denormalization
tutorial.

## schema.md

Authoritative description of the `deriva-ml` schema.

To change the schema:

1. Edit `schema.md` to describe the intended state.
2. Edit `src/deriva_ml/schema/create_schema.py` to match.
3. Run `uv run deriva-ml-validate-schema` locally.
4. Commit both files together.

See `schema.md` for what is and isn't validated.
