"""Client-side FK-reachability engine for fast bag-size estimation.

Replaces the deep-server-join estimate path (which issued ~477 multi-hop
FK-JOIN queries the server evaluated at 16-60s each) with: fetch each
reached table's edge columns ONCE (whole-table projected scan, ~0.3s for
240k rows), then reconstruct FK reachability by in-memory BFS over the
symbolic FK paths the bag walker discovered. Exact union counts, ~10x
faster. See docs/adr/0008 and the estimate-bag-size-client-side-join memo.

The engine is pure: its only catalog contact is the injected ``fetch``
callable, so the BFS and FK-resolution logic are unit-testable against a
synthetic in-memory catalog with no live server.
"""

from __future__ import annotations

from typing import Any, Callable

ReachedPaths = dict[tuple[str, str], list[tuple[tuple[str, str], ...]]]
FetchFn = Callable[[str, str, set[str]], list[dict]]


def _fk_join_columns(
    prev_seg: tuple[str, str],
    cur_seg: tuple[str, str],
    model: Any,
) -> list[dict]:
    """Return the FK constraints linking two adjacent path segments.

    Each constraint dict is ``{"fk_on": "prev"|"cur", "pairs": [(prev_col,
    cur_col), ...]}`` where ``pairs[i]`` are column NAMES that must be equal
    to follow ``prev -> cur``. ``fk_on`` records which table physically
    holds the FK, which decides the in-memory join direction.

    Args:
        prev_seg: ``(schema, table)`` of the source segment.
        cur_seg: ``(schema, table)`` of the destination segment.
        model: deriva-py Model.

    Returns:
        List of constraint dicts (one per FK constraint between the tables,
        in either direction); empty if no FK links them.

    Example:
        >>> # See tests/dataset/test_reachability.py for synthetic-model
        >>> # exercises; this needs a Model object.
        >>> ...  # doctest: +SKIP
    """
    ps, pt = prev_seg
    cs, ct = cur_seg
    prev_tbl = model.schemas[ps].tables[pt]
    cur_tbl = model.schemas[cs].tables[ct]
    constraints: list[dict] = []
    # FK on prev pointing to cur.
    for fk in prev_tbl.foreign_keys:
        if fk.pk_table is cur_tbl:
            pairs = [(fkc.name, rc.name) for fkc, rc in zip(fk.foreign_key_columns, fk.referenced_columns)]
            constraints.append({"fk_on": "prev", "pairs": pairs})
    # FK on cur pointing to prev.
    for fk in cur_tbl.foreign_keys:
        if fk.pk_table is prev_tbl:
            pairs = [
                (rc.name, fkc.name)  # (prev_col, cur_col)
                for fkc, rc in zip(fk.foreign_key_columns, fk.referenced_columns)
            ]
            constraints.append({"fk_on": "cur", "pairs": pairs})
    return constraints
