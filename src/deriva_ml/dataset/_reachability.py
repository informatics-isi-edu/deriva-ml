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

from collections import defaultdict
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


def _needed_columns(seg: tuple[str, str], model: Any) -> set[str]:
    """Minimal column projection to fetch for one table in the walk.

    Returns ``RID`` plus every column the in-memory join needs:

    - **Outbound FK columns** (this table -> parent): needed when this table
      is the FK-holder for a hop.
    - **Inbound-FK *referenced* columns** (child -> this table): needed when
      this table is the FK TARGET. Critically, a vocabulary FK references the
      target's ``Name`` (e.g. ``*_Execution.Asset_Role -> Asset_Role.Name``),
      not ``RID`` -- omitting ``Name`` silently undercounts vocab-leaf tables
      (the prototype was off by 817 rows until this was added).
    - **Asset ``Length``**: for asset-byte summation.

    Args:
        seg: ``(schema, table)``.
        model: deriva-py Model.

    Returns:
        Set of column names to project in the table fetch.

    Example:
        >>> ...  # doctest: +SKIP  (needs a Model)
    """
    s, t = seg
    tbl = model.schemas[s].tables[t]
    cols = {"RID"}
    for fk in tbl.foreign_keys:
        for c in fk.foreign_key_columns:
            cols.add(c.name)
    for fk in tbl.referenced_by:
        for c in fk.referenced_columns:
            cols.add(c.name)
    if tbl.is_asset() and "Length" in tbl.column_definitions.elements:
        cols.add("Length")
    return cols


def _reached_rids_for_path(
    fk_path: tuple[tuple[str, str], ...],
    *,
    anchor_rids: set[str],
    fetched_rows: dict[tuple[str, str], list[dict]],
    model: Any,
) -> set[str] | None:
    """RIDs of the terminal table reachable along one FK path.

    Walks the path hop-by-hop from the anchor RIDs, maintaining the set of
    in-scope RIDs of the current table. Each hop advances the scope using the
    FK constraint(s) between adjacent segments, hash-indexed for O(n) probes.

    Args:
        fk_path: Anchor-first tuple of ``(schema, table)`` segments.
        anchor_rids: RIDs of ``fk_path[0]`` that seed the walk.
        fetched_rows: ``{(schema, table): [row dict, ...]}`` -- the once-fetched
            projected rows for every involved table.
        model: deriva-py Model (for FK resolution).

    Returns:
        Set of reachable terminal-table RIDs, or ``None`` if any hop has no
        resolvable FK (an unfollowable path -- caller treats as no contribution).

    Example:
        >>> ...  # doctest: +SKIP  (needs fetched rows + Model)
    """
    cur_seg = fk_path[0]
    if len(fk_path) == 1:
        all_rids = {r["RID"] for r in fetched_rows.get(cur_seg, [])}
        return all_rids & anchor_rids

    cur_scope: set[str] = set(anchor_rids)
    prev_seg = fk_path[0]
    # Local FK-value index cache: (seg, col) -> {fk_value: {RID, ...}}.
    fk_index_cache: dict[tuple[tuple[str, str], str], dict] = {}

    def _fk_index(seg: tuple[str, str], col: str) -> dict:
        keyc = (seg, col)
        idx = fk_index_cache.get(keyc)
        if idx is None:
            idx = defaultdict(set)
            for r in fetched_rows.get(seg, []):
                v = r.get(col)
                if v is not None:
                    idx[v].add(r["RID"])
            fk_index_cache[keyc] = idx
        return idx

    for nxt_seg in fk_path[1:]:
        constraints = _fk_join_columns(prev_seg, nxt_seg, model)
        if not constraints:
            return None
        new_scope: set[str] = set()
        for con in constraints:
            pairs = con["pairs"]
            if con["fk_on"] == "cur":
                if len(pairs) == 1 and pairs[0][0] == "RID":
                    cur_col = pairs[0][1]
                    idx = _fk_index(nxt_seg, cur_col)
                    for pv in cur_scope:
                        hit = idx.get(pv)
                        if hit:
                            new_scope |= hit
                else:
                    prev_rows = fetched_rows.get(prev_seg, [])
                    allowed = {tuple(r.get(pc) for pc, _cc in pairs) for r in prev_rows if r["RID"] in cur_scope}
                    for r in fetched_rows.get(nxt_seg, []):
                        if tuple(r.get(cc) for _pc, cc in pairs) in allowed:
                            new_scope.add(r["RID"])
            else:  # fk_on == "prev"
                prev_rows = fetched_rows.get(prev_seg, [])
                if len(pairs) == 1 and pairs[0][1] == "RID":
                    prev_col = pairs[0][0]
                    for r in prev_rows:
                        if r["RID"] in cur_scope:
                            v = r.get(prev_col)
                            if v is not None:
                                new_scope.add(v)
                else:
                    # Composite / non-RID FK held on prev: match nxt rows whose
                    # full referenced-column tuple appears among the in-scope
                    # prev rows' FK-column tuples. Matching on pairs[0] alone
                    # would over-count rows that share only the first column.
                    allowed = {tuple(r.get(pc) for pc, _cc in pairs) for r in prev_rows if r["RID"] in cur_scope}
                    for r in fetched_rows.get(nxt_seg, []):
                        if tuple(r.get(cc) for _pc, cc in pairs) in allowed:
                            new_scope.add(r["RID"])
        cur_scope = new_scope
        prev_seg = nxt_seg
    return cur_scope


def compute_reachability(
    *,
    reached: ReachedPaths,
    anchor_rids: list[str],
    model: Any,
    fetch: FetchFn,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]], dict[tuple[str, str], list[dict]]]:
    """Compute exact per-table reachable RID sets + asset byte maps.

    The client-side replacement for the deep-server-join estimate path.
    Fetches each involved edge/target table once (projected to the columns
    the in-memory join needs), then for every reached table unions the RID
    sets contributed by each FK path. For asset tables, builds ``{RID:
    Length}`` over the reached RIDs only.

    Args:
        reached: ``CatalogBagBuilder.iter_reached_paths()`` output --
            ``{(schema, table): [fk_path, ...]}``.
        anchor_rids: Dataset RID + every recursive descendant dataset RID.
        model: deriva-py Model (``cb._get_model()``).
        fetch: ``(schema, table, columns) -> rows`` -- injected so the engine
            is testable. Production binds it to the path builder.

    Returns:
        ``(rids_by_table, asset_lengths_by_table, fetched_rows)``:

        - ``rids_by_table`` -- table name -> set of reachable RIDs (exact
          union across all FK paths). Shape matches the retired
          ``run_estimate_queries`` first output so ``assemble_estimate``
          consumes it unchanged.
        - ``asset_lengths_by_table`` -- table name -> ``{RID: Length}`` for
          asset tables (second retired output shape).
        - ``fetched_rows`` -- ``{(schema, table): rows}``, the once-fetched
          projected rows, returned so the caller can derive CSV-byte samples
          without a second fetch (see :func:`sample_rows_from_fetched`).

    Example:
        >>> ...  # doctest: +SKIP  (needs a live walk + Model)
    """
    # 1. Every distinct (schema, table) that appears as a hop in any path.
    edge_tables: set[tuple[str, str]] = set()
    for fk_paths in reached.values():
        for fk_path in fk_paths:
            edge_tables.update(fk_path)

    # 2. Fetch each edge table ONCE, projected.
    fetched_rows: dict[tuple[str, str], list[dict]] = {}
    for seg in edge_tables:
        s, t = seg
        fetched_rows[seg] = fetch(s, t, _needed_columns(seg, model))

    anchor_set = set(anchor_rids)

    # 3. Union RIDs per reached table across its FK paths.
    rids_by_table: dict[str, set[str]] = {}
    # (schema, table)-keyed alongside the bare-name rids_by_table: phase 4
    # needs the schema to look up is_asset()/Length on model.schemas[s].tables[t].
    client_rids: dict[tuple[str, str], set[str]] = {}
    for key, fk_paths in reached.items():
        union: set[str] = set()
        for fk_path in fk_paths:
            rr = _reached_rids_for_path(fk_path, anchor_rids=anchor_set, fetched_rows=fetched_rows, model=model)
            # None (unfollowable hop) or empty set: no contribution to the union.
            if rr:
                union |= rr
        client_rids[key] = union
        rids_by_table[key[1]] = union

    # 4. Asset Length sums over reached RIDs only.
    asset_lengths_by_table: dict[str, dict[str, int]] = {}
    for key, rids in client_rids.items():
        s, t = key
        tbl = model.schemas[s].tables[t]
        if not tbl.is_asset():
            continue
        by_rid = {r["RID"]: r for r in fetched_rows.get(key, [])}
        lengths: dict[str, int] = {}
        for rid in rids:
            r = by_rid.get(rid)
            if r and r.get("Length") is not None:
                lengths[rid] = int(r["Length"])
        asset_lengths_by_table[t] = lengths

    return rids_by_table, asset_lengths_by_table, fetched_rows


__all__ = ["ReachedPaths", "FetchFn", "compute_reachability"]
