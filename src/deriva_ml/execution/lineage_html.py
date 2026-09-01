"""Render a :class:`~deriva_ml.execution.lineage.LineageResult` as HTML.

One self-contained page a reviewer can open, share, or archive: the root
artifact with its current version and origin attribution, the full
version-attribution trace (origin first, later authors as touchers), the
data-flow walk with workflow code identity on every node, and the walk's
completeness flags.

Design properties (issue #378, adopted from the deploy-repo renderer that
proved them):

- **Decoupled through JSON** — rendering consumes a ``LineageResult`` or
  its ``model_dump()`` dict and never touches a catalog, so a report can
  be regenerated, restyled, or diffed long after the walk; the JSON stays
  the audit artifact of record.
- **Self-contained** — a single HTML file: inline CSS, no JavaScript, no
  external assets.
- **Zero dependencies** — standard library only.
- **Untrusted text discipline** — descriptions, names, and versions are
  catalog data; everything is HTML-escaped, and workflow URLs become
  links only when they are ``http(s)`` web URLs.

Example:
    Render a walk to a file::

        $ deriva-ml-lineage --rid 1-ABC0 --host example.org \\
              --catalog 42 --output lineage.html

    Re-render offline from a saved ``model_dump()``::

        $ deriva-ml-lineage --input lineage.json --output lineage.html
"""

from __future__ import annotations

import argparse
import html as _html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CSS = """
:root { --fg:#1c2430; --dim:#6b7686; --line:#dde3ea; --bg:#f7f9fb; --card:#ffffff;
        --ok:#0d7a45; --warn:#a15c00; --bad:#a12727; --accent:#1c4f8a; }
body { margin:0; padding:2rem; background:var(--bg); color:var(--fg);
       font:15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size:1.3rem; margin:0 0 .25rem; }
h2 { font-size:1.05rem; margin:1.8rem 0 .5rem; }
.note { color:var(--dim); font-size:.85rem; margin-bottom:1.2rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:1rem 1.2rem; margin:.6rem 0; }
.rid { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.9em;
       background:#eef2f6; border-radius:4px; padding:.05rem .35rem; }
.badge { display:inline-block; font-size:.72rem; font-weight:600; letter-spacing:.02em;
         border-radius:9999px; padding:.1rem .55rem; vertical-align:middle; margin-left:.4rem; }
.badge.ok { background:#e2f4ea; color:var(--ok); }
.badge.bad { background:#fbe9e9; color:var(--bad); }
.badge.warn { background:#fdf1e0; color:var(--warn); }
.badge.ver { background:#e8eef7; color:var(--accent); }
.badge.origin { background:#e8eef7; color:var(--accent); }
.dim { color:var(--dim); }
table { border-collapse:collapse; width:100%; background:var(--card);
        border:1px solid var(--line); border-radius:8px; overflow:hidden; }
th, td { text-align:left; padding:.45rem .7rem; border-top:1px solid var(--line);
         vertical-align:top; font-size:.9rem; }
thead th { border-top:none; background:#eef2f6; font-size:.78rem;
           text-transform:uppercase; letter-spacing:.04em; color:var(--dim); }
.walk { list-style:none; padding-left:0; }
.walk ul { list-style:none; border-left:2px solid var(--line);
           margin:.3rem 0 .3rem .55rem; padding-left:1rem; }
.walk li { margin:.5rem 0; }
.exec { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:.6rem .9rem; }
.wf { color:var(--dim); font-size:.85rem; }
.io { font-size:.85rem; margin:.2rem 0 0; color:var(--fg); }
footer { margin-top:2rem; color:var(--dim); font-size:.8rem; }
a { color:var(--accent); }
"""


def _esc(value: Any) -> str:
    """HTML-escape catalog-sourced text; None renders as an em dash."""
    if value is None:
        return '<span class="dim">&mdash;</span>'
    return _html.escape(str(value))


def _workflow_html(wf: dict | None) -> str:
    """Render a workflow summary: name, version, and a link for web URLs only."""
    if not wf:
        return '<span class="dim">no workflow recorded</span>'
    name = _esc(wf.get("name") or wf.get("rid"))
    version = wf.get("version")
    version_html = f' <span class="badge ver">{_esc(version)}</span>' if version else ""
    url = wf.get("url") or ""
    if url.startswith(("http://", "https://")):
        name = f'<a href="{_html.escape(url, quote=True)}">{name}</a>'
    elif url:
        name = f'{name} <span class="dim">({_esc(url)})</span>'
    return f"{name}{version_html}"


def _execution_html(ex: dict | None) -> str:
    """Render one execution summary line."""
    if not ex:
        return '<span class="dim">unresolved execution</span>'
    desc = ex.get("description")
    desc_html = f" &mdash; {_esc(desc)}" if desc else ""
    return (
        f'<span class="rid">{_esc(ex.get("rid"))}</span>'
        f'{desc_html}<div class="wf">{_workflow_html(ex.get("workflow"))}'
        f" &middot; {_esc(ex.get('status') or 'Unknown')}</div>"
    )


def _walk_html(node: dict | None) -> str:
    """Render the walk tree as nested lists, iteratively.

    Iterative on purpose: a valid walk can approach the 500-execution cap
    as one deep chain, and recursive rendering (a frame plus a generator
    frame per level) can exceed Python's recursion limit after the walk
    itself succeeded. An explicit stack has no depth ceiling.
    """
    if not node:
        return '<p class="dim">No walk &mdash; the artifact has no expandable producer.</p>'

    def node_open(n: dict) -> str:
        parts = [f'<li><div class="exec">{_execution_html(n.get("execution"))}']
        if n.get("already_shown"):
            parts.append('<div class="io dim">already shown elsewhere in this tree</div>')
        datasets = n.get("consumed_datasets") or []
        if datasets:
            items = ", ".join(
                f'<span class="rid">{_esc(d.get("rid"))}</span>'
                + (f' <span class="badge ver">{_esc(d["version"])}</span>' if d.get("version") else "")
                for d in datasets
            )
            parts.append(f'<div class="io">consumed datasets: {items}</div>')
        assets = n.get("consumed_assets") or []
        if assets:
            items = ", ".join(
                f'{_esc(a.get("filename") or a.get("rid"))} <span class="rid">{_esc(a.get("rid"))}</span>'
                for a in assets
            )
            parts.append(f'<div class="io">consumed assets: {items}</div>')
        parts.append("</div>")
        return "".join(parts)

    out: list[str] = ['<ul class="walk">']
    # Stack of (node, None) to open, or (None, literal) to emit closers.
    stack: list[tuple[dict | None, str | None]] = [(node, None)]
    while stack:
        current, literal = stack.pop()
        if literal is not None:
            out.append(literal)
            continue
        assert current is not None
        out.append(node_open(current))
        parents = current.get("parents") or []
        stack.append((None, "</li>"))
        if parents:
            stack.append((None, "</ul>"))
            for p in reversed(parents):
                stack.append((p, None))
            stack.append((None, "<ul>"))
    out.append("</ul>")
    return "".join(out)


def _origin_badge(root: dict) -> str:
    """Tri-state origin badge; empty for non-Dataset roots (not applicable)."""
    recorded = root.get("origin_recorded")
    if recorded is True:
        return '<span class="badge ok">origin recorded</span>'
    if recorded is False:
        return '<span class="badge bad">origin unrecorded</span>'
    return ""


def _history_html(history: list[dict]) -> str:
    """Render the version-attribution trace, earliest first, origin marked."""
    if not history:
        return ""
    rows = []
    for i, entry in enumerate(history):
        marker = '<span class="badge origin">ORIGIN</span>' if i == 0 else ""
        ex = entry.get("execution")
        author = (
            _execution_html(ex)
            if ex
            else (
                f'<span class="rid">{_esc(entry["execution_rid"])}</span> <span class="dim">(unresolved)</span>'
                if entry.get("execution_rid")
                else '<span class="dim">no author recorded</span>'
            )
        )
        rows.append(
            f'<tr><td><span class="badge ver">{_esc(entry.get("version"))}</span>{marker}</td>'
            f"<td>{author}</td><td>{_esc(entry.get('description'))}</td></tr>"
        )
    return (
        "<h2>Version attribution</h2>"
        '<p class="note">Every version\'s author, earliest recorded first. Only the first '
        "row answers &ldquo;how did this dataset come to exist?&rdquo;; later rows are "
        "touchers &mdash; migrations, backfills, re-releases.</p>"
        "<table><thead><tr><th>Version</th><th>Author</th><th>Notes</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _features_html(feature_producers: dict[str, list[dict]] | None) -> str:
    """Render the feature-producer candidates, grouped by dataset."""
    if not feature_producers:
        return ""
    blocks = []
    for dataset_rid, records in feature_producers.items():
        rows = []
        for rec in records:
            ex = rec.get("execution_rid")
            who = (
                f'<span class="rid">{_esc(ex)}</span>'
                if ex
                else '<span class="badge bad">no producing execution</span>'
            )
            rows.append(
                f"<tr><td>{who}</td><td>{_esc(rec.get('feature_name'))}</td>"
                f"<td>{_esc(rec.get('element_type'))}</td>"
                f"<td>{int(rec.get('value_count') or 0)}</td></tr>"
            )
        blocks.append(
            f'<h3>on dataset <span class="rid">{_esc(dataset_rid)}</span></h3>'
            "<table><thead><tr><th>Execution</th><th>Feature</th><th>Element</th>"
            f"<th>Values</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    return (
        "<h2>Feature producers &middot; candidates</h2>"
        '<p class="note">Executions that wrote feature values onto a walked dataset\'s '
        "members. These are <strong>candidates, not lineage edges</strong>: the catalog "
        "cannot record which features downstream code actually read, so this is the "
        "bounded superset with evidence &mdash; a complete candidate set, never a claim "
        "of use. A row with no producing execution is itself a provenance gap.</p>" + "".join(blocks)
    )


def _graph_svg(data: dict, feature_producers: dict[str, list[dict]] | None) -> str:
    """Draw the walk as a layered inline SVG.

    Honesty constraint: ``LineageNode.parents`` is a flat list — the model
    does not record which parent produced which consumed dataset — so the
    figure draws only edges the data supports: execution boxes chained by
    depth, consumed-dataset pills under their consuming execution, and
    dashed feature-producer boxes attached to the dataset they annotated.
    """
    root = data.get("lineage")
    if not root:
        return ""

    # Flatten by depth, iteratively (deep chains — same reason as _walk_html).
    columns: list[list[dict]] = []
    stack: list[tuple[dict, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        while len(columns) <= depth:
            columns.append([])
        columns[depth].append(node)
        for parent in node.get("parents") or []:
            stack.append((parent, depth + 1))

    box_w, box_h, ds_h, gap_x, gap_y, pad = 180, 46, 22, 70, 18, 20
    feat_counts = {rid: len(recs) for rid, recs in (feature_producers or {}).items()}

    def node_height(n: dict) -> int:
        n_ds = len(n.get("consumed_datasets") or [])
        n_assets = len(n.get("consumed_assets") or [])
        n_feat = sum(1 for d in (n.get("consumed_datasets") or []) if (d.get("rid") in feat_counts))
        return box_h + (n_ds + n_assets + n_feat) * (ds_h + 4)

    col_heights = [sum(node_height(n) for n in col) + gap_y * max(len(col) - 1, 0) for col in columns]
    height = max(col_heights or [box_h]) + 2 * pad
    width = len(columns) * (box_w + gap_x) - gap_x + 2 * pad

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Lineage walk figure" '
        f'style="max-width:100%;height:auto;overflow:visible;background:#fff;'
        f'border:1px solid #dde3ea;border-radius:8px">'
    ]

    # CSS-drawn hover tooltips. Native SVG <title> hover text is rendered
    # by the browser chrome: embedded preview panes never surface it, and
    # full browsers would draw it ON TOP of these overlays as a second,
    # differently-styled tooltip — so the figure carries the accessible
    # text as aria-label instead and draws its own overlays, hidden until
    # :hover, revealed by a pure-CSS sibling rule. Overlays
    # are appended after all content groups: that puts them last in
    # z-order and after their hover targets, which the `~` combinator
    # requires.
    overlays: list[str] = []
    hover_rules: list[str] = []

    def hover_cls(lines: list[str], ax: float, ay: float) -> str:
        """Register a tooltip overlay for `lines` anchored near (ax, ay);
        returns the class attribute value for the hover target group."""
        i = len(overlays)
        shown = [ln if len(ln) <= 64 else ln[:63] + "…" for ln in lines[:12]]
        if len(lines) > 12:
            shown.append(f"… {len(lines) - 12} more (full list in the tables below)")
        t_w = max(len(s) for s in shown) * 6.3 + 18
        t_h = len(shown) * 13 + 12
        tx = max(2.0, min(ax, width - t_w - 2))
        tspans = "".join(f'<tspan x="{tx + 9}" dy="{13 if j else 0}">{_esc(s)}</tspan>' for j, s in enumerate(shown))
        overlays.append(
            f'<g class="tt" id="tt{i}">'
            f'<rect x="{tx}" y="{ay}" width="{t_w:.0f}" height="{t_h}" rx="5" '
            f'fill="#1c2430" opacity="0.95"/>'
            f'<text x="{tx + 9}" y="{ay + 16}" font-size="10" fill="#fff" '
            f'font-family="ui-monospace,monospace">{tspans}</text></g>'
        )
        hover_rules.append(f".hv{i}:hover~#tt{i}{{visibility:visible}}")
        return f"hv{i}"

    centers: dict[int, tuple[float, float]] = {}
    for depth, col in enumerate(columns):
        x = pad + depth * (box_w + gap_x)
        y = pad
        for n in col:
            ex = n.get("execution") or {}
            rid = ex.get("rid") or ""
            wf_rec = ex.get("workflow") or {}
            wf = wf_rec.get("name") or ""
            tip_lines = [
                f"{rid} [{ex.get('status') or 'Unknown'}]",
                "already shown elsewhere in this tree" if n.get("already_shown") else "",
                ex.get("description") or "",
                f"workflow: {wf}" + (f" v{wf_rec['version']}" if wf_rec.get("version") else ""),
                wf_rec.get("url") or "",
            ]
            ex_lines = [x for x in tip_lines if x]
            tip = _esc(chr(10).join(ex_lines))
            parts.append(
                f'<g class="{hover_cls(ex_lines, x + 12, y + box_h + 6)}" role="img" aria-label="{tip}">'
                f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
                f'fill="#eef2f6" stroke="#1c4f8a"'
                f"{" stroke-dasharray='5 3'" if n.get('already_shown') else ''}/>"
                f'<text x="{x + 10}" y="{y + 19}" font-size="12" '
                f'font-family="ui-monospace,monospace">{_esc(rid)}</text>'
                f'<text x="{x + 10}" y="{y + 36}" font-size="10" fill="#6b7686">'
                f"{_esc(wf[:28])}</text></g>"
            )
            centers[id(n)] = (x, y + box_h / 2)
            dy = y + box_h + 4
            for d in n.get("consumed_datasets") or []:
                label = _esc(d.get("rid"))
                if d.get("version"):
                    label += f" @ {_esc(d['version'])}"
                ds_tip_lines = [
                    f"{d.get('rid')}" + (f" @ v{d['version']}" if d.get("version") else ""),
                    d.get("description") or "",
                    f"consumed by {rid}",
                ]
                ds_lines = [x for x in ds_tip_lines if x]
                ds_tip = _esc(chr(10).join(ds_lines))
                parts.append(
                    f'<g class="{hover_cls(ds_lines, x + 20, dy + ds_h + 4)}" role="img" aria-label="{ds_tip}">'
                    f'<rect x="{x + 14}" y="{dy}" width="{box_w - 28}" height="{ds_h}" '
                    f'rx="11" fill="#e8eef7" stroke="#9db4d3"/>'
                    f'<text x="{x + 24}" y="{dy + 15}" font-size="10" '
                    f'font-family="ui-monospace,monospace">{label}</text></g>'
                )
                dy += ds_h + 4
                n_feat = feat_counts.get(d.get("rid"))
                if n_feat:
                    feat_tip_lines = ["feature-producer candidates (not lineage edges):"] + [
                        f"{r.get('execution_rid') or '(no producing execution)'}: "
                        f"{r.get('feature_name')} x{r.get('value_count')} on {r.get('element_type')}"
                        for r in (feature_producers or {}).get(d.get("rid"), [])
                    ]
                    feat_tip = _esc(chr(10).join(feat_tip_lines))
                    parts.append(
                        f'<g class="{hover_cls(feat_tip_lines, x + 34, dy + ds_h + 4)}" '
                        f'role="img" aria-label="{feat_tip}">'
                        f'<rect x="{x + 28}" y="{dy}" width="{box_w - 42}" height="{ds_h}" '
                        f'rx="4" fill="none" stroke="#a15c00" stroke-dasharray="4 3"/>'
                        f'<text x="{x + 36}" y="{dy + 15}" font-size="10" fill="#a15c00">'
                        f"{n_feat} feature producer(s)</text></g>"
                    )
                    dy += ds_h + 4
            for a in n.get("consumed_assets") or []:
                a_label = _esc((a.get("filename") or a.get("rid") or "")[:24])
                a_lines = [
                    x
                    for x in [
                        a.get("filename") or "",
                        f"{a.get('rid')} in {a.get('asset_table')}",
                        f"consumed by {rid}",
                    ]
                    if x
                ]
                a_tip = _esc(chr(10).join(a_lines))
                parts.append(
                    f'<g class="{hover_cls(a_lines, x + 20, dy + ds_h + 4)}" role="img" aria-label="{a_tip}">'
                    f'<rect x="{x + 14}" y="{dy}" width="{box_w - 28}" height="{ds_h}" '
                    f'rx="3" fill="#eef7ee" stroke="#4e8a5a"/>'
                    f'<text x="{x + 24}" y="{dy + 15}" font-size="10" '
                    f'font-family="ui-monospace,monospace">{a_label}</text></g>'
                )
                dy += ds_h + 4
            y += node_height(n) + gap_y
        # Arrows child ← parent: draw from this column's nodes to their parents.
    for depth, col in enumerate(columns):
        for n in col:
            cx, cy = centers[id(n)]
            for parent in n.get("parents") or []:
                px, py = centers[id(parent)]
                p_rid = (parent.get("execution") or {}).get("rid") or "?"
                c_rid = (n.get("execution") or {}).get("rid") or "?"
                arrow_text = f"{p_rid} is an upstream producer of inputs consumed by {c_rid}"
                arrow_tip = _esc(arrow_text)
                arrow_cls = hover_cls([arrow_text], (px + cx + box_w) / 2, (py + cy) / 2 + 10)
                parts.append(
                    f'<g class="{arrow_cls}" role="img" aria-label="{arrow_tip}">'
                    f'<line x1="{px}" y1="{py}" x2="{cx + box_w}" y2="{cy}" '
                    f'stroke="#6b7686" stroke-width="1.5"/>'
                    f'<polygon points="{cx + box_w + 6},{cy} {cx + box_w + 14},{cy - 4} '
                    f'{cx + box_w + 14},{cy + 4}" fill="#6b7686"/></g>'
                )
    parts.append("<style>.tt{visibility:hidden;pointer-events:none}" + "".join(hover_rules) + "</style>")
    parts.extend(overlays)
    parts.append("</svg>")
    return (
        "<h2>Walk figure</h2>"
        '<p class="note">Executions by depth (root leftmost; arrows point from producer '
        "to consumer), consumed datasets (blue pills) and assets (green pills) under their "
        "consuming execution, dashed marks where feature-producer candidates exist. Hover "
        "any element for its full record. The model records parents as a set, so "
        "dataset&rarr;producer pairings are deliberately not drawn.</p>" + "".join(parts)
    )


def lineage_result_to_html(
    result: Any,
    *,
    feature_producers: dict[str, list[dict]] | None = None,
    generated_note: str | None = None,
) -> str:
    """Render a ``LineageResult`` (or its ``model_dump()`` dict) to HTML.

    Args:
        result: A :class:`~deriva_ml.execution.lineage.LineageResult` or the
            dict produced by its ``model_dump()``. Rendering never touches a
            catalog.
        feature_producers: Optional companion data — mapping of dataset RID
            to :meth:`DerivaML.find_feature_producers` records (as dicts).
            Rendered as a clearly-framed CANDIDATES section and marked in
            the figure; deliberately not part of ``LineageResult`` (the walk
            is data-flow only, per ADR-0001).
        generated_note: One-line provenance of the report itself; defaults to
            a UTC date stamp.

    Returns:
        A complete, self-contained HTML page (inline CSS, no JavaScript, no
        external assets). All catalog-sourced text is escaped.

    Example:
        >>> from deriva_ml.execution.lineage import LineageResult, RootDescriptor
        >>> page = lineage_result_to_html(
        ...     LineageResult(root=RootDescriptor(rid="1-0001", type="Asset", description=None))
        ... )
        >>> page.lstrip().startswith("<!DOCTYPE html>")
        True
    """
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    root = data.get("root") or {}
    note = generated_note or f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"

    version_html = f' <span class="badge ver">{_esc(root["version"])}</span>' if root.get("version") else ""
    producing = root.get("producing_execution")
    origin_html = (
        f'<div class="io">origin: {_execution_html(producing)}</div>'
        if producing
        else (
            '<div class="io dim">origin execution unresolved or unrecorded &mdash; see the '
            "version attribution below</div>"
            if root.get("origin_recorded") is not None
            else ""
        )
    )

    flags = []
    if not data.get("walked_complete", True):
        flags.append(
            '<span class="badge warn">partial walk</span> the dependency tree below is '
            "truncated (cap or cycle) and must not be read as complete"
        )
    if data.get("cycle_detected"):
        flags.append('<span class="badge warn">cycle detected</span>')
    if data.get("depth_capped"):
        flags.append('<span class="badge warn">depth capped</span>')
    flags_html = "".join(f"<p>{f}</p>" for f in flags)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lineage &middot; {_esc(root.get("rid"))}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <h1>Lineage of <span class="rid">{_esc(root.get("rid"))}</span>
      <span class="badge ver">{_esc(root.get("type"))}</span>{version_html}{_origin_badge(root)}</h1>
  <p class="note">{_esc(note)} &middot; executions visited: {int(data.get("executions_visited") or 0)}</p>
  {flags_html}
  <section class="card">
    <p>{_esc(root.get("description"))}</p>
    {origin_html}
  </section>
  {_history_html(root.get("version_history") or [])}
  {_graph_svg(data, feature_producers)}
  <h2>Data-flow walk</h2>
  <p class="note">Producing executions of the artifact and of every consumed input,
  walked to the natural root of each branch. Consumed datasets show the version
  actually consumed.</p>
  {_walk_html(data.get("lineage"))}
  {_features_html(feature_producers)}
  <footer>Rendered by deriva-ml <code>lineage_result_to_html</code> &mdash; the source
  JSON (<code>LineageResult.model_dump()</code>) is the audit artifact of record.</footer>
</main>
</body>
</html>
"""


def _gather_feature_producers(ml: Any, data: dict) -> dict[str, list[dict]] | None:
    """Collect feature-producer candidates for every dataset in a walk.

    Datasets are the root (when it is one) plus every consumed dataset
    across the walk tree, deduplicated. Each is scanned with
    :meth:`DerivaML.find_feature_producers`; a dataset whose scan fails
    degrades with a warning on stderr rather than losing the page.
    """
    rids: list[str] = []
    root = data.get("root") or {}
    if root.get("type") == "Dataset" and root.get("rid"):
        rids.append(root["rid"])
    stack = [data.get("lineage")]
    while stack:
        node = stack.pop()
        if not node:
            continue
        for d in node.get("consumed_datasets") or []:
            if d.get("rid"):
                rids.append(d["rid"])
        stack.extend(node.get("parents") or [])
    out: dict[str, list[dict]] = {}
    for rid in dict.fromkeys(rids):
        try:
            records = ml.find_feature_producers(rid)
        except Exception as exc:  # noqa: BLE001 — one dataset must not lose the page
            print(f"warning: feature scan failed for {rid}: {exc}", file=sys.stderr)
            continue
        if records:
            out[rid] = [r.model_dump() for r in records]
    return out or None


def main(argv: list[str] | None = None) -> int:
    """CLI: render a lineage walk (or a saved walk) to a self-contained HTML file.

    Two modes: ``--rid`` + ``--host``/``--catalog`` performs the walk and
    renders it; ``--input lineage.json`` re-renders a saved
    ``model_dump()`` offline with no catalog access.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success).

    Example:
        $ deriva-ml-lineage --rid 1-ABC0 --host example.org --catalog 42 \\
              --output lineage.html --json lineage.json
    """
    parser = argparse.ArgumentParser(
        prog="deriva-ml-lineage",
        description="Render a DerivaML lineage walk as a self-contained HTML report.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rid", help="Artifact RID to walk (Dataset, Asset, Feature value, or Execution).")
    source.add_argument("--input", help="Saved LineageResult.model_dump() JSON to re-render offline.")
    parser.add_argument("--host", help="Catalog hostname (walk mode).")
    parser.add_argument("--catalog", help="Catalog ID (walk mode).")
    parser.add_argument("--output", required=True, help="HTML file to write.")
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Walk mode: skip the find_feature_producers candidate scan.",
    )
    parser.add_argument("--json", dest="json_out", help="Also write the walk's model_dump() JSON here.")
    args = parser.parse_args(argv)

    feature_producers: dict[str, list[dict]] | None = None
    if args.input:
        loaded: Any = json.loads(Path(args.input).read_text(encoding="utf-8"))
        # Envelope {lineage, feature_producers} or a bare model_dump().
        if isinstance(loaded, dict) and "lineage" in loaded and "root" not in loaded:
            data: Any = loaded["lineage"]
            feature_producers = loaded.get("feature_producers") or None
        else:
            data = loaded
        note = f"Re-rendered from {Path(args.input).name} on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
    else:
        if not (args.host and args.catalog):
            parser.error("--rid requires --host and --catalog")
        from deriva_ml import DerivaML

        ml = DerivaML(hostname=args.host, catalog_id=args.catalog)
        result = ml.lookup_lineage(args.rid)
        data = result.model_dump()
        if not args.no_features:
            feature_producers = _gather_feature_producers(ml, data)
        note = f"Walked {args.rid} on {args.host}/{args.catalog} at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"

    if args.json_out:
        envelope = {"lineage": data, "feature_producers": feature_producers}
        Path(args.json_out).write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
    Path(args.output).write_text(
        lineage_result_to_html(data, feature_producers=feature_producers, generated_note=note),
        encoding="utf-8",
    )
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
