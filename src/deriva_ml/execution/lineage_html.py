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
    """Render the walk tree recursively as nested lists."""
    if not node:
        return '<p class="dim">No walk &mdash; the artifact has no expandable producer.</p>'

    def render_node(n: dict) -> str:
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
        parents = n.get("parents") or []
        if parents:
            parts.append("<ul>" + "".join(render_node(p) for p in parents) + "</ul>")
        parts.append("</li>")
        return "".join(parts)

    return f'<ul class="walk">{render_node(node)}</ul>'


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


def lineage_result_to_html(result: Any, *, generated_note: str | None = None) -> str:
    """Render a ``LineageResult`` (or its ``model_dump()`` dict) to HTML.

    Args:
        result: A :class:`~deriva_ml.execution.lineage.LineageResult` or the
            dict produced by its ``model_dump()``. Rendering never touches a
            catalog.
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
  <h2>Data-flow walk</h2>
  <p class="note">Producing executions of the artifact and of every consumed input,
  walked to the natural root of each branch. Consumed datasets show the version
  actually consumed.</p>
  {_walk_html(data.get("lineage"))}
  <footer>Rendered by deriva-ml <code>lineage_result_to_html</code> &mdash; the source
  JSON (<code>LineageResult.model_dump()</code>) is the audit artifact of record.</footer>
</main>
</body>
</html>
"""


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
    parser.add_argument("--json", dest="json_out", help="Also write the walk's model_dump() JSON here.")
    args = parser.parse_args(argv)

    if args.input:
        data: Any = json.loads(Path(args.input).read_text())
        note = f"Re-rendered from {Path(args.input).name} on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
    else:
        if not (args.host and args.catalog):
            parser.error("--rid requires --host and --catalog")
        from deriva_ml import DerivaML

        ml = DerivaML(hostname=args.host, catalog_id=args.catalog)
        result = ml.lookup_lineage(args.rid)
        data = result.model_dump()
        note = f"Walked {args.rid} on {args.host}/{args.catalog} at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(data, indent=2, default=str))
    Path(args.output).write_text(lineage_result_to_html(data, generated_note=note))
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
