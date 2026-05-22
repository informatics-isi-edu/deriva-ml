"""AST-based parser for DerivaML hydra-zen config files.

Parses ``src/configs/*.py`` files via Python's ``ast`` module and
finds every ``*Config(...)``/``Workflow(...)`` constructor call. The
parser does NOT execute the file, so it's safe against arbitrary
user code in the configs/ dir.

The four recognized constructor names:

- ``DatasetSpecConfig`` -- dataset entries; ``rid=`` and ``version=`` kwargs
- ``AssetSpecConfig`` -- asset entries; ``rid=`` (and optional ``cache=``)
- ``Workflow`` -- workflow entries; ``rid=`` kwarg
- ``DerivaMLConfig`` -- connection entries; ``hostname=`` and ``catalog_id=``

Edge cases the walker handles:

- ``with_description(DatasetSpecConfig(...), "...")`` -- the wrapper
  is unwrapped; we recurse into its arguments.
- ``builds(DatasetSpec, rid=..., version=...)`` -- hydra-zen's lazy
  partial. The first positional arg names the spec class; we treat
  the call shape like the named-constructor.
- Module-level constant resolution: ``TRAINING_RID = "2-B4C8"`` at
  module scope makes ``rid=TRAINING_RID`` resolvable. Only simple
  name = literal-string assignments at module scope are resolved;
  anything more complex emits a ``ConfigEntry`` with ``rid=None``
  (which the validator then surfaces as ``rid_unresolvable``).
- Comments are naturally skipped by AST parsing.

Matching is by call-name only, not by import. Identical names from
unrelated modules are picked up; callers diagnose false matches via
the file + line + col fields on :class:`ConfigEntry`.
"""

from __future__ import annotations

import ast
import os
import tokenize

from deriva_ml.config.validation import (
    ConfigEntry,
    ConfigEntryKind,
    ConfigFileParseError,
)

# Names we recognize as constructor calls worth validating. ``builds()``
# is recognized only as a *wrapper* -- the inner spec class name (the
# first positional arg) is what determines ``entry_kind``.
_KNOWN_KINDS: set[str] = {
    "DatasetSpecConfig",
    "AssetSpecConfig",
    "Workflow",
    "DerivaMLConfig",
}

# Wrapper calls we unwrap to find the inner spec constructor.
# - ``with_description(spec, description)`` -- first positional arg
# - ``builds(<spec_class>, **kwargs)`` -- treat ``<spec_class>`` as the kind
_UNWRAP_WRAPPERS: set[str] = {"with_description"}


def parse_config_file(
    path: str | os.PathLike[str],
) -> tuple[list[ConfigEntry], ConfigFileParseError | None]:
    """Parse one config file and return its entries.

    Args:
        path: Path to the file. Read with ``tokenize.open()`` so the
            file's declared encoding is respected.

    Returns:
        A pair ``(entries, parse_error)``. ``entries`` is the list of
        ``ConfigEntry`` constructor calls found (empty if the file
        had none). ``parse_error`` is a :class:`ConfigFileParseError`
        if the file couldn't be parsed, else ``None``. The two are
        mutually exclusive: a parse error means ``entries`` is empty.

    Does not raise on syntax errors -- the parse error is returned
    structurally so callers walking many files can record it without
    aborting the walk.
    """
    file_str = os.fspath(path)
    try:
        with tokenize.open(file_str) as handle:
            source = handle.read()
    except (OSError, SyntaxError) as exc:
        # tokenize.open can raise SyntaxError on encoding declarations
        # it doesn't like; record but don't propagate.
        return [], ConfigFileParseError(
            file=file_str,
            line=getattr(exc, "lineno", None),
            message=str(exc),
        )

    try:
        tree = ast.parse(source, filename=file_str)
    except SyntaxError as exc:
        return [], ConfigFileParseError(
            file=file_str,
            line=exc.lineno,
            message=exc.msg or "syntax error",
        )

    source_lines = source.splitlines()
    name_map = _collect_module_constants(tree)

    visitor = _CallVisitor(
        file=file_str,
        source_lines=source_lines,
        name_map=name_map,
    )
    visitor.visit(tree)
    return visitor.entries, None


# ---------------------------------------------------------------------------
# Module-level constant resolution
# ---------------------------------------------------------------------------


def _collect_module_constants(tree: ast.AST) -> dict[str, str]:
    """Build a name -> string-constant map for module-scope assignments.

    Only resolves simple ``NAME = "literal"`` shapes. Anything more
    complex (tuple unpacking, function call, expression) is ignored
    on purpose -- the validator surfaces unresolved references as
    ``rid_unresolvable`` rather than guessing.
    """
    name_map: dict[str, str] = {}
    if not isinstance(tree, ast.Module):
        return name_map

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            value = stmt.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    name_map[target.id] = value.value
        elif isinstance(stmt, ast.AnnAssign):
            # `FOO: str = "..."` shape
            value = stmt.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            target = stmt.target
            if isinstance(target, ast.Name):
                name_map[target.id] = value.value
    return name_map


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------


class _CallVisitor(ast.NodeVisitor):
    """Walks the AST collecting :class:`ConfigEntry` objects.

    Handles wrapper unwrapping (`with_description`, `builds`,
    hydra-zen `store(<spec_class>, ...)`) and constant resolution
    (`rid=SOME_NAME` -> looks up SOME_NAME in ``name_map``).
    """

    def __init__(
        self,
        *,
        file: str,
        source_lines: list[str],
        name_map: dict[str, str],
    ) -> None:
        self.file = file
        self.source_lines = source_lines
        self.name_map = name_map
        self.entries: list[ConfigEntry] = []
        # Track which Call nodes have already been classified through
        # a wrapper so we don't double-count when generic_visit then
        # descends into them.
        self._claimed: set[int] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if id(node) in self._claimed:
            # Already counted via a wrapper; just recurse for nested calls.
            self.generic_visit(node)
            return

        kind, target_call = self._classify_call(node)
        if kind is not None and target_call is not None:
            entry = self._build_entry(kind, target_call)
            if entry is not None:
                self.entries.append(entry)
                # Mark the inner call as claimed so descending into it
                # via generic_visit doesn't double-emit.
                if target_call is not node:
                    self._claimed.add(id(target_call))
        # Recurse so nested calls are still found
        # (e.g. assets_store(spec=AssetSpecConfig(...)) -- both the outer
        # assets_store call and the inner AssetSpecConfig become candidates
        # for visit_Call; the inner one is what classifies as a known kind).
        self.generic_visit(node)

    # -- classification --------------------------------------------------

    def _classify_call(
        self,
        node: ast.Call,
    ) -> tuple[ConfigEntryKind | None, ast.Call | None]:
        """Determine if this call is a known constructor (possibly via wrapper).

        Returns ``(kind, call_node)``. ``kind`` is the spec-class name;
        ``call_node`` is the AST node whose kwargs hold the spec fields
        (which may be ``node`` itself or an inner call when wrapped).
        Returns ``(None, None)`` if this call is neither a known
        constructor nor a recognized wrapper around one.
        """
        name = _call_name(node)
        if name in _KNOWN_KINDS:
            return name, node  # type: ignore[return-value]

        if name == "builds":
            # builds(<spec_class>, **kwargs) -- treat as the spec class.
            inner_name = _builds_target_name(node)
            if inner_name in _KNOWN_KINDS:
                # builds()'s kwargs ARE the spec's kwargs (modulo the
                # first positional which names the class). So the call
                # node to read kwargs from is `node` itself.
                return inner_name, node  # type: ignore[return-value]
            # builds() target_name might also be a DatasetSpec or AssetSpec
            # (without the Config suffix) -- normalize.
            if inner_name == "DatasetSpec":
                return "DatasetSpecConfig", node
            if inner_name == "AssetSpec":
                return "AssetSpecConfig", node
            return None, None

        if name in _UNWRAP_WRAPPERS:
            # with_description(<inner_call>, "...") -- recurse on the
            # first positional arg if it's itself a Call.
            if node.args and isinstance(node.args[0], ast.Call):
                return self._classify_call(node.args[0])
            return None, None

        # Hydra-zen store-call shape: ``deriva_store(DerivaMLConfig,
        # name="...", hostname="...", ...)``. The first positional arg
        # names the spec class; the call's kwargs are the spec fields.
        # This is the canonical shape for ``configs/deriva.py`` and the
        # builds-style entries in datasets.py / assets.py / workflow.py.
        if node.args and isinstance(node.args[0], (ast.Name, ast.Attribute)):
            first = node.args[0]
            first_name = first.id if isinstance(first, ast.Name) else first.attr
            if first_name in _KNOWN_KINDS:
                return first_name, node  # type: ignore[return-value]
            if first_name == "DatasetSpec":
                return "DatasetSpecConfig", node
            if first_name == "AssetSpec":
                return "AssetSpecConfig", node

        return None, None

    # -- entry construction ---------------------------------------------

    def _build_entry(
        self,
        kind: ConfigEntryKind,
        call: ast.Call,
    ) -> ConfigEntry | None:
        """Pull the kwargs off a known constructor call into a :class:`ConfigEntry`."""
        kwargs = _kwarg_map(call)
        rid = self._resolve_str_value(kwargs.get("rid"))
        version = self._resolve_str_value(kwargs.get("version"))
        hostname = self._resolve_str_value(kwargs.get("hostname"))

        catalog_id_raw = kwargs.get("catalog_id")
        catalog_id: str | None = None
        if isinstance(catalog_id_raw, ast.Constant):
            if isinstance(catalog_id_raw.value, str):
                catalog_id = catalog_id_raw.value
            elif isinstance(catalog_id_raw.value, int):
                catalog_id = str(catalog_id_raw.value)

        cache_raw = kwargs.get("cache")
        cache: bool | None = None
        if isinstance(cache_raw, ast.Constant) and isinstance(cache_raw.value, bool):
            cache = cache_raw.value

        snippet = self._snippet_for(call)

        return ConfigEntry(
            file=self.file,
            line=call.lineno,
            col=call.col_offset,
            entry_kind=kind,
            rid=rid,
            version=version,
            hostname=hostname,
            catalog_id=catalog_id,
            cache=cache,
            snippet=snippet,
        )

    def _resolve_str_value(self, node: ast.expr | None) -> str | None:
        """Resolve a kwarg value to a string. Returns None if not resolvable.

        Handles:
        - string literal: ``"2-B4C8"``
        - module-level constant: ``TRAINING_RID``
        Anything else (function call, attribute access, expression) is
        treated as unresolvable.
        """
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            return None
        if isinstance(node, ast.Name):
            return self.name_map.get(node.id)
        return None

    def _snippet_for(self, node: ast.Call) -> str | None:
        """Return a short excerpt of the source around the call."""
        if not self.source_lines:
            return None
        idx = node.lineno - 1
        if idx < 0 or idx >= len(self.source_lines):
            return None
        return self.source_lines[idx].strip()


# ---------------------------------------------------------------------------
# Small AST utilities
# ---------------------------------------------------------------------------


def _call_name(node: ast.Call) -> str | None:
    """Return the name of the called function, or None if not a simple name.

    Handles:
    - ``Foo(...)`` -> ``"Foo"``
    - ``mod.Foo(...)`` -> ``"Foo"`` (the attribute name)
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _builds_target_name(node: ast.Call) -> str | None:
    """For ``builds(<target>, ...)``, return ``<target>``'s name.

    ``builds`` accepts the target as its first positional arg, which can
    be either a ``Name`` (the class is in scope) or an ``Attribute``
    (e.g. ``mod.Spec``).
    """
    if not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Name):
        return arg.id
    if isinstance(arg, ast.Attribute):
        return arg.attr
    return None


def _kwarg_map(node: ast.Call) -> dict[str, ast.expr]:
    """Return a {arg-name -> value-expr} map for a call's kwargs.

    For ``builds(SomeSpec, rid="X", version="Y")``, the map is
    ``{"rid": <Constant 'X'>, "version": <Constant 'Y'>}``.
    Keyword-only args (no name -- ``**kwargs``) are skipped.
    """
    out: dict[str, ast.expr] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        out[kw.arg] = kw.value
    return out
