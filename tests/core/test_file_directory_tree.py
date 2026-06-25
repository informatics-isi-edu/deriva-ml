"""Unit tests for add_files' directory-tree building (pure path math).

These exercise the helper that turns the set of source directories (parsed from
File URLs) into a single-root containment tree, WITHOUT a catalog. The key
regression here is the relative-path hang: a relative source path parses to a
``//``-prefixed Path, whose leading double-slash ``commonpath`` collapses to a
single slash but ``Path.parent`` preserves — so the old ancestor walk never
reached the root and looped forever. The helper must canonicalize so every case
terminates with one connected root.
"""

from __future__ import annotations

from pathlib import Path

from deriva_ml.core.mixins.file import _directory_tree


class TestDirectoryTree:
    def test_relative_paths_do_not_hang_and_form_one_root(self):
        """Relative source dirs (``//``-prefixed after URL parsing) must build a
        single-root tree, not loop forever. This is the v1.51.13 hang bug."""
        # These are the directory Paths as add_files derives them from relative
        # tag-URL file paths (urlsplit(...).path of a relative file:// URL).
        dirs = [Path("//data/a"), Path("//data/b")]
        ingest_root, nodes = _directory_tree(dirs)

        # One connected root that is an ancestor of every node.
        assert all(node == ingest_root or ingest_root in node.parents for node in nodes), (
            "every node must descend from the single ingest root"
        )
        # The two leaf dirs resolve to 'a' and 'b' relative to the common root.
        leaf_rels = {node.relative_to(ingest_root).as_posix() for node in nodes if node != ingest_root}
        assert {"a", "b"} <= leaf_rels

    def test_absolute_paths_still_form_one_root(self):
        """Absolute source dirs (the common case) keep working."""
        dirs = [Path("/tmp/base/a"), Path("/tmp/base/b")]
        ingest_root, nodes = _directory_tree(dirs)
        assert all(node == ingest_root or ingest_root in node.parents for node in nodes)
        leaf_rels = {node.relative_to(ingest_root).as_posix() for node in nodes if node != ingest_root}
        assert {"a", "b"} <= leaf_rels

    def test_single_directory_is_its_own_root(self):
        """A single source directory is the ingest root itself."""
        dirs = [Path("//data/only")]
        ingest_root, nodes = _directory_tree(dirs)
        assert ingest_root in nodes
        # No node fails to relate to the root.
        assert all(node == ingest_root or ingest_root in node.parents for node in nodes)

    def test_deeply_nested_forest_converges(self):
        """A forest whose common ancestor holds no files still converges on one
        root, and terminates (no hang) even for relative ``//`` paths."""
        dirs = [Path("//d/a/x"), Path("//d/b/y")]
        ingest_root, nodes = _directory_tree(dirs)
        # Common ancestor 'd' is the root; intermediates a, b are synthesized.
        assert all(node == ingest_root or ingest_root in node.parents for node in nodes)
        rels = {node.relative_to(ingest_root).as_posix() for node in nodes if node != ingest_root}
        assert {"a", "a/x", "b", "b/y"} <= rels
