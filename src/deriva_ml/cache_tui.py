"""Interactive TUI for managing DerivaML cache and working directories.

Provides a terminal UI for browsing and cleaning up ~/.deriva-ml/ contents
including cached dataset bags, execution directories, hydra configs, and
other artifacts.

Usage:
    deriva-ml-storage          # Launch interactive TUI
    deriva-ml-storage --list   # Non-interactive listing
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Static,
)


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.1f} {units[i]}"


def _dir_size(path: Path) -> int:
    """Calculate total size of a directory."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def _dir_file_count(path: Path) -> int:
    """Count files in a directory."""
    try:
        return sum(1 for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def _dir_mtime(path: Path) -> datetime | None:
    """Get modification time of a directory."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def _cache_entry_version(cache_entry_path: Path, rid: str) -> str | None:
    """Extract dataset version from a cached bag's Dataset.csv."""
    import csv

    bag_dir = cache_entry_path / f"Dataset_{rid}"
    dataset_csv = bag_dir / "data" / "Dataset.csv"
    if dataset_csv.exists():
        try:
            with dataset_csv.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("RID") == rid:
                        return row.get("Version", None) or None
        except Exception:
            pass
    return None


class DirectoryEntry:
    """Represents a deletable directory entry."""

    def __init__(
        self,
        path: Path,
        label: str,
        category: str,
        size_bytes: int,
        item_count: int,
        modified: datetime | None,
        parent_label: str = "",
        rid: str | None = None,
    ):
        self.path = path
        self.label = label
        self.category = category
        self.size_bytes = size_bytes
        self.item_count = item_count
        self.modified = modified
        self.parent_label = parent_label
        self.rid = rid
        self.selected = False

    @property
    def size(self) -> str:
        return _human_readable_size(self.size_bytes)

    @property
    def modified_str(self) -> str:
        if self.modified is None:
            return ""
        return self.modified.strftime("%Y-%m-%d %H:%M")

    def to_dict(self) -> dict:
        """Convert to a plain dict for JSON serialization."""
        return {
            "rid": self.rid,
            "label": self.label,
            "location": self.parent_label,
            "category": self.category,
            "size_bytes": self.size_bytes,
            "size": self.size,
            "item_count": self.item_count,
            "modified": self.modified.isoformat() if self.modified else None,
            "path": str(self.path),
        }


def discover_entries() -> list[DirectoryEntry]:
    """Discover all deletable entries in ~/.deriva-ml/."""
    entries: list[DirectoryEntry] = []
    default_root = Path.home() / ".deriva-ml"

    if not default_root.exists():
        return entries

    def _scan_workdir(workdir: Path, parent_label: str) -> None:
        """Scan a working directory for deletable entries."""
        if not workdir.is_dir():
            return

        for entry in sorted(workdir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            # Skip snapshot dirs
            if "@" in entry.name:
                continue

            size = _dir_size(entry)
            mtime = _dir_mtime(entry)
            file_count = _dir_file_count(entry)

            if entry.name == "cache":
                # Collect cache entries, then disambiguate duplicate RIDs
                cache_items: list[tuple[Path, str]] = []
                rid_counts: dict[str, int] = {}
                for cache_entry in sorted(entry.iterdir()):
                    if cache_entry.is_dir() and "_" in cache_entry.name:
                        rid = cache_entry.name.rsplit("_", 1)[0]
                        cache_items.append((cache_entry, rid))
                        rid_counts[rid] = rid_counts.get(rid, 0) + 1

                for cache_entry, rid in cache_items:
                    ce_size = _dir_size(cache_entry)
                    ce_mtime = _dir_mtime(cache_entry)

                    # Disambiguate if multiple entries share the same RID
                    if rid_counts[rid] > 1:
                        version = _cache_entry_version(cache_entry, rid)
                        if version:
                            label = f"{rid} (v{version})"
                        elif ce_mtime:
                            label = f"{rid} ({ce_mtime.strftime('%Y-%m-%d %H:%M')})"
                        else:
                            label = f"{rid}"
                    else:
                        label = rid

                    entries.append(DirectoryEntry(
                        path=cache_entry,
                        label=label,
                        category="dataset",
                        size_bytes=ce_size,
                        item_count=_dir_file_count(cache_entry),
                        modified=ce_mtime,
                        parent_label=parent_label,
                        rid=rid,
                    ))
            elif entry.name == "deriva-ml":
                exec_dir = entry / "execution"
                if exec_dir.exists():
                    for exec_entry in sorted(exec_dir.iterdir()):
                        if exec_entry.is_dir():
                            ex_size = _dir_size(exec_entry)
                            ex_mtime = _dir_mtime(exec_entry)
                            entries.append(DirectoryEntry(
                                path=exec_entry,
                                label=exec_entry.name,
                                category="execution",
                                size_bytes=ex_size,
                                item_count=_dir_file_count(exec_entry),
                                modified=ex_mtime,
                                parent_label=parent_label,
                                rid=exec_entry.name,
                            ))
                # If there's nothing in execution but the dir exists, offer the whole thing
                if size > 0 and not (exec_dir.exists() and any(exec_dir.iterdir())):
                    entries.append(DirectoryEntry(
                        path=entry,
                        label="deriva-ml/",
                        category="execution",
                        size_bytes=size,
                        item_count=file_count,
                        modified=mtime,
                        parent_label=parent_label,
                    ))
            else:
                entries.append(DirectoryEntry(
                    path=entry,
                    label=entry.name,
                    category="other",
                    size_bytes=size,
                    item_count=file_count,
                    modified=mtime,
                    parent_label=parent_label,
                ))

    # Scan two levels: hostname/catalog_id
    for host_dir in sorted(default_root.iterdir()):
        if not host_dir.is_dir() or host_dir.name.startswith("."):
            continue

        has_catalog_subdirs = False
        for catalog_dir in sorted(host_dir.iterdir()):
            if not catalog_dir.is_dir() or "@" in catalog_dir.name:
                continue

            # Check if this looks like a catalog working dir
            is_workdir = any(
                (catalog_dir / sub).exists()
                for sub in ["cache", "deriva-ml", "hydra", "hydra-sweep", "client_export"]
            )

            if is_workdir:
                has_catalog_subdirs = True
                label = f"{host_dir.name}/{catalog_dir.name}"
                _scan_workdir(catalog_dir, label)

        # Also check if host_dir itself is a working dir (e.g., ~/.deriva-ml/eye-ai/)
        if not has_catalog_subdirs:
            is_workdir = any(
                (host_dir / sub).exists()
                for sub in ["cache", "deriva-ml", "hydra", "hydra-sweep", "client_export"]
            ) or any(
                "_" in child.name and child.is_dir() and len(child.name) > 40
                for child in host_dir.iterdir()
                if not child.name.startswith(".")
            )
            if is_workdir:
                _scan_workdir(host_dir, host_dir.name)

    return entries


class SelectionSummary(Static):
    """Widget showing the current selection summary."""

    def update_summary(self, entries: list[DirectoryEntry]) -> None:
        selected = [e for e in entries if e.selected]
        count = len(selected)
        total = sum(e.size_bytes for e in selected)
        if count == 0:
            self.update("No items selected")
        else:
            self.update(
                f"[bold]{count}[/bold] selected  |  "
                f"[bold red]{_human_readable_size(total)}[/bold red] to free"
            )


class CacheTUI(App):
    """Interactive TUI for managing DerivaML storage."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #summary-bar {
        height: 3;
        background: $surface;
        padding: 0 2;
        content-align: center middle;
    }

    #summary-bar Label {
        width: 1fr;
        text-align: center;
    }

    #table-container {
        height: 1fr;
    }

    DataTable {
        height: 1fr;
    }

    #status-bar {
        height: 1;
        background: $surface;
        padding: 0 2;
    }
    """

    BINDINGS = [
        Binding("space", "toggle_select", "Toggle select"),
        Binding("a", "select_all", "Select all"),
        Binding("n", "select_none", "Clear selection"),
        Binding("d", "delete_selected", "Delete selected"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[DirectoryEntry] = []
        self._entry_map: dict[str, DirectoryEntry] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Horizontal(id="summary-bar"):
                yield SelectionSummary(id="selection-summary")
            with VerticalScroll(id="table-container"):
                yield DataTable(id="dir-table")
            yield Label("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "DerivaML Storage Manager"
        self.sub_title = "~/.deriva-ml"

        table = self.query_one("#dir-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("", "Location", "RID", "Type", "Size", "Items", "Modified")

        # Load entries
        status = self.query_one("#status-bar", Label)
        status.update("Scanning directories...")
        self.entries = discover_entries()

        # Sort by size descending
        self.entries.sort(key=lambda e: e.size_bytes, reverse=True)

        for entry in self.entries:
            key = str(entry.path)
            self._entry_map[key] = entry
            table.add_row(
                "  ",
                entry.parent_label,
                entry.label,
                entry.category,
                entry.size,
                str(entry.item_count),
                entry.modified_str,
                key=key,
            )

        total_size = sum(e.size_bytes for e in self.entries)
        status.update(
            f"{len(self.entries)} entries  |  {_human_readable_size(total_size)} total"
        )
        self._update_summary()

    def _update_summary(self) -> None:
        summary = self.query_one("#selection-summary", SelectionSummary)
        summary.update_summary(self.entries)

    def _update_row_checkbox(self, key: str) -> None:
        entry = self._entry_map[key]
        table = self.query_one("#dir-table", DataTable)
        marker = "[bold green]\u2713[/]" if entry.selected else "  "
        table.update_cell(key, table.columns[list(table.columns.keys())[0]].key, marker)

    def _get_current_row_key(self) -> str | None:
        """Get the key of the currently selected row."""
        table = self.query_one("#dir-table", DataTable)
        if table.cursor_row is None:
            return None
        ordered = table.ordered_rows
        if table.cursor_row < len(ordered):
            return str(ordered[table.cursor_row].key.value)
        return None

    def action_toggle_select(self) -> None:
        key = self._get_current_row_key()
        if key is None or key not in self._entry_map:
            return
        entry = self._entry_map[key]
        entry.selected = not entry.selected
        self._update_row_checkbox(key)
        self._update_summary()

    def action_select_all(self) -> None:
        for key, entry in self._entry_map.items():
            entry.selected = True
            self._update_row_checkbox(key)
        self._update_summary()

    def action_select_none(self) -> None:
        for key, entry in self._entry_map.items():
            entry.selected = False
            self._update_row_checkbox(key)
        self._update_summary()

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Toggle selection on Enter/click."""
        key = str(event.row_key.value)
        if key in self._entry_map:
            entry = self._entry_map[key]
            entry.selected = not entry.selected
            self._update_row_checkbox(key)
            self._update_summary()

    def action_delete_selected(self) -> None:
        selected = [e for e in self.entries if e.selected]
        if not selected:
            status = self.query_one("#status-bar", Label)
            status.update("[yellow]Nothing selected to delete[/]")
            return

        total = sum(e.size_bytes for e in selected)
        self.push_screen(
            ConfirmDeleteScreen(selected, total),
            callback=self._handle_delete_result,
        )

    def _handle_delete_result(self, confirmed: bool) -> None:
        if not confirmed:
            status = self.query_one("#status-bar", Label)
            status.update("Delete cancelled")
            return

        selected = [e for e in self.entries if e.selected]
        deleted = 0
        errors = 0
        freed = 0

        for entry in selected:
            try:
                shutil.rmtree(entry.path)
                freed += entry.size_bytes
                deleted += 1
            except Exception:
                errors += 1

        # Remove deleted entries from the table and lists
        table = self.query_one("#dir-table", DataTable)
        keys_to_remove = []
        for entry in selected:
            key = str(entry.path)
            if entry.path.exists():
                continue  # Failed to delete
            keys_to_remove.append(key)

        for key in keys_to_remove:
            try:
                table.remove_row(key)
            except Exception:
                pass
            if key in self._entry_map:
                self.entries.remove(self._entry_map[key])
                del self._entry_map[key]

        # Clean up empty parent dirs
        cleaned_parents = set()
        for entry in selected:
            if not entry.path.exists():
                parent = entry.path.parent
                # Walk up cleaning empty dirs, but stop at ~/.deriva-ml
                deriva_root = Path.home() / ".deriva-ml"
                while parent != deriva_root and parent.exists():
                    try:
                        if not any(parent.iterdir()):
                            parent.rmdir()
                            cleaned_parents.add(parent)
                            parent = parent.parent
                        else:
                            break
                    except OSError:
                        break

        status = self.query_one("#status-bar", Label)
        msg = f"[green]Deleted {deleted} entries, freed {_human_readable_size(freed)}[/]"
        if errors:
            msg += f"  [red]({errors} errors)[/]"
        if cleaned_parents:
            msg += f"  [dim]({len(cleaned_parents)} empty dirs removed)[/]"
        status.update(msg)
        self._update_summary()


class ConfirmDeleteScreen(ModalScreen[bool]):
    """Confirmation dialog for deletion."""

    CSS = """
    ConfirmDeleteScreen {
        align: center middle;
    }

    #dialog {
        width: 60;
        height: auto;
        max-height: 20;
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }

    #dialog Label {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }

    #buttons {
        height: 3;
        align: center middle;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, entries: list[DirectoryEntry], total_bytes: int) -> None:
        super().__init__()
        self.del_entries = entries
        self.total_bytes = total_bytes

    def compose(self) -> ComposeResult:
        count = len(self.del_entries)
        size = _human_readable_size(self.total_bytes)

        with Vertical(id="dialog"):
            yield Label(f"[bold red]Delete {count} entries ({size})?[/]")
            yield Label("This cannot be undone.")
            with Horizontal(id="buttons"):
                yield Button("Delete", variant="error", id="confirm")
                yield Button("Cancel", variant="default", id="cancel")

    @on(Button.Pressed, "#confirm")
    def on_confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(False)


def main() -> None:
    """Entry point for the cache TUI."""
    import argparse

    parser = argparse.ArgumentParser(description="DerivaML storage manager")
    parser.add_argument(
        "--list", nargs="?", const="all", choices=["cache", "executions", "all"],
        help="Non-interactive listing (cache, executions, or all). Defaults to all.",
    )
    parser.add_argument(
        "--delete", metavar="RID", nargs="+",
        help="Delete entries matching dataset or execution RID(s)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt (use with --delete)",
    )
    args = parser.parse_args()

    if args.delete:
        _cli_delete(args.delete, confirm=args.yes)
    elif args.list is not None:
        _cli_list(args.list)
    else:
        app = CacheTUI()
        app.run()


def _cli_list(filter_type: str = "all") -> None:
    """Non-interactive listing of storage entries."""
    entries = discover_entries()

    if filter_type == "cache":
        entries = [e for e in entries if e.category == "dataset"]
    elif filter_type == "executions":
        entries = [e for e in entries if e.category == "execution"]

    entries.sort(key=lambda e: e.size_bytes, reverse=True)
    total = sum(e.size_bytes for e in entries)

    current_parent = ""
    for entry in entries:
        if entry.parent_label != current_parent:
            current_parent = entry.parent_label
            print(f"\n=== {current_parent} ===")
        print(
            f"  {entry.label:40s} {entry.category:12s} "
            f"{entry.size:>10s}  ({entry.item_count} items)  {entry.modified_str}"
        )
    print(f"\nTotal: {_human_readable_size(total)} across {len(entries)} entries")


def _cli_delete(rids: list[str], confirm: bool = False) -> None:
    """Delete entries matching the given RIDs."""
    entries = discover_entries()

    # Match by RID field
    rid_set = set(rids)
    matches = [e for e in entries if e.rid in rid_set]

    if not matches:
        print(f"No entries found matching: {', '.join(rids)}")
        return

    total = sum(e.size_bytes for e in matches)
    print(f"Found {len(matches)} matching entries ({_human_readable_size(total)}):\n")
    for entry in matches:
        print(
            f"  {entry.parent_label:30s} {entry.label:40s} "
            f"{entry.category:12s} {entry.size:>10s}"
        )

    if not confirm:
        response = input(f"\nDelete {len(matches)} entries? [y/N] ").strip().lower()
        if response != "y":
            print("Cancelled.")
            return

    deleted = 0
    freed = 0
    for entry in matches:
        try:
            shutil.rmtree(entry.path)
            freed += entry.size_bytes
            deleted += 1
        except Exception as e:
            print(f"  Error deleting {entry.path}: {e}")

    # Clean up empty parent dirs
    deriva_root = Path.home() / ".deriva-ml"
    for entry in matches:
        if not entry.path.exists():
            parent = entry.path.parent
            while parent != deriva_root and parent.exists():
                try:
                    if not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                    else:
                        break
                except OSError:
                    break

    print(f"\nDeleted {deleted} entries, freed {_human_readable_size(freed)}")


if __name__ == "__main__":
    main()
