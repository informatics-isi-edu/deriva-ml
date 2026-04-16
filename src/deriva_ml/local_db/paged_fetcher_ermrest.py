"""ERMrest adapter for :class:`PagedFetcher`.

Translates the narrow ``PagedClient`` protocol into ERMrest HTTP calls via
an ``ErmrestCatalog`` handle. Responsible for URL construction, GET
transport, and JSON response parsing.

**Path convention:** ``ErmrestCatalog.get(path)`` already prepends the
server URI and catalog prefix (``https://host/ermrest/catalog/{N}``), so
all paths here are **relative to the catalog root** — they start with
``/aggregate/...``, ``/entity/...``, etc.

URL forms used (relative to catalog root):

- count:   ``/aggregate/{schema}:{table}/n:=cnt(*)``
- page:    ``/entity/{schema}:{table}[/predicate]@sort({s})[@after({a})]?limit={L}``
- RID-IN (GET):  ``/entity/{schema}:{table}/{col}=any({r1,r2,...})``

Note: POST-based filtering is **not** supported — ``POST /entity/{table}``
in ERMrest is an entity-insert endpoint, not a filter query. For large RID
sets that would exceed GET URL limits, use
``fetch_by_rids_or_predicate`` with a predicate-scan fallback instead.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


class ErmrestPagedClient:
    """Adapter conforming to :class:`~deriva_ml.local_db.paged_fetcher.PagedClient`."""

    def __init__(self, *, catalog: Any, catalog_id: str | None = None) -> None:
        self._catalog = catalog
        self._catalog_id = str(catalog_id if catalog_id is not None else getattr(catalog, "catalog_id"))

    def count(self, table: str) -> int:
        url = f"/aggregate/{table}/n:=cnt(*)"
        resp = self._catalog.get(url)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return 0
        return int(data[0]["n"])

    def fetch_page(
        self,
        table: str,
        sort: tuple[str, ...],
        after: tuple | None,
        predicate: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        parts = [f"/entity/{table}"]
        if predicate:
            parts.append(f"/{predicate}")
        sort_cols = ",".join(quote(c) for c in sort)
        parts.append(f"@sort({sort_cols})")
        if after is not None:
            after_str = ",".join(quote(str(v)) for v in after)
            parts.append(f"@after({after_str})")
        parts.append(f"?limit={limit}")
        url = "".join(parts)
        resp = self._catalog.get(url)
        resp.raise_for_status()
        return list(resp.json())

    def fetch_rid_batch(
        self,
        table: str,
        column: str,
        rids: list[str],
        method: str = "GET",
    ) -> list[dict[str, Any]]:
        if method != "GET":
            raise RuntimeError(
                "POST-based RID filtering is not supported by ERMrest. "
                "Use fetch_by_rids_or_predicate with a predicate scan fallback "
                "for large RID sets that exceed GET URL limits."
            )
        rid_list = ",".join(quote(r) for r in rids)
        url = f"/entity/{table}/{quote(column)}=any({rid_list})"
        if len(url) > 7000:
            raise RuntimeError(f"GET URL too long ({len(url)} bytes)")
        resp = self._catalog.get(url)
        resp.raise_for_status()
        return list(resp.json())
