"""ERMrest adapter for :class:`PagedFetcher`.

Translates the narrow ``PagedClient`` protocol into ERMrest HTTP calls via
an ``ErmrestCatalog`` handle. Responsible for URL construction, GET/POST
transport choice, and JSON response parsing.

URL forms used:

- count:   ``/ermrest/catalog/{N}/aggregate/{schema}:{table}/n:=cnt(*)``
- page:    ``/ermrest/catalog/{N}/entity/{schema}:{table}[/predicate]@sort({s})[@after({a})]?limit={L}``
- RID-IN (GET):  ``/ermrest/catalog/{N}/entity/{schema}:{table}/{col}=any({r1,r2,...})``
- RID-IN (POST): ``POST /ermrest/catalog/{N}/entity/{schema}:{table}`` with a
  JSON body describing the filter.
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
        url = f"/ermrest/catalog/{self._catalog_id}/aggregate/{table}/n:=cnt(*)"
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
        parts = [f"/ermrest/catalog/{self._catalog_id}/entity/{table}"]
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
        if method == "GET":
            rid_list = ",".join(quote(r) for r in rids)
            url = f"/ermrest/catalog/{self._catalog_id}/entity/{table}/{quote(column)}=any({rid_list})"
            if len(url) > 7000:
                raise RuntimeError(f"GET URL too long ({len(url)} bytes)")
            resp = self._catalog.get(url)
            resp.raise_for_status()
            return list(resp.json())
        # POST fallback
        url = f"/ermrest/catalog/{self._catalog_id}/entity/{table}"
        body = {"filter": {"and": [{column: {"in": rids}}]}}
        resp = self._catalog.post(url, json=body)
        resp.raise_for_status()
        return list(resp.json())
