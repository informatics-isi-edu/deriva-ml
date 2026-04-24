"""Small pandas-shaped helpers shared across catalog / bag / database code.

Keeps the pandas-coupling narrow — callers that only need the DataFrame
shape go through these helpers instead of re-inlining the
``pd.DataFrame(list(rows))`` idiom.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def rows_to_dataframe(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame from an iterable of row dicts.

    A thin wrapper over ``pd.DataFrame(list(rows))`` that centralises the
    idiom shared by ``PathBuilderMixin.get_table_as_dataframe``,
    ``DatasetBag.get_table_as_dataframe``, and
    ``DatasetDatabase.get_table_as_dataframe``. If any of them grows
    special-case handling (e.g. dtype hints, NaN normalization), this is
    the single site to evolve.

    Args:
        rows: Iterable yielding row dicts. Generators, lists, or any
            other iterable work — the helper materializes to a list
            before handing off to pandas so a partially-consumed
            generator can't produce a mismatched frame.

    Returns:
        A :class:`pandas.DataFrame` with one row per input dict and
        columns inferred from the union of dict keys (pandas' default
        behaviour for a list-of-dicts).
    """
    return pd.DataFrame(list(rows))
