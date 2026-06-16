"""Tests for ``deriva_ml.execution.rid_lease.LeaseAggregator`` (audit P1 Ex-batch).

Pre-extraction, ``bag_commit`` made three separate
``post_lease_batch`` calls per commit — one for the Execution
association, one for Asset_Type pairs, one for each feature
group. The aggregator collapses all reservations into a single
POST at commit time.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

import pytest

from deriva_ml.execution.rid_lease import LeaseAggregator


class TestReserveAndResolve:
    """Tokens reserved on the aggregator survive flush + resolve."""

    def test_reserve_returns_unique_tokens(self):
        agg = LeaseAggregator()
        tokens = agg.reserve(5)
        assert len(tokens) == 5
        assert len(set(tokens)) == 5  # uniqueness

    def test_multiple_reserves_accumulate(self):
        """Tokens from multiple ``reserve`` calls land on the same aggregator."""
        agg = LeaseAggregator()
        a = agg.reserve(2)
        b = agg.reserve(3)
        # Internal token list is the concatenation in call order.
        assert agg._tokens == a + b  # noqa: SLF001

    def test_reserve_zero_is_a_noop(self):
        """``reserve(0)`` returns an empty list and adds no tokens."""
        agg = LeaseAggregator()
        result = agg.reserve(0)
        assert result == []
        assert agg._tokens == []  # noqa: SLF001

    def test_reserve_negative_raises(self):
        agg = LeaseAggregator()
        with pytest.raises(ValueError, match="n >= 0"):
            agg.reserve(-1)

    def test_resolve_before_flush_raises(self):
        """``resolve(t)`` without a prior ``flush()`` is a programming error."""
        agg = LeaseAggregator()
        token = agg.reserve(1)[0]
        with pytest.raises(RuntimeError, match="before flush"):
            agg.resolve(token)


class TestFlush:
    """``flush()`` issues exactly one ``post_lease_batch`` POST."""

    def test_flush_posts_all_reserved_tokens_in_one_call(self, monkeypatch):
        """One flush → one ``post_lease_batch`` call with the union of tokens."""
        calls: list[list[str]] = []

        def _fake_post(*, catalog, tokens):
            calls.append(list(tokens))
            return {t: f"LEASED-{i}" for i, t in enumerate(tokens)}

        monkeypatch.setattr("deriva_ml.execution.rid_lease.post_lease_batch", _fake_post)

        agg = LeaseAggregator()
        agg.reserve(2)
        agg.reserve(3)
        agg.reserve(1)

        result = agg.flush(catalog=object())

        # Exactly one POST.
        assert len(calls) == 1
        # All 6 reserved tokens went together.
        assert len(calls[0]) == 6
        assert len(result) == 6

    def test_flush_then_resolve_returns_leased_rids(self, monkeypatch):
        """After flush, every reserved token resolves to its leased RID."""
        monkeypatch.setattr(
            "deriva_ml.execution.rid_lease.post_lease_batch",
            lambda *, catalog, tokens: {t: f"RID-{i}" for i, t in enumerate(tokens)},
        )
        agg = LeaseAggregator()
        tokens = agg.reserve(3)
        agg.flush(catalog=object())
        rids = [agg.resolve(t) for t in tokens]
        assert rids == ["RID-0", "RID-1", "RID-2"]

    def test_flush_empty_aggregator_is_a_noop(self, monkeypatch):
        """Flushing without any reserve calls returns an empty map.

        Used in commit paths where the aggregator is unconditionally
        flushed but the commit might have nothing to lease (e.g.
        zero pending entries).
        """
        called = []
        monkeypatch.setattr(
            "deriva_ml.execution.rid_lease.post_lease_batch",
            lambda *, catalog, tokens: called.append(tokens) or {},
        )
        agg = LeaseAggregator()
        result = agg.flush(catalog=object())
        assert result == {}
        # ``post_lease_batch`` itself short-circuits on empty tokens,
        # but the aggregator should call through with the empty list.
        assert called == [[]]

    def test_double_flush_raises(self, monkeypatch):
        monkeypatch.setattr(
            "deriva_ml.execution.rid_lease.post_lease_batch",
            lambda *, catalog, tokens: {},
        )
        agg = LeaseAggregator()
        agg.flush(catalog=object())
        with pytest.raises(RuntimeError, match="twice"):
            agg.flush(catalog=object())

    def test_reserve_after_flush_raises(self, monkeypatch):
        """Reserving after flush would leave the new token un-POSTed."""
        monkeypatch.setattr(
            "deriva_ml.execution.rid_lease.post_lease_batch",
            lambda *, catalog, tokens: {},
        )
        agg = LeaseAggregator()
        agg.flush(catalog=object())
        with pytest.raises(RuntimeError, match="after flush"):
            agg.reserve(1)

    def test_resolve_unknown_token_raises_keyerror(self, monkeypatch):
        monkeypatch.setattr(
            "deriva_ml.execution.rid_lease.post_lease_batch",
            lambda *, catalog, tokens: {t: f"RID-{i}" for i, t in enumerate(tokens)},
        )
        agg = LeaseAggregator()
        agg.reserve(1)
        agg.flush(catalog=object())
        with pytest.raises(KeyError):
            agg.resolve("not-a-real-token")
