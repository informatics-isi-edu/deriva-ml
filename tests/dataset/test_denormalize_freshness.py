"""Server-side mutation freshness tests for denormalize (spec §8 C.5x).

The denormalize subsystem's local SQLite cache is **write-through-only**:
``_insert_rows`` uses ``INSERT OR IGNORE`` and never issues a ``DELETE``
or ``UPDATE`` to reconcile against the server. A server-side row that
was present at the time of the first denormalize call and is then
deleted will **still appear** in the result of a subsequent call made
by the same ``Denormalizer`` instance.

Spec §6 Freshness caveat and §7 row F3/F4 document the limitation;
spec §8 row C.5x promises an ``xfail`` test that makes the limitation
visible. The audit's TC-04 finding noted the spec promise wasn't
backed by a test. This file lands that test.

The test is marked ``xfail(strict=False)`` because the assertion that
the deleted row is gone is **expected to fail** today — that's the
limitation. If a future change adds cache invalidation, the test will
``xpass`` and we can flip it to a real positive assertion.

See:
- ``docs/audits/2026-05-26-denormalize-audit.md`` — TC-04.
- ``docs/design/denormalization.md`` §6 Freshness caveat, §7 F3/F4, §8
  row C.5x.
"""

from __future__ import annotations

import pytest

from deriva_ml import DerivaML


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Known freshness limitation: the local SQLite cache is "
        "write-through-only and does NOT observe server-side deletions. "
        "See spec §6 Freshness caveat + §7 F3/F4. Audit finding TC-04 "
        "(2026-05-26-denormalize-audit.md). xfail makes the limitation "
        "visible to anyone reading the test suite."
    ),
)
def test_denormalize_does_not_observe_server_side_delete(dataset_test, tmp_path):
    """Server-side delete between two denormalize calls is NOT observed.

    Procedure:

    1. Live-catalog denormalize on a small dataset; capture the
       baseline Subject RID set.
    2. Delete one Subject membership from the dataset server-side via
       ``Dataset.delete_dataset_members``.
    3. Re-denormalize **on the same** ``Dataset`` instance.
    4. Assert the deleted Subject is **not** in the new result — this
       is what we'd want; the limitation is that it *will* still
       appear. The xfail marker captures the gap; spec §6 commits to
       documenting (not fixing) this today.

    The test is intentionally one-shot — it doesn't probe Denormalizer
    internals. The point is the contract surface: from a caller's
    perspective, denormalize-after-delete still returns the deleted
    row. That's what xfail makes visible.
    """
    ml = DerivaML(
        dataset_test.catalog.hostname,
        dataset_test.catalog.catalog_id,
        working_dir=tmp_path,
        use_minid=False,
    )

    dataset = dataset_test.dataset_description.dataset

    # First denormalize: capture the baseline Subject RID set.
    members_before = dataset.list_dataset_members(recurse=False)
    subject_members = members_before.get("Subject", [])
    if len(subject_members) < 2:
        pytest.skip(
            "Freshness test needs ≥2 Subject members in the dataset; "
            "demo fixture changed?"
        )

    df_before = dataset.get_denormalized_as_dataframe(
        include_tables=["Subject"],
        ignore_unrelated_anchors=True,
    )
    subjects_before = set(df_before["Subject.RID"].dropna())
    assert len(subjects_before) >= 2, (
        "Sanity check: first denormalize should return at least 2 "
        f"Subjects; got {subjects_before}."
    )

    # Pick one Subject to delete from the dataset (we delete the
    # membership row in the association table — the Subject record
    # itself remains, but it's no longer part of the dataset).
    victim_rid = next(iter(subjects_before))

    # Delete the membership server-side via the public API.
    dataset.delete_dataset_members(members=[victim_rid])

    # Re-denormalize on the SAME Dataset instance — this is the path
    # whose cache the freshness limitation affects. Fresh Denormalizer
    # instances are out of scope; the limitation is specifically about
    # the long-lived process pattern §6 names.
    df_after = dataset.get_denormalized_as_dataframe(
        include_tables=["Subject"],
        ignore_unrelated_anchors=True,
    )
    subjects_after = set(df_after["Subject.RID"].dropna())

    # The behavior we'd want: the deleted Subject is gone. The xfail
    # marker captures that this assertion fails today — the cache is
    # not invalidated, so the deleted row reappears in the result.
    assert victim_rid not in subjects_after, (
        f"Freshness limitation surfaced: deleted Subject {victim_rid!r} "
        f"still appears in the denormalize result. The write-through "
        f"cache did not observe the server-side delete. Spec §6 "
        f"acknowledges this; TC-04 names the missing xfail."
    )
