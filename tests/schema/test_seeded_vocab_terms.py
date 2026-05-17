"""Tests for the term set seeded by ``initialize_ml_schema``.

Closes E5 from the e2e findings doc
(``deriva-ml-model-template/docs/findings/2026-05-16-phase-1-improvements.md``):

> "Bootstrapping ships an opinionated vocab term set."

The seeded vocabulary was carrying domain-specific terms
(``VGG19``, ``RETFound``, ``Multimodal``) inherited from an
earlier project use. Those terms have been removed; every term
that remains describes a **platform-level** concept (workflow
shape, execution state, asset role/purpose, dataset role).

Two test classes:

1. ``TestPlatformTermsPresent`` — pins the load-bearing terms.
   A future cleanup that accidentally drops one (e.g., a
   "consolidation" that removes ``Training`` from
   ``Workflow_Type``) would break every consumer expecting
   that term.

2. ``TestNoDomainSpecificTerms`` — pins the absence of the
   three removed terms and the principle behind their
   removal. A future PR that re-adds a domain-specific term
   should be caught by this test and forced to either
   (a) demonstrate the term is genuinely platform-level, or
   (b) put it in a per-project initializer instead.

These run by parsing the source of ``initialize_ml_schema`` —
no live catalog required.
"""

from __future__ import annotations

import inspect

from deriva_ml.schema.create_schema import initialize_ml_schema


def _get_initialize_source() -> str:
    """Return the source of initialize_ml_schema as a single string."""
    return inspect.getsource(initialize_ml_schema)


# ---------------------------------------------------------------------------
# Platform terms — pinned-present
# ---------------------------------------------------------------------------


class TestPlatformTermsPresent:
    """Each platform-level seeded term is still present.

    These tests use string-contains rather than a full parse so a
    rename of one term doesn't silently look like a delete (it
    would fail this test as expected).
    """

    def test_workflow_type_includes_load_bearing_terms(self) -> None:
        """Workflow_Type seeds the core operation categories."""
        src = _get_initialize_source()
        for term in (
            '"Name": "Training"',
            '"Name": "Testing"',
            '"Name": "Prediction"',
            '"Name": "Feature_Creation"',
            '"Name": "Visualization"',
            '"Name": "Analysis"',
            '"Name": "Ingest"',
            '"Name": "Data_Cleaning"',
            '"Name": "Dataset_Management"',
        ):
            assert term in src, f"Workflow_Type seed missing: {term}"

    def test_dataset_type_includes_load_bearing_terms(self) -> None:
        """Dataset_Type seeds the standard ML split labels + container types."""
        src = _get_initialize_source()
        for term in (
            '"Name": "Complete"',
            '"Name": "File"',
            '"Name": "Training"',
            '"Name": "Testing"',
            '"Name": "Validation"',
            '"Name": "Split"',
            '"Name": "Labeled"',
            '"Name": "Unlabeled"',
        ):
            assert term in src, f"Dataset_Type seed missing: {term}"

    def test_asset_role_includes_input_and_output(self) -> None:
        """Asset_Role seeds the two roles every execution recognizes."""
        src = _get_initialize_source()
        assert '"Name": "Input"' in src
        assert '"Name": "Output"' in src

    def test_execution_status_includes_state_machine_states(self) -> None:
        """Execution_Status seeds every state the lifecycle reaches."""
        src = _get_initialize_source()
        for state in (
            '"Name": "Created"',
            '"Name": "Running"',
            '"Name": "Stopped"',
            '"Name": "Failed"',
            '"Name": "Pending_Upload"',
            '"Name": "Uploaded"',
            '"Name": "Aborted"',
        ):
            assert state in src, f"Execution_Status seed missing: {state}"


# ---------------------------------------------------------------------------
# Domain-specific terms — pinned-absent
# ---------------------------------------------------------------------------


class TestNoDomainSpecificTerms:
    """The seed must not include domain-specific terms.

    A "domain-specific" term is one that's only meaningful inside a
    specific project, model family, or research community —
    ``VGG19`` (specific model architecture), ``RETFound`` (specific
    foundation model for retinal images), ``Multimodal`` (research-
    area category) are concrete examples that were removed.

    The platform-level principle is documented in
    ``initialize_ml_schema``'s docstring:

        Every term seeded here describes a platform-level concept …
        Domain-specific terms must not appear here — specific model
        architectures (VGG19, RETFound), research-area categories
        (Multimodal), or dataset/asset names tied to a single
        project all belong in user vocabularies added at the catalog
        level after schema creation.
    """

    # The match form ``'"Name": "<term>"'`` matches the exact shape
    # of a seeded term dict (``{"Name": "...", "Description": "..."}``)
    # and not bare mentions of the term in the function's docstring
    # (which intentionally names the removed terms as anti-examples).

    def test_no_vgg19_seeded(self) -> None:
        """``VGG19`` is a specific neural-network architecture — not platform-level."""
        src = _get_initialize_source()
        assert '"Name": "VGG19"' not in src, (
            "VGG19 is a specific neural-network architecture from a 2014 "
            "paper. It belongs in a per-project vocabulary, not in the "
            "platform's default Workflow_Type set. See "
            "initialize_ml_schema's docstring."
        )

    def test_no_retfound_seeded(self) -> None:
        """``RETFound`` is a specific retinal-imaging foundation model — not platform-level."""
        src = _get_initialize_source()
        assert '"Name": "RETFound"' not in src, (
            "RETFound is a specific retinal-imaging foundation model "
            "from a 2023 paper. It belongs in a per-project vocabulary, "
            "not in the platform's default Workflow_Type set. See "
            "initialize_ml_schema's docstring."
        )

    def test_no_multimodal_seeded(self) -> None:
        """``Multimodal`` is a research-area category — not a workflow shape."""
        src = _get_initialize_source()
        assert '"Name": "Multimodal"' not in src, (
            "Multimodal is a research-area category ('workflows "
            "combining multiple modalities'), not a workflow shape. "
            "Multimodal workflows are structurally Training / "
            "Prediction / Feature_Creation. See initialize_ml_schema's "
            "docstring."
        )

    def test_docstring_documents_platform_only_principle(self) -> None:
        """The function's docstring records why domain terms are excluded.

        Future readers and reviewers should land on the principle
        first — a docstring that lost the explanation would let a
        future PR slip in a new domain term without anyone
        challenging it.
        """
        assert initialize_ml_schema.__doc__ is not None
        doc = initialize_ml_schema.__doc__.lower()
        assert "platform-level" in doc, (
            "initialize_ml_schema's docstring must record the "
            "platform-only principle for seeded terms — otherwise "
            "the next reviewer has no guardrail against a new "
            "domain term being added."
        )
        # Verify the three example removed terms are named explicitly
        # in the docstring as anti-examples.
        for example in ("vgg19", "retfound", "multimodal"):
            assert example in doc, (
                f"The docstring should call out '{example}' as an "
                f"example of a domain-specific term that doesn't "
                f"belong in the platform defaults. Without the "
                f"anti-examples, a future PR could add a "
                f"similar-shaped term and pass review."
            )
