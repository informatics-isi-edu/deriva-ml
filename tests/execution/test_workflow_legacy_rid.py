"""Unit tests for the Workflow legacy ``rid`` -> ``workflow_rid`` normalizer.

Background: pre-#226 ``configuration.json`` files serialize the workflow
substructure with a ``rid`` key (the old name for ``workflow_rid``). The
current ``Workflow`` model renamed the field and sets ``extra="forbid"``
to catch the #226 silent-drop regression, which makes those legacy
configs fail to parse with ``rid -- extra_forbidden``. A
``model_validator(mode="before")`` maps the legacy key to the new field
name WITHOUT relaxing ``extra="forbid"`` (so genuinely-unknown fields
still raise).

These are pure model-construction tests -- no catalog required.
"""

import pytest
from pydantic import ValidationError

from deriva_ml.execution.workflow import Workflow

# The exact legacy workflow substructure observed in a real
# dev.eye-ai.org configuration.json (note the ``rid`` key):
_LEGACY_WORKFLOW = {
    "name": "API Workflow",
    "url": "https://github.com/informatics-isi-edu/deriva-ml/blob/62d5dc2--/src/deriva_ml/demo_catalog.py",
    "workflow_type": "Feature Notebook Workflow",
    "version": None,
    "description": "",
    "rid": None,
    "checksum": "793f7dadfa14ecc227d353e400cebc449bd2ec94",
}


def test_legacy_rid_key_parses_and_maps_to_workflow_rid():
    """A legacy config with ``rid`` parses; the value lands on ``workflow_rid``."""
    wf = Workflow.model_validate(_LEGACY_WORKFLOW)
    assert wf.name == "API Workflow"
    assert wf.workflow_rid is None  # legacy rid was null
    assert wf.checksum == "793f7dadfa14ecc227d353e400cebc449bd2ec94"


def test_legacy_rid_with_value_maps_to_workflow_rid():
    """A non-null legacy ``rid`` is carried over to ``workflow_rid``."""
    data = {**_LEGACY_WORKFLOW, "rid": "1-ABCD"}
    wf = Workflow.model_validate(data)
    assert wf.workflow_rid == "1-ABCD"


def test_new_shape_workflow_rid_still_parses():
    """The current (post-rename) shape with ``workflow_rid`` is unaffected."""
    data = {
        "name": "API Workflow",
        "workflow_type": "Feature Notebook Workflow",
        "workflow_rid": "2-WXYZ",
        "checksum": "abc",
    }
    wf = Workflow.model_validate(data)
    assert wf.workflow_rid == "2-WXYZ"


def test_workflow_rid_takes_precedence_when_both_present():
    """If both keys appear, the canonical ``workflow_rid`` wins and no error is raised."""
    data = {
        "name": "API Workflow",
        "workflow_type": "Feature Notebook Workflow",
        "workflow_rid": "2-WXYZ",  # valid RID form
        "rid": "1-ABCD",  # legacy key, should be ignored in favor of workflow_rid
    }
    wf = Workflow.model_validate(data)
    assert wf.workflow_rid == "2-WXYZ"


def test_validator_does_not_mutate_caller_input():
    """The before-validator copies its input; the caller's dict is left intact.

    Guards the property whose earlier absence caused an order-dependent test
    failure: the validator must not pop ``rid`` out of the passed dict.
    """
    original = dict(_LEGACY_WORKFLOW)  # has a "rid" key
    Workflow.model_validate(original)
    assert "rid" in original, "validator must not mutate the caller's dict"
    assert original["rid"] is None


def test_genuinely_unknown_field_still_forbidden():
    """The #226 guard is preserved: an unknown field (not the legacy alias) still raises."""
    data = {
        "name": "API Workflow",
        "workflow_type": "Feature Notebook Workflow",
        "bogus_field": "should not be allowed",
    }
    with pytest.raises(ValidationError):
        Workflow.model_validate(data)
