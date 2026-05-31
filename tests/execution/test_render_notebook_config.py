"""Tests for ``render_notebook_config`` — compose-only config rendering.

``deriva-ml-run-notebook --cfg job`` / ``--info config`` resolve a notebook's
hydra-zen config *in the runner process* and render it the way Hydra's own
``--cfg`` / ``--info`` would, **without** executing the notebook and **without**
touching a live catalog (pure config composition). This module exercises the
``render_notebook_config`` entry point that performs that compose-and-render.

The configs registered here are fully self-contained (``hydra_defaults`` is just
``["_self_"]``) so composition needs no group configs and no network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest
from hydra_zen import builds, store

from deriva_ml.execution.base_config import (
    BaseConfig,
    _notebook_configs,
    render_notebook_config,
)


@dataclass
class _RenderSampleConfig(BaseConfig):
    """Self-contained notebook config used only by these tests."""

    threshold: float = 0.5


def _register() -> None:
    """Register a self-contained notebook config (no group defaults).

    Uses ``hydra_defaults=["_self_"]`` so composition needs no group configs
    (``deriva_ml``/``datasets``/``assets``) and stays catalog-free. Idempotent:
    hydra-zen's default ``store`` rejects duplicate names, so we register only
    once per process.
    """
    if "render_sample_nb" in _notebook_configs:
        return
    config_builds = builds(
        _RenderSampleConfig,
        populate_full_signature=True,
        hydra_defaults=["_self_"],
        threshold=0.25,
    )
    store(config_builds, name="render_sample_nb")
    _notebook_configs["render_sample_nb"] = (config_builds, "render_sample_nb")
    store.add_to_hydra_store(overwrite_ok=True)


class TestRenderNotebookConfig:
    """``render_notebook_config`` composes + renders without instantiating."""

    def test_cfg_job_renders_yaml_with_field(self):
        """--cfg job renders the composed job config as YAML text."""
        _register()
        text = render_notebook_config("render_sample_nb", overrides=[], cfg_mode="job")
        assert "threshold: 0.25" in text
        assert "_target_" in text

    def test_cfg_job_applies_overrides(self):
        """User overrides are reflected in the rendered config."""
        _register()
        text = render_notebook_config("render_sample_nb", overrides=["threshold=0.9"], cfg_mode="job")
        assert "threshold: 0.9" in text

    def test_info_config_renders_composed_config(self):
        """--info config renders Hydra's composed-config section."""
        _register()
        text = render_notebook_config("render_sample_nb", overrides=[], info_mode="config")
        assert "threshold: 0.25" in text

    def test_info_searchpath_renders_searchpath(self):
        """--info searchpath renders Hydra's search-path section."""
        _register()
        text = render_notebook_config("render_sample_nb", overrides=[], info_mode="searchpath")
        assert "Config search path" in text

    def test_cfg_hydra_renders_hydra_block(self):
        """--cfg hydra renders the hydra: block (not the job config)."""
        _register()
        text = render_notebook_config("render_sample_nb", overrides=[], cfg_mode="hydra")
        assert "hydra:" in text

    def test_requires_a_mode(self):
        """Calling with neither mode is a programming error."""
        _register()
        with pytest.raises(ValueError):
            render_notebook_config("render_sample_nb", overrides=[])

    def test_does_not_touch_catalog(self):
        """Rendering is pure composition — no DerivaML / network access.

        The config carries ``deriva_ml: null`` here; instantiating it would not
        connect, but to be certain we never instantiate, we assert the rendered
        text is the raw config (contains ``_target_``) rather than an
        instantiated object's repr.
        """
        _register()
        text = render_notebook_config("render_sample_nb", overrides=[], cfg_mode="job")
        assert "_target_:" in text

    def test_info_robust_to_leaked_logging_disable(self):
        """--info captures Hydra's logger output even under a stray disable.

        Hydra's ``show_info`` emits via the ``hydra._internal.hydra`` logger at
        DEBUG. A leaked global ``logging.disable`` (as can happen when running
        in a large shared test process) would mute it under a naive stdout
        redirect. The renderer must lift the disable for the duration and
        restore it afterward. Regression for the full-suite-only failure.
        """
        _register()
        prev_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            text = render_notebook_config("render_sample_nb", overrides=[], info_mode="config")
            assert "threshold: 0.25" in text
            search = render_notebook_config("render_sample_nb", overrides=[], info_mode="searchpath")
            assert "Config search path" in search
            # The renderer must restore the ambient disable threshold it found.
            assert logging.root.manager.disable == logging.CRITICAL
        finally:
            logging.disable(prev_disable)
