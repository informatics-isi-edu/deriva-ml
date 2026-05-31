"""Tests for the shared config-group menu renderer used by both runners."""

from __future__ import annotations

from hydra_zen import builds, store

from deriva_ml.cli.show_info import render_config_groups


def _register_sample_configs():
    """Register a couple of grouped configs into the hydra-zen store."""
    store(builds(dict, a=1), group="model_config", name="sample_model")
    store(builds(dict, b=2), group="datasets", name="sample_dataset")


class TestRenderConfigGroups:
    def test_lists_registered_groups_and_options(self):
        _register_sample_configs()
        text = render_config_groups(include_multirun=False)
        assert "Available Hydra Configuration Groups:" in text
        assert "model_config:" in text
        assert "  - sample_model" in text
        assert "datasets:" in text
        assert "  - sample_dataset" in text

    def test_points_at_cfg_job_for_resolved_config(self):
        """The footer must steer users to --cfg job for the resolved config
        (the operation --info/--list-configs does NOT perform)."""
        text = render_config_groups(include_multirun=False)
        assert "--cfg job" in text

    def test_no_multirun_section_when_excluded(self):
        text = render_config_groups(include_multirun=False)
        assert "multirun:" not in text

    def test_multirun_section_optional_include(self):
        """include_multirun=True is accepted and renders without error even
        when no multirun configs are registered (section simply omitted)."""
        text = render_config_groups(include_multirun=True)
        assert "Available Hydra Configuration Groups:" in text

    def test_never_raises_returns_string(self):
        assert isinstance(render_config_groups(include_multirun=True), str)
        assert isinstance(render_config_groups(include_multirun=False), str)
