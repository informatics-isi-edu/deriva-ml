"""Tests for ``deriva-ml-run-notebook`` argument handling and inspection flags.

Unlike the model runner, the notebook runner never hands argv to Hydra: it
serializes overrides into ``DERIVA_ML_HYDRA_OVERRIDES`` and the papermill kernel
calls ``hydra_zen.launch(...)``. So Hydra's ``--cfg`` / ``--info`` cannot flow
through as argv. Instead the runner implements them by *resolving the config in
the runner process* (``render_notebook_config``) and rendering it the way Hydra
would, **without executing the notebook**.

The hydra-zen config-group menu moved from ``--info`` to ``--list-configs`` so
the ``--info`` name is free to carry Hydra's own info vocabulary. Papermill keeps
its orthogonal flags (``--parameter/-p``, ``--kernel/-k``, ``--file/-f``,
``--inspect``, ``--log-output``) and the ``notebook_file`` positional.

These are fast unit tests: the argparse surface is exercised directly, and the
compose+render and papermill execution paths are patched so no real notebook
runs and no catalog is touched.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest
from hydra_zen import builds, store

from deriva_ml.execution.base_config import BaseConfig, _notebook_configs
from deriva_ml.run_notebook import DerivaMLRunNotebookCLI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class _CLISampleConfig(BaseConfig):
    """Self-contained config used by the inspection tests."""

    threshold: float = 0.5


def _register_sample() -> None:
    """Register a self-contained notebook config (no group defaults).

    Uses ``hydra_defaults=["_self_"]`` so composition is catalog-free.
    Idempotent: hydra-zen's default ``store`` rejects duplicate names, so we
    register only once per process.
    """
    if "cli_sample_nb" in _notebook_configs:
        return
    config_builds = builds(
        _CLISampleConfig,
        populate_full_signature=True,
        hydra_defaults=["_self_"],
        threshold=0.25,
    )
    store(config_builds, name="cli_sample_nb")
    _notebook_configs["cli_sample_nb"] = (config_builds, "cli_sample_nb")
    store.add_to_hydra_store(overwrite_ok=True)


def _make_cli() -> DerivaMLRunNotebookCLI:
    """Build the CLI with the nbstripout check stubbed out (no git needed)."""
    with mock.patch(
        "deriva_ml.run_notebook.Workflow._check_nbstrip_status",
        return_value=None,
    ):
        return DerivaMLRunNotebookCLI(description="test", epilog="")


def _run_with_argv(argv: list[str], *, configs_loaded: bool = True):
    """Run ``cli.main()`` with a patched ``sys.argv`` and stubbed execution.

    Patches ``run_notebook`` (the heavy papermill+catalog path) so we can assert
    whether the runner executed the notebook or short-circuited on an inspection
    flag. Also patches ``_load_project_configs`` (which imports the real
    ``src/configs`` package off disk): the tests pre-register their config in
    the hydra-zen store directly, so this boundary is stubbed to report success
    without needing a project tree on disk.

    Args:
        argv: CLI arguments after the program name.
        configs_loaded: Return value for the patched ``_load_project_configs``.

    Returns:
        Tuple of (cli, run_notebook_mock, exit_code-or-None).
    """
    cli = _make_cli()
    original = sys.argv
    sys.argv = ["deriva-ml-run-notebook", *argv]
    try:
        with (
            mock.patch.object(cli, "run_notebook") as run_mock,
            mock.patch.object(
                DerivaMLRunNotebookCLI,
                "_load_project_configs",
                staticmethod(lambda _notebook_file: configs_loaded),
            ),
        ):
            try:
                result = cli.main()
            except SystemExit as exc:  # argparse choices rejection
                return cli, run_mock, exc.code
            return cli, run_mock, result
    finally:
        sys.argv = original


# ---------------------------------------------------------------------------
# argparse surface
# ---------------------------------------------------------------------------
class TestArgparseSurface:
    """The notebook runner keeps its papermill flags and adds inspection ones."""

    def _parse(self, argv: list[str]):
        return _make_cli().parser.parse_args(argv)

    def test_list_configs_is_store_true(self):
        ns = self._parse(["nb.ipynb", "--list-configs"])
        assert ns.list_configs is True

    def test_info_bare_defaults_to_all(self):
        ns = self._parse(["nb.ipynb", "--info"])
        assert ns.info == "all"

    def test_info_accepts_hydra_modes(self):
        for mode in ("all", "config", "defaults", "defaults-tree", "plugins", "searchpath"):
            ns = self._parse(["nb.ipynb", "--info", mode])
            assert ns.info == mode

    def test_cfg_bare_defaults_to_job(self):
        ns = self._parse(["nb.ipynb", "--cfg"])
        assert ns.cfg == "job"

    def test_cfg_accepts_modes(self):
        for mode in ("job", "hydra", "all"):
            ns = self._parse(["nb.ipynb", "--cfg", mode])
            assert ns.cfg == mode

    def test_info_bad_mode_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["nb.ipynb", "--info", "bogus"])

    def test_cfg_bad_mode_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["nb.ipynb", "--cfg", "bogus"])

    def test_parameter_long_form(self):
        ns = self._parse(["nb.ipynb", "--parameter", "foo", "1"])
        assert ["foo", "1"] in ns.parameter

    def test_parameter_short_form_is_papermill(self):
        """-p belongs to papermill on this runner (NOT Hydra's --package)."""
        ns = self._parse(["nb.ipynb", "-p", "foo", "1"])
        assert ["foo", "1"] in ns.parameter

    def test_kernel_file_inspect_logoutput_and_positional(self):
        ns = self._parse(
            [
                "nb.ipynb",
                "--kernel",
                "myenv",
                "--file",
                "params.json",
                "--inspect",
                "--log-output",
            ]
        )
        assert ns.notebook_file == Path("nb.ipynb")
        assert ns.kernel == "myenv"
        assert ns.file == Path("params.json")
        assert ns.inspect is True
        assert ns.log_output is True


# ---------------------------------------------------------------------------
# main() behavior: inspection flags short-circuit execution
# ---------------------------------------------------------------------------
class TestInspectionShortCircuits:
    """--list-configs / --info / --cfg never execute the notebook."""

    def test_list_configs_prints_menu_no_execution(self, capsys):
        _register_sample()
        cli, run_mock, _ = _run_with_argv(["nb.ipynb", "--list-configs"])
        out = capsys.readouterr().out
        assert "Available Hydra Configuration Groups:" in out
        run_mock.assert_not_called()

    def test_info_config_renders_without_execution(self, capsys):
        _register_sample()
        with mock.patch(
            "deriva_ml.run_notebook.render_notebook_config",
            return_value="RENDERED-CONFIG-INFO",
        ) as render_mock:
            cli, run_mock, _ = _run_with_argv(["cli_sample_nb.ipynb", "--info", "config"])
        out = capsys.readouterr().out
        assert "RENDERED-CONFIG-INFO" in out
        run_mock.assert_not_called()
        # info_mode plumbed through; config name derived from notebook stem.
        _, kwargs = render_mock.call_args
        assert kwargs["info_mode"] == "config"

    def test_cfg_job_renders_without_execution(self, capsys):
        _register_sample()
        with mock.patch(
            "deriva_ml.run_notebook.render_notebook_config",
            return_value="RENDERED-CFG-JOB",
        ) as render_mock:
            cli, run_mock, _ = _run_with_argv(["cli_sample_nb.ipynb", "--cfg", "job"])
        out = capsys.readouterr().out
        assert "RENDERED-CFG-JOB" in out
        run_mock.assert_not_called()
        _, kwargs = render_mock.call_args
        assert kwargs["cfg_mode"] == "job"

    def test_cfg_passes_overrides_through(self):
        """The positional hydra_overrides reach render_notebook_config."""
        _register_sample()
        with mock.patch(
            "deriva_ml.run_notebook.render_notebook_config",
            return_value="x",
        ) as render_mock:
            _run_with_argv(["cli_sample_nb.ipynb", "threshold=0.9", "--cfg", "job"])
        _, kwargs = render_mock.call_args
        assert kwargs["overrides"] == ["threshold=0.9"]

    def test_render_path_needs_no_catalog(self, capsys):
        """End-to-end render through the real entry point — no network.

        Uses the real ``render_notebook_config`` (not patched) against a
        self-contained registered config, proving the compose-only path works
        with no live catalog.
        """
        _register_sample()
        cli, run_mock, _ = _run_with_argv(["cli_sample_nb.ipynb", "--cfg", "job"])
        out = capsys.readouterr().out
        assert "threshold: 0.25" in out
        run_mock.assert_not_called()


# ---------------------------------------------------------------------------
# guard: bare positional overrides still raise the friendly error
# ---------------------------------------------------------------------------
class TestOverrideGuard:
    def test_bare_positional_override_rejected(self, capsys):
        _register_sample()
        cli, run_mock, code = _run_with_argv(["nb.ipynb", "roc_analysis"])
        err = capsys.readouterr().err
        assert "looks like a positional argument" in err
        assert code == 1
        run_mock.assert_not_called()
