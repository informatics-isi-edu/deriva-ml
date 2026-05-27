"""Tests for Hydra override validation on the deriva-ml CLIs.

The CLIs ``deriva-ml-run`` and ``deriva-ml-run-notebook`` accept Hydra
overrides as trailing positional arguments. A bare positional token
(no ``=``) used to slip past ``argparse`` and only fail deep inside
Hydra's ANTLR parser with ``missing EQUAL at '<EOF>'``. These tests
lock in the friendlier upfront error message.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from deriva_ml.cli.hydra_overrides import (
    _looks_like_override,
    validate_hydra_overrides,
)


class TestLooksLikeOverride:
    """Unit tests for the override-shape predicate."""

    @pytest.mark.parametrize(
        "token",
        [
            "key=value",
            "+experiment=cifar10_quick",
            "++force.add=1",
            "assets=roc_quick_probabilities",
            "~deriva_ml.catalog_id",
            "~deriva_ml.catalog_id=42",
            "deriva_ml.hostname=host.example.org",
            "key@package=value",
        ],
    )
    def test_accepts_valid_hydra_forms(self, token: str) -> None:
        """Every documented Hydra override shape passes the check."""
        assert _looks_like_override(token) is True

    @pytest.mark.parametrize(
        "token",
        [
            "roc_analysis",
            "cifar10_quick",
            "my_experiment",
            "",
        ],
    )
    def test_rejects_bare_positional(self, token: str) -> None:
        """Bare positional tokens (no '=' and no '~') are rejected."""
        assert _looks_like_override(token) is False


class TestValidateHydraOverrides:
    """Unit tests for the public validator."""

    def test_accepts_empty_list(self) -> None:
        """An empty override list is valid (no overrides supplied)."""
        validate_hydra_overrides([])

    def test_accepts_all_valid_overrides(self) -> None:
        """A mix of all valid override forms passes."""
        validate_hydra_overrides(
            [
                "assets=roc_quick_probabilities",
                "+experiment=cifar10_quick",
                "~deriva_ml.catalog_id",
                "++force.flag=true",
            ]
        )

    def test_rejects_bare_positional_with_diagnostic(self) -> None:
        """A bare positional triggers ValueError naming the bad token."""
        with pytest.raises(ValueError) as excinfo:
            validate_hydra_overrides(["roc_analysis"])

        msg = str(excinfo.value)
        # Names the offending token
        assert "'roc_analysis'" in msg
        # Mentions the expected form
        assert "key=value" in msg
        # Offers the canonical suggestions
        assert "assets=roc_analysis" in msg
        assert "+experiment=roc_analysis" in msg

    def test_rejects_first_bad_token_in_mixed_list(self) -> None:
        """When valid and invalid tokens are mixed, the invalid one is reported."""
        with pytest.raises(ValueError, match=r"'cifar10_quick'"):
            validate_hydra_overrides(
                [
                    "assets=roc_quick_probabilities",
                    "cifar10_quick",  # bare positional
                    "+experiment=other",
                ]
            )

    def test_cli_name_appears_in_message(self) -> None:
        """The cli_name kwarg is reflected in the suggestion text."""
        with pytest.raises(ValueError) as excinfo:
            validate_hydra_overrides(["foo"], cli_name="deriva-ml-run")
        assert "deriva-ml-run" in str(excinfo.value)
        # Suggestions should also use the custom name
        assert "deriva-ml-run ... assets=foo" in str(excinfo.value)


class TestEndToEndCLIErrorMessage:
    """End-to-end check that ``deriva-ml-run`` surfaces the friendly error."""

    def test_run_model_cli_rejects_bare_positional(self, tmp_path) -> None:
        """Running ``deriva-ml-run roc_analysis`` exits non-zero with the helper text.

        We invoke ``deriva-ml-run`` as a subprocess against a minimal
        configs directory so we don't depend on any specific project
        layout. The validator runs BEFORE config loading, so even an
        empty configs dir is fine -- but we still need ``--config-dir``
        to point somewhere that exists.
        """
        # Minimal valid config dir for the CLI's early existence check.
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        (configs_dir / "__init__.py").write_text("")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "deriva_ml.run_model",
                "--config-dir",
                str(configs_dir),
                "roc_analysis",  # the offending bare positional
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "'roc_analysis'" in combined
        assert "key=value" in combined
        # Ensure we did NOT leak the cryptic ANTLR error
        assert "missing EQUAL" not in combined
