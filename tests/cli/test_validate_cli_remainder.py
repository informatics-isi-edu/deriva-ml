"""Tests for ``validate_cli_remainder`` — the flag-value-aware bare-positional guard.

The deriva-ml runners forward their full ``parse_known_args`` remainder
(overrides + Hydra-native flags) to Hydra. ``validate_cli_remainder`` preserves
the friendly "you typed a bare positional" error without false-positiving on
Hydra flags or their values. The Hydra flag→arity map is introspected from
Hydra itself, so the guard auto-tracks Hydra's flag set.
"""

from __future__ import annotations

import pytest

from deriva_ml.cli.hydra_overrides import _hydra_flag_arities, validate_cli_remainder


class TestHydraFlagArities:
    def test_known_value_flags(self):
        arities = _hydra_flag_arities()
        assert arities.get("--cfg") == 1
        assert arities.get("--package") == 1
        assert arities.get("--config-name") == 1

    def test_known_boolean_flags(self):
        arities = _hydra_flag_arities()
        assert arities.get("--resolve") == 0
        assert arities.get("--multirun") == 0

    def test_short_forms_present(self):
        arities = _hydra_flag_arities()
        # -c is Hydra's --cfg short form
        assert "-c" in arities


class TestValidateCliRemainderAccepts:
    """Legitimate remainders must NOT raise."""

    @pytest.mark.parametrize(
        "remainder",
        [
            [],
            ["model_config=cifar10_quick"],
            ["+experiment=quick"],
            ["++force=add"],
            ["~deriva_ml.catalog_id"],
            ["model_config.learning_rate=0.001"],
            ["--cfg", "job"],
            ["--cfg", "all"],
            ["--info"],  # bare, Hydra defaults to all
            ["--info", "config"],
            ["--resolve"],
            ["--package", "model_config"],
            ["+experiment=quick", "--cfg", "job"],
            ["--cfg", "job", "model_config=quick"],  # flag before override
            ["--resolve", "--package", "model_config", "model=a"],
            ["model=a,b", "--multirun"],
            ["-c", "job"],  # short form
        ],
    )
    def test_accepts(self, remainder):
        # Should not raise.
        validate_cli_remainder(remainder, cli_name="deriva-ml-run")


class TestValidateCliRemainderRejects:
    """Genuine bare positionals must raise the friendly error."""

    def test_bare_positional_alone(self):
        with pytest.raises(ValueError, match="looks like a positional argument"):
            validate_cli_remainder(["roc_analysis"], cli_name="deriva-ml-run")

    def test_bare_positional_after_override(self):
        with pytest.raises(ValueError, match="roc_analysis"):
            validate_cli_remainder(["model_config=quick", "roc_analysis"], cli_name="deriva-ml-run")

    def test_bare_positional_after_flag_and_its_value(self):
        """--cfg job consumes 'job'; a trailing bare positional is still caught."""
        with pytest.raises(ValueError, match="roc_analysis"):
            validate_cli_remainder(["--cfg", "job", "roc_analysis"], cli_name="deriva-ml-run")

    def test_error_names_cli_and_suggests_list_configs(self):
        with pytest.raises(ValueError, match="--list-configs"):
            validate_cli_remainder(["roc_analysis"], cli_name="deriva-ml-run")

    def test_bare_positional_between_flags(self):
        """A bare positional that is NOT a recognized flag's value is caught,
        even sitting next to a zero-arity flag."""
        with pytest.raises(ValueError, match="oops"):
            validate_cli_remainder(["--resolve", "oops"], cli_name="deriva-ml-run")
