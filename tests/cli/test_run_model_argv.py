"""Tests for ``deriva-ml-run`` argument handling and Hydra-flag passthrough.

The model runner uses ``parse_known_args`` (no greedy ``nargs="*"`` positional,
which would steal an interleaved flag's value) and treats the entire ordered
remainder as overrides + Hydra-native flags, forwarding it verbatim into the
``sys.argv`` that ``hydra.main`` re-parses. So every Hydra-doc flag (``--cfg``,
``--info``, ``--resolve``, ``--package``, ...) works without per-flag wiring.
The hydra-zen config-group menu moved from ``--info`` to ``--list-configs`` so
the ``--info`` name is free to flow to Hydra.

These tests exercise the pure ``build_hydra_argv`` helper (no catalog, no
execution) plus the argparse surface, so they are fast unit tests.
"""

from __future__ import annotations

from deriva_ml.run_model import DerivaMLRunCLI, build_hydra_argv

# =============================================================================
# build_hydra_argv: forwarding overrides + Hydra-native flags + deriva injection
# =============================================================================


class TestBuildHydraArgv:
    """The pure argv builder handed to ``hydra.main`` via ``sys.argv``."""

    PROG = "deriva-ml-run"

    def _argv(self, **kwargs) -> list[str]:
        params = dict(
            prog=self.PROG,
            forwarded=[],
            use_multirun=False,
            host=None,
            catalog=None,
        )
        params.update(kwargs)
        return build_hydra_argv(**params)

    def test_prog_is_argv0(self):
        assert self._argv()[0] == self.PROG

    def test_cfg_job_forwarded(self):
        """--cfg job survives into the Hydra argv, value adjacent (case 1)."""
        argv = self._argv(forwarded=["--cfg", "job"])
        assert "--cfg" in argv
        assert argv[argv.index("--cfg") + 1] == "job"

    def test_cfg_modes_forwarded(self):
        """--cfg hydra / --cfg all forwarded verbatim (case 2)."""
        for mode in ("hydra", "all"):
            argv = self._argv(forwarded=["--cfg", mode])
            assert "--cfg" in argv and mode in argv

    def test_info_modes_forwarded(self):
        """--info config/defaults/searchpath forwarded to Hydra (case 3)."""
        for mode in ("config", "defaults", "searchpath"):
            argv = self._argv(forwarded=["--info", mode])
            assert "--info" in argv and mode in argv

    def test_bare_info_forwarded(self):
        """bare --info forwarded; Hydra defaults it to 'all' (case 4)."""
        assert "--info" in self._argv(forwarded=["--info"])

    def test_resolve_and_package_forwarded(self):
        """--resolve and --package model_config forwarded (case 5)."""
        argv = self._argv(forwarded=["--resolve", "--package", "model_config"])
        assert {"--resolve", "--package", "model_config"} <= set(argv)

    def test_override_present(self):
        argv = self._argv(forwarded=["model_config=cifar10_quick"])
        assert "model_config=cifar10_quick" in argv

    def test_override_and_cfg_coexist(self):
        """+experiment=X AND --cfg job both reach Hydra, both intact (case 8)."""
        argv = self._argv(forwarded=["+experiment=cifar10_quick", "--cfg", "job"])
        assert "+experiment=cifar10_quick" in argv
        assert argv[argv.index("--cfg") + 1] == "job"

    def test_multirun_flag_inserted(self):
        """use_multirun inserts --multirun (case 9)."""
        argv = self._argv(forwarded=["+experiment=a,b"], use_multirun=True)
        assert "--multirun" in argv

    def test_multirun_not_double_inserted(self):
        """If the user already passed --multirun, don't add a second (case 10)."""
        argv = self._argv(forwarded=["model=a,b", "--multirun"], use_multirun=True)
        assert argv.count("--multirun") == 1

    def test_multirun_short_flag_not_doubled(self):
        argv = self._argv(forwarded=["model=a,b", "-m"], use_multirun=True)
        assert "--multirun" not in argv  # user's -m already signals it

    def test_catalog_injected(self):
        """--catalog 45 → deriva_ml.catalog_id=45 (case 13)."""
        assert "deriva_ml.catalog_id=45" in self._argv(catalog="45")

    def test_host_injected(self):
        """--host h → deriva_ml.hostname=h (case 13)."""
        assert "deriva_ml.hostname=example.org" in self._argv(host="example.org")

    def test_everything_coexists(self):
        """Overrides + forwarded flags + injection all survive (case 15)."""
        argv = self._argv(
            forwarded=["model_config=cifar10_quick", "+experiment=x", "--cfg", "job", "--resolve"],
            catalog="45",
        )
        for token in (
            "model_config=cifar10_quick",
            "+experiment=x",
            "--cfg",
            "job",
            "--resolve",
            "deriva_ml.catalog_id=45",
        ):
            assert token in argv


# =============================================================================
# argparse surface: --list-configs is ours; Hydra flags fall to the remainder
# intact (no greedy positional stealing flag values)
# =============================================================================


class TestArgparseSurface:
    def _parse(self, argv: list[str]):
        return DerivaMLRunCLI().parser.parse_known_args(argv)

    def test_list_configs_defined_and_consumed(self):
        """--list-configs is the deriva-ml menu flag, consumed not forwarded (case 6)."""
        ns, unknown = self._parse(["--list-configs"])
        assert ns.list_configs is True
        assert unknown == []

    def test_cfg_and_value_stay_together_in_remainder(self):
        """The fix: --cfg job is NOT split (no greedy positional steals 'job')."""
        ns, unknown = self._parse(["--cfg", "job"])
        assert unknown == ["--cfg", "job"]

    def test_info_and_value_stay_together(self):
        ns, unknown = self._parse(["--info", "config"])
        assert unknown == ["--info", "config"]

    def test_flag_before_override_keeps_value(self):
        """--cfg job model=x: 'job' must stay with --cfg, not become an override."""
        ns, unknown = self._parse(["--cfg", "job", "model_config=quick"])
        assert unknown == ["--cfg", "job", "model_config=quick"]

    def test_deriva_flags_consumed_not_forwarded(self):
        ns, unknown = self._parse(["--catalog", "45", "--multirun", "--allow-dirty"])
        assert ns.catalog == "45"
        assert ns.multirun is True
        assert ns.allow_dirty is True
        for tok in ("--catalog", "--multirun", "--allow-dirty"):
            assert tok not in unknown

    def test_order_preserved_in_remainder(self):
        ns, unknown = self._parse(["+experiment=x", "--resolve", "model=a", "--cfg", "job"])
        assert unknown == ["+experiment=x", "--resolve", "model=a", "--cfg", "job"]
