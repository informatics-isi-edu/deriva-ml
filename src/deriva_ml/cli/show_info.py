"""Shared rendering of the hydra-zen config-group menu for the deriva-ml CLIs.

``deriva-ml-run --list-configs`` and ``deriva-ml-run-notebook --list-configs``
both print the registered hydra-zen config groups and their selectable options
(the menu of ``group=value`` choices). This is a deriva-ml convenience that
Hydra has no native equivalent for — Hydra's own ``--info`` modes report the
*composed/resolved* config and search paths, not the registry of selectable
options.

This module is the single source of that rendering so the two runners stay in
lockstep (a previous revision kept two drifting copies). To inspect the
*resolved* config a run would use, callers should reach for Hydra's ``--cfg
job`` instead; this listing is strictly "what can I pass?".
"""

from __future__ import annotations


def render_config_groups(*, include_multirun: bool) -> str:
    """Render the registered hydra-zen config groups + options as text.

    Walks the hydra-zen store's registration queue and groups every registered
    config by its group (top-level configs are bucketed under
    ``"Top-level configs"``). Optionally appends a ``multirun:`` section listing
    named multirun configs with a one-line description.

    Args:
        include_multirun: When True, append the registered named multirun
            configs (the model runner uses this; the notebook runner, which has
            no multirun surface, passes False).

    Returns:
        A printable multi-line string. On any introspection error the returned
        string carries a short diagnostic line rather than raising — listing the
        menu must never crash the CLI.

    Example:
        >>> text = render_config_groups(include_multirun=True)  # doctest: +SKIP
        >>> print(text)  # doctest: +SKIP
        Available Hydra Configuration Groups:
        ...
    """
    from hydra_zen import store

    lines: list[str] = ["Available Hydra Configuration Groups:", "=" * 50]

    try:
        groups: dict[str, list[str]] = {}
        for group, name in store._queue:
            bucket = group if group else "__root__"
            groups.setdefault(bucket, [])
            if name not in groups[bucket]:
                groups[bucket].append(name)

        for group in sorted(groups.keys()):
            lines.append("")
            lines.append("Top-level configs:" if group == "__root__" else f"{group}:")
            for name in sorted(groups[group]):
                lines.append(f"  - {name}")

        if include_multirun:
            from deriva_ml.execution import get_all_multirun_configs

            multirun_configs = get_all_multirun_configs()
            if multirun_configs:
                lines.append("")
                lines.append("multirun:")
                for name in sorted(multirun_configs.keys()):
                    spec = multirun_configs[name]
                    if spec.description:
                        summary = spec.description.strip().split("\n")[0].lstrip("#").strip()
                        if len(summary) > 50:
                            summary = summary[:47] + "..."
                    else:
                        summary = ", ".join(spec.overrides[:2])
                    lines.append(f"  - {name}: {summary}")

        lines.append("")
        lines.append("=" * 50)
        lines.append("Pass a choice as a Hydra override, e.g.:")
        lines.append("  deriva-ml-run model_config=cifar10_quick")
        lines.append("  deriva-ml-run +experiment=cifar10_quick")
        lines.append("To inspect the RESOLVED config a run would use (Hydra):")
        lines.append("  deriva-ml-run +experiment=cifar10_quick --cfg job")
    except Exception as exc:  # pragma: no cover - defensive; listing must not crash
        lines.append(f"Error inspecting hydra-zen store: {exc}")

    return "\n".join(lines)
