from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{x:.4f}"


def build_markdown_report(
    output_path: str | Path,
    title: str,
    universe: list[str],
    model_metrics: dict[str, Any],
    strategy_metrics: dict[str, Any],
    calibration_metrics: dict[str, Any],
    artifacts: dict[str, str],
    notes: list[str],
) -> Path:
    """Write a publication-style research report from precomputed diagnostics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Research question",
        "Can the evaluated signal improve out-of-sample risk-adjusted performance after the stated trading frictions?",
        "",
        "## Experimental setup",
        f"- Universe: `{', '.join(universe)}`",
        "- Evaluation: chronological, out-of-sample",
        "- Costs: included in reported strategy returns",
        "",
        "## Model diagnostics",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in model_metrics.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines += ["", "## Strategy diagnostics", "| Metric | Value |", "|---|---:|"]
    for key, value in strategy_metrics.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines += ["", "## Probability calibration", "| Metric | Value |", "|---|---:|"]
    for key, value in calibration_metrics.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines += ["", "## Evidence artifacts"]
    for label, artifact in artifacts.items():
        lines.append(f"- [{label}]({artifact})")
    lines += ["", "## Interpretation notes"]
    lines.extend(f"- {note}" for note in notes)
    lines += [
        "",
        "## Limitations",
        "This report is a historical research artifact. It does not establish future profitability or live-trading suitability. Model, data, execution, and regime risks remain.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
