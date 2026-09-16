from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_experiment(
    output_dir: str | Path,
    experiment_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a self-contained, JSON-serializable experiment record."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": experiment_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": metrics,
        "metadata": metadata or {},
    }
    path = output / f"{experiment_name}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def append_results_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Append structured experiment rows for cross-run comparison."""
    import pandas as pd

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(target, mode="a", header=not target.exists(), index=False)
