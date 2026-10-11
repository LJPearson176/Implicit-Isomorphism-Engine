from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    config_version: str
    data_snapshot_sha256: str
    strategy: str
    solver: str
    solver_status: str
    software_versions: Dict[str, str]
    validation_checks: Dict[str, Any]
    metrics: Dict[str, float]
    passed: bool
    known_limitations: list[str]
    generated_at_utc: str


def build_manifest(
    *,
    experiment_id: str,
    config_version: str,
    data_snapshot_sha256: str,
    strategy: str,
    solver: str,
    solver_status: str,
    software_versions: Dict[str, str],
    validation_checks: Dict[str, Any],
    metrics: Dict[str, float],
    passed: bool,
    known_limitations: list[str],
) -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id=experiment_id,
        config_version=config_version,
        data_snapshot_sha256=data_snapshot_sha256,
        strategy=strategy,
        solver=solver,
        solver_status=solver_status,
        software_versions=software_versions,
        validation_checks=validation_checks,
        metrics=metrics,
        passed=passed,
        known_limitations=known_limitations,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def write_manifest(manifest: ExperimentManifest, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
