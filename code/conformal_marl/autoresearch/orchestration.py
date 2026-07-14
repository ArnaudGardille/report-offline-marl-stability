"""Generate and validate AI Scientist's constrained MARL search configuration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


FROZEN_PATHS = (
    "code/conformal_marl/autoresearch/api.py",
    "code/conformal_marl/autoresearch/evaluate.py",
    "code/conformal_marl/autoresearch/frozen",
    "code/conformal_marl/envs",
    "code/conformal_marl/iql.py",
    "code/conformal_marl/replay.py",
    "code/conformal_marl/samplers.py",
    "code/tests/test_autoresearch_operator.py",
    "code/tests/test_autoresearch_learning.py",
    "code/pyproject.toml",
    "code/uv.lock",
)


def _git_ref_exists(repository: Path, ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def build_search_config(
    *,
    repository: Path,
    orchestrator: Path,
    base_ref: str,
    output_dir: Path,
    task_context_file: Path,
    max_nodes: int = 12,
    branching_factor: int = 3,
    operator_trials: int = 128,
    learning_seeds: int = 8,
) -> dict:
    """Build a portable constrained-search config after strict local preflight."""

    repository = repository.resolve()
    orchestrator = orchestrator.resolve()
    task_context_file = task_context_file.resolve()
    candidate = repository / "code/conformal_marl/autoresearch/candidate.py"
    launcher = orchestrator / "launch_constrained_bfts.py"

    if not (repository / ".git").exists() and not _git_ref_exists(repository, "HEAD"):
        raise ValueError(f"not a Git repository: {repository}")
    if not _git_ref_exists(repository, base_ref):
        raise ValueError(f"Git ref {base_ref!r} does not exist in {repository}")
    if not candidate.is_file():
        raise ValueError(f"candidate contract is missing: {candidate}")
    if not launcher.is_file():
        raise ValueError(f"constrained AI Scientist launcher is missing: {launcher}")
    if not task_context_file.is_file():
        raise ValueError(f"task context is missing: {task_context_file}")
    if max_nodes < 2 or branching_factor < 1 or operator_trials < 2 or learning_seeds < 2:
        raise ValueError("invalid search or evaluation budget")

    missing_frozen = [path for path in FROZEN_PATHS if not (repository / path).exists()]
    if missing_frozen:
        raise ValueError(f"frozen benchmark paths are missing: {missing_frozen}")

    return {
        "repository": str(repository),
        "base_ref": base_ref,
        "candidate_path": "code/conformal_marl/autoresearch/candidate.py",
        "working_subdir": "code",
        "frozen_paths": list(FROZEN_PATHS),
        "stages": [
            {
                "name": "operator",
                "command": [
                    "uv", "run", "python", "-m",
                    "conformal_marl.autoresearch.evaluate",
                    "--tier", "operator", "--trials", str(operator_trials),
                    "--output", "{output}",
                ],
                "minimum_score": -0.25,
                "required_status": "ok",
            },
            {
                "name": "learning",
                "command": [
                    "uv", "run", "python", "-m",
                    "conformal_marl.autoresearch.evaluate",
                    "--tier", "learning", "--num-seeds", str(learning_seeds),
                    "--output", "{output}",
                ],
                "required_status": "ok",
            },
        ],
        "task_context_file": str(task_context_file),
        "output_dir": str(output_dir.resolve()),
        "branching_factor": branching_factor,
        "max_nodes": max_nodes,
        "timeout_seconds": 3_600,
    }


def write_search_config(config: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
