"""Generate and validate AI Scientist's constrained MARL search configuration."""

from __future__ import annotations

import json
import os
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
    shared_python: Path,
    max_nodes: int = 12,
    initial_capacity: int = 5,
    revisions_per_expansion: int = 2,
    max_per_family: int = 2,
    operator_trials: int = 128,
    learning_seeds: int = 8,
    validation_trials: int = 128,
    validation_seeds: int = 32,
    finalist_count: int = 1,
) -> dict:
    """Build a portable constrained-search config after strict local preflight."""

    repository = repository.resolve()
    orchestrator = orchestrator.resolve()
    task_context_file = task_context_file.resolve()
    # Preserve the virtualenv executable path. Resolving its symlink points at uv's
    # bare CPython installation, which no longer discovers the venv site-packages.
    shared_python = Path(os.path.abspath(shared_python))
    candidate = repository / "code/conformal_marl/autoresearch/candidate"
    launcher = orchestrator / "launch_hierarchical_campaign.py"

    if not (repository / ".git").exists() and not _git_ref_exists(repository, "HEAD"):
        raise ValueError(f"not a Git repository: {repository}")
    if not _git_ref_exists(repository, base_ref):
        raise ValueError(f"Git ref {base_ref!r} does not exist in {repository}")
    if not candidate.is_dir():
        raise ValueError(f"candidate contract is missing: {candidate}")
    if not launcher.is_file():
        raise ValueError(f"constrained AI Scientist launcher is missing: {launcher}")
    if not task_context_file.is_file():
        raise ValueError(f"task context is missing: {task_context_file}")
    if not shared_python.is_file():
        raise ValueError(f"shared Python interpreter is missing: {shared_python}")
    if (
        max_nodes < 2
        or initial_capacity < 1
        or initial_capacity > max_nodes
        or revisions_per_expansion < 1
        or max_per_family < 1
        or operator_trials < 2
        or learning_seeds < 2
        or validation_trials < 4
        or validation_seeds < 4
        or finalist_count != 1
    ):
        raise ValueError("invalid search or evaluation budget")

    missing_frozen = [path for path in FROZEN_PATHS if not (repository / path).exists()]
    if missing_frozen:
        raise ValueError(f"frozen benchmark paths are missing: {missing_frozen}")

    return {
        "repository": str(repository),
        "base_ref": base_ref,
        "candidate_path": "code/conformal_marl/autoresearch/candidate",
        "working_subdir": "code",
        "shared_python": str(shared_python),
        "frozen_paths": list(FROZEN_PATHS),
        "stages": [
            {
                "name": "operator",
                "command": [
                    "{python}", "-m",
                    "conformal_marl.autoresearch.evaluate",
                    "--tier", "operator", "--trials", str(operator_trials),
                    "--output", "{output}",
                ],
                "minimum_score": -0.25,
                "required_status": "ok",
            },
            {
                "name": "designed_diagnostic",
                "command": [
                    "{python}", "-m",
                    "conformal_marl.autoresearch.evaluate",
                    "--tier", "mechanism", "--trials", "32",
                    "--mechanism-config", "{stage_config}",
                    "--output", "{output}",
                ],
                "contributes_to_priority": False,
                "required_status": "ok",
            },
            {
                "name": "mechanism",
                "command": [
                    "{python}", "-m",
                    "conformal_marl.autoresearch.evaluate",
                    "--tier", "mechanism", "--trials", "32",
                    "--output", "{output}",
                ],
                "minimum_score": -0.5,
                "required_status": "ok",
            },
            {
                "name": "learning",
                "command": [
                    "{python}", "-m",
                    "conformal_marl.autoresearch.evaluate",
                    "--tier", "learning", "--num-seeds", str(learning_seeds),
                    "--output", "{output}",
                ],
                "minimum_score": 0.001,
                "required_status": "ok",
            },
        ],
        "final_stages": [
            {
                "name": "heldout_mechanism",
                "command": [
                    "{python}", "-m", "conformal_marl.autoresearch.evaluate",
                    "--tier", "mechanism", "--seed", "20260713",
                    "--trials", str(validation_trials), "--output", "{output}",
                ],
                "minimum_score": -0.5,
                "required_status": "ok",
            },
            {
                "name": "heldout_learning",
                "command": [
                    "{python}", "-m", "conformal_marl.autoresearch.evaluate",
                    "--tier", "learning", "--seed", "20260713",
                    "--num-seeds", str(validation_seeds), "--output", "{output}",
                ],
                "minimum_score": 0.001,
                "required_status": "ok",
            },
        ],
        "research_lenses": [
            {
                "name": "statistical-estimation",
                "instruction": "Derive an estimator from the conditional bias and variance structure.",
            },
            {
                "name": "counterfactual-causal",
                "instruction": "Exploit stationary dynamics conditional on observed joint actions.",
            },
            {
                "name": "game-theoretic",
                "instruction": "Address teammate adaptation and cooperative equilibrium selection.",
            },
            {
                "name": "coverage-exploration",
                "instruction": "Improve information coverage without starving rare useful support.",
            },
            {
                "name": "robust-control",
                "instruction": "Optimize against bounded teammate-policy drift and misspecification.",
            },
        ],
        "task_context_file": str(task_context_file),
        "output_dir": str(output_dir.resolve()),
        "campaign": {
            "initial_capacity": initial_capacity,
            "max_nodes": max_nodes,
            "revisions_per_expansion": revisions_per_expansion,
            "max_per_family": max_per_family,
            "min_conceptual_distance": 0.2,
            "finalist_count": finalist_count,
            "experiment_stage_name": "designed_diagnostic",
        },
        "timeout_seconds": 3_600,
        "model_timeout_seconds": 900,
    }


def write_search_config(config: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
