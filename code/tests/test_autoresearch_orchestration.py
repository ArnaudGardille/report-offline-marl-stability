"""Tests for portable constrained-search configuration generation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conformal_marl.autoresearch.orchestration import build_search_config


def _init_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    orchestrator = tmp_path / "orchestrator"
    context = tmp_path / "task.md"
    repo.mkdir()
    orchestrator.mkdir()
    context.write_text("task")
    (orchestrator / "launch_constrained_bfts.py").write_text("")
    (repo / "code/conformal_marl/autoresearch/frozen").mkdir(parents=True)
    for path in (
        "code/conformal_marl/autoresearch/api.py",
        "code/conformal_marl/autoresearch/evaluate.py",
        "code/conformal_marl/autoresearch/candidate.py",
        "code/conformal_marl/iql.py",
        "code/conformal_marl/replay.py",
        "code/conformal_marl/samplers.py",
        "code/tests/test_autoresearch_operator.py",
        "code/tests/test_autoresearch_learning.py",
        "code/pyproject.toml",
        "code/uv.lock",
    ):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
    (repo / "code/conformal_marl/envs").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
            "commit", "-qm", "fixture",
        ],
        cwd=repo,
        check=True,
    )
    return repo, orchestrator, context


def test_build_search_config_uses_absolute_paths_and_staged_gate(tmp_path: Path):
    repo, orchestrator, context = _init_fixture(tmp_path)
    config = build_search_config(
        repository=repo,
        orchestrator=orchestrator,
        base_ref="HEAD",
        output_dir=tmp_path / "results",
        task_context_file=context,
        max_nodes=4,
        learning_seeds=3,
    )
    assert Path(config["repository"]).is_absolute()
    assert Path(config["task_context_file"]).is_absolute()
    assert [stage["name"] for stage in config["stages"]] == ["operator", "learning"]
    assert config["stages"][0]["minimum_score"] == pytest.approx(-0.25)
    learning_command = config["stages"][1]["command"]
    seed_option = learning_command.index("--num-seeds")
    assert learning_command[seed_option + 1] == "3"


def test_build_search_config_rejects_unknown_ref(tmp_path: Path):
    repo, orchestrator, context = _init_fixture(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        build_search_config(
            repository=repo,
            orchestrator=orchestrator,
            base_ref="missing-branch",
            output_dir=tmp_path / "results",
            task_context_file=context,
        )
