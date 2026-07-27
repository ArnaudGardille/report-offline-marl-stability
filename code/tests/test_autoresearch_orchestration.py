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
    (orchestrator / "launch_hierarchical_campaign.py").write_text("")
    (repo / "code/conformal_marl/autoresearch/frozen").mkdir(parents=True)
    for path in (
        "code/conformal_marl/autoresearch/api.py",
        "code/conformal_marl/autoresearch/evaluate.py",
        "code/conformal_marl/iql.py",
        "code/conformal_marl/networks.py",
        "code/conformal_marl/conformal.py",
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
    candidate = repo / "code/conformal_marl/autoresearch/candidate"
    candidate.mkdir()
    (candidate / "__init__.py").write_text("")
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
    shared_python = tmp_path / "venv-python"
    shared_python.write_text("")
    config = build_search_config(
        repository=repo,
        orchestrator=orchestrator,
        base_ref="HEAD",
        output_dir=tmp_path / "results",
        task_context_file=context,
        shared_python=shared_python,
        max_nodes=4,
        initial_capacity=2,
        learning_seeds=3,
        validation_trials=16,
        validation_seeds=4,
        finalist_count=1,
    )
    assert Path(config["repository"]).is_absolute()
    assert Path(config["task_context_file"]).is_absolute()
    assert Path(config["shared_python"]).is_absolute()
    assert config["shared_python"] == str(shared_python)
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert config["base_ref"] == expected_commit
    assert [stage["name"] for stage in config["stages"]] == [
        "operator", "designed_diagnostic", "mechanism", "learning"
    ]
    assert config["stages"][0]["minimum_score"] == pytest.approx(-0.25)
    assert "{stage_config}" in config["stages"][1]["command"]
    assert config["stages"][1]["contributes_to_priority"] is False
    assert "{stage_config}" not in config["stages"][2]["command"]
    learning_command = config["stages"][3]["command"]
    seed_option = learning_command.index("--num-seeds")
    assert learning_command[seed_option + 1] == "3"
    assert config["stages"][3]["minimum_score"] == pytest.approx(0.001)
    assert config["campaign"]["initial_capacity"] == 2
    assert config["campaign"]["max_nodes"] == 4
    assert config["campaign"]["finalist_count"] == 1
    assert config["campaign"]["experiment_stage_name"] == "designed_diagnostic"
    assert [stage["name"] for stage in config["final_stages"]] == [
        "heldout_mechanism", "heldout_learning"
    ]
    heldout_command = config["final_stages"][1]["command"]
    assert "heldout" in config["final_stages"][0]["command"]
    assert "heldout" in heldout_command
    heldout_seed_option = heldout_command.index("--num-seeds")
    assert heldout_command[heldout_seed_option + 1] == "4"
    assert config["final_stages"][1]["minimum_score"] == pytest.approx(0.001)
    assert len(config["research_lenses"]) == 5


def test_build_search_config_rejects_unknown_ref(tmp_path: Path):
    repo, orchestrator, context = _init_fixture(tmp_path)
    shared_python = tmp_path / "venv-python"
    shared_python.write_text("")
    with pytest.raises(ValueError, match="does not exist"):
        build_search_config(
            repository=repo,
            orchestrator=orchestrator,
            base_ref="missing-branch",
            output_dir=tmp_path / "results",
            task_context_file=context,
            shared_python=shared_python,
        )


def test_build_search_config_rejects_heldout_model_selection(tmp_path: Path):
    repo, orchestrator, context = _init_fixture(tmp_path)
    shared_python = tmp_path / "venv-python"
    shared_python.write_text("")
    with pytest.raises(ValueError, match="invalid search or evaluation budget"):
        build_search_config(
            repository=repo,
            orchestrator=orchestrator,
            base_ref="HEAD",
            output_dir=tmp_path / "results",
            task_context_file=context,
            shared_python=shared_python,
            finalist_count=2,
        )
