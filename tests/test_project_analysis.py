"""Tests for nanochat.project_analysis."""

from __future__ import annotations

from pathlib import Path

from nanochat.project_analysis import gather_project_facts, render_markdown_report


def test_gather_project_facts_has_expected_keys() -> None:
    root = Path(__file__).resolve().parents[1]
    facts = gather_project_facts(root)
    for key in (
        "analysis_utc",
        "repo_root",
        "git_commit",
        "git_branch",
        "nanochat_py_files",
        "scripts_py_files",
        "training_scripts_present",
    ):
        assert key in facts
    assert facts["nanochat_py_files"] >= 1
    assert "base_train.py" in facts["training_scripts_present"]


def test_render_markdown_includes_core_sections() -> None:
    md = render_markdown_report(
        {
            "analysis_utc": "2026-01-01T00:00:00+00:00",
            "repo_root": "/tmp/r",
            "git_commit": "abc",
            "git_branch": "main",
            "git_dirty_files": 0,
            "pyproject_name": "nanochat",
            "pyproject_version": "0.1.0",
            "dependency_count": 5,
            "nanochat_py_files": 10,
            "scripts_py_files": 3,
            "nanochat_lines": 100,
            "scripts_lines": 50,
            "training_scripts_present": "base_train.py",
        }
    )
    assert "Git" in md
    assert "abc" in md
    assert "base_train.py" in md
