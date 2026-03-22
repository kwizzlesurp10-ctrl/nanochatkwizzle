"""
Static repo facts for W&B project snapshots (no network).
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from pathlib import Path
from typing import Any


TRAINING_SCRIPT_NAMES = (
    "base_train.py",
    "chat_sft.py",
    "chat_rl.py",
    "tok_train.py",
    "distill.py",
)


def _count_lines(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        try:
            total += sum(1 for _ in p.open("rb"))
        except OSError:
            continue
    return total


def _git_text(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def gather_project_facts(root: Path) -> dict[str, Any]:
    root = root.resolve()
    pyproject_path = root / "pyproject.toml"
    meta: dict[str, Any] = {
        "analysis_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
    }

    if pyproject_path.is_file():
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        proj = data.get("project") or {}
        meta["pyproject_name"] = proj.get("name", "")
        meta["pyproject_version"] = proj.get("version", "")
        deps = proj.get("dependencies") or []
        meta["dependency_count"] = len(deps)

    commit = _git_text(root, "rev-parse", "--short", "HEAD")
    branch = _git_text(root, "branch", "--show-current")
    dirty = _git_text(root, "status", "--porcelain")
    meta["git_commit"] = commit or "unknown"
    meta["git_branch"] = branch or "unknown"
    meta["git_dirty_files"] = len([l for l in (dirty or "").splitlines() if l.strip()])

    nanochat_dir = root / "nanochat"
    scripts_dir = root / "scripts"
    py_nano = sorted(
        p for p in nanochat_dir.rglob("*.py") if p.is_file() and "__pycache__" not in p.parts
    ) if nanochat_dir.is_dir() else []
    py_scripts = sorted(scripts_dir.glob("*.py")) if scripts_dir.is_dir() else []

    meta["nanochat_py_files"] = len(py_nano)
    meta["scripts_py_files"] = len(py_scripts)
    meta["nanochat_lines"] = _count_lines(py_nano)
    meta["scripts_lines"] = _count_lines(py_scripts)

    present_training = [n for n in TRAINING_SCRIPT_NAMES if (scripts_dir / n).is_file()]
    meta["training_scripts_present"] = ",".join(present_training)

    return meta


def render_markdown_report(facts: dict[str, Any]) -> str:
    lines = [
        "# nanochat project snapshot",
        "",
        f"- **When (UTC):** {facts.get('analysis_utc', '')}",
        f"- **Root:** `{facts.get('repo_root', '')}`",
        "",
        "## Git",
        "",
        f"- **Commit:** `{facts.get('git_commit')}`",
        f"- **Branch:** `{facts.get('git_branch')}`",
        f"- **Dirty files (porcelain lines):** {facts.get('git_dirty_files')}",
        "",
        "## Package",
        "",
        f"- **Name / version:** {facts.get('pyproject_name')} {facts.get('pyproject_version')}",
        f"- **`pyproject` dependencies:** {facts.get('dependency_count')}",
        "",
        "## Code volume",
        "",
        f"| Area | Python files | Lines (bytes read as lines) |",
        f"|------|-------------|------------------------------|",
        f"| `nanochat/` | {facts.get('nanochat_py_files')} | {facts.get('nanochat_lines')} |",
        f"| `scripts/` | {facts.get('scripts_py_files')} | {facts.get('scripts_lines')} |",
        "",
        "## Training entrypoints detected",
        "",
        f"`{facts.get('training_scripts_present', '')}`",
        "",
        "## W&B integration (repo)",
        "",
        "- Training: `scripts/base_train.py`, `scripts/chat_sft.py`, `scripts/chat_rl.py` use `wandb` + `resolve_wandb_init_kwargs`.",
        "- Defaults: projects `nanochat`, `nanochat-sft`, `nanochat-rl` (override with `WANDB_ENTITY` / `WANDB_PROJECT` or CLI).",
        "",
    ]
    return "\n".join(lines)
