"""
Upload a static repo analysis to Weights & Biases (Models).

Requires `wandb login` or `WANDB_API_KEY`. Uses project `nanochat-analysis` by default
(override with `WANDB_PROJECT` or `--wandb-project`).

  python -m scripts.wandb_project_analysis --run repo-snapshot-1
  python -m scripts.wandb_project_analysis --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import wandb

from nanochat.project_analysis import gather_project_facts, render_markdown_report


def _resolve_wandb_init_kwargs(
    default_project: str,
    entity: str | None = None,
    project: str | None = None,
) -> dict:
    """Same semantics as nanochat.common.resolve_wandb_init_kwargs (no torch import)."""
    resolved_entity = (entity or os.environ.get("WANDB_ENTITY") or "").strip() or None
    resolved_project = (project or os.environ.get("WANDB_PROJECT") or default_project).strip()
    return {"entity": resolved_entity, "project": resolved_project}


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(20):
        if (cur / "pyproject.toml").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Post nanochat repo analysis to W&B")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repo root (default: directory containing pyproject.toml, searching upward from cwd)",
    )
    parser.add_argument("--run", type=str, default="project-analysis", help="W&B run name")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Markdown report and skip wandb.init",
    )
    args = parser.parse_args()

    root = args.root or _find_repo_root(Path.cwd())
    facts = gather_project_facts(root)
    report = render_markdown_report(facts)

    if args.dry_run:
        print(report)
        return 0

    if not os.environ.get("WANDB_API_KEY"):
        # wandb may still use ~/.netrc or prior login; continue and let wandb fail clearly
        pass

    kw = _resolve_wandb_init_kwargs("nanochat-analysis", args.wandb_entity, args.wandb_project)
    run = wandb.init(
        name=args.run,
        job_type="project_analysis",
        tags=["nanochat", "repo-snapshot", "analysis"],
        notes="Automated static analysis of the nanochat repo (see artifact report).",
        **kw,
    )

    for k, v in facts.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            run.config[k] = v

    run.log(
        {
            "analysis/nanochat_py_files": facts.get("nanochat_py_files", 0),
            "analysis/scripts_py_files": facts.get("scripts_py_files", 0),
            "analysis/nanochat_lines": facts.get("nanochat_lines", 0),
            "analysis/scripts_lines": facts.get("scripts_lines", 0),
            "analysis/dependency_count": facts.get("dependency_count", 0),
            "analysis/git_dirty_files": facts.get("git_dirty_files", 0),
        }
    )

    with tempfile.TemporaryDirectory() as td:
        report_path = Path(td) / "PROJECT_ANALYSIS.md"
        report_path.write_text(report, encoding="utf-8")
        art = wandb.Artifact(name="project-analysis", type="report", description="Repo snapshot Markdown")
        art.add_file(str(report_path), name="PROJECT_ANALYSIS.md")
        run.log_artifact(art)

    url = run.url
    run.finish()
    print(f"W&B run finished: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
