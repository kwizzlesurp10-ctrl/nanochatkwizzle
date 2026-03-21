from pathlib import Path
import sys

# Allow `python tests/test_wandb_config.py` (any interpreter); pytest already sets pythonpath when run as `pytest`.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest

from nanochat.common import resolve_wandb_init_kwargs


def test_resolve_wandb_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    kw = resolve_wandb_init_kwargs("nanochat-sft")
    assert kw["entity"] is None
    assert kw["project"] == "nanochat-sft"


def test_resolve_wandb_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "kwizzlesurp10-sevtech")
    monkeypatch.setenv("WANDB_PROJECT", "custom-sft")
    kw = resolve_wandb_init_kwargs("nanochat-sft")
    assert kw["entity"] == "kwizzlesurp10-sevtech"
    assert kw["project"] == "custom-sft"


def test_resolve_wandb_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "env-entity")
    monkeypatch.setenv("WANDB_PROJECT", "env-project")
    kw = resolve_wandb_init_kwargs("nanochat", entity="cli-entity", project="cli-project")
    assert kw["entity"] == "cli-entity"
    assert kw["project"] == "cli-project"


def test_resolve_wandb_empty_entity_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "   ")
    kw = resolve_wandb_init_kwargs("nanochat")
    assert kw["entity"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
