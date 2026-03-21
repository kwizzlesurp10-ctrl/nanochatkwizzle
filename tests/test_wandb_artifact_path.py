from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest

from nanochat.common import resolve_wandb_artifact_full_name


def test_resolve_artifact_full_string() -> None:
    s = "kwizzlesurp10-sevtech/nanochat/run-k9head2f-events:v0"
    assert resolve_wandb_artifact_full_name(s) == s


def test_resolve_artifact_from_parts() -> None:
    assert (
        resolve_wandb_artifact_full_name(
            None,
            entity="kwizzlesurp10-sevtech",
            project="nanochat",
            name_with_alias="run-k9head2f-events:v0",
        )
        == "kwizzlesurp10-sevtech/nanochat/run-k9head2f-events:v0"
    )


def test_resolve_artifact_rejects_short_path() -> None:
    with pytest.raises(ValueError, match="entity/project"):
        resolve_wandb_artifact_full_name("only/one")


def test_resolve_artifact_requires_parts() -> None:
    with pytest.raises(ValueError, match="Provide artifact"):
        resolve_wandb_artifact_full_name(None, entity="e", project="p", name_with_alias="")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
