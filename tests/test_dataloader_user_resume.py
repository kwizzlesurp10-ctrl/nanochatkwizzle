from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest

from nanochat.dataloader import _should_prepend_user_documents


def test_prepend_only_fresh_train_with_path() -> None:
    assert _should_prepend_user_documents("train", None, "/tmp/x.jsonl") is True


def test_no_prepend_on_resume_even_epoch_1() -> None:
    resume = {"pq_idx": 2, "rg_idx": 0, "epoch": 1}
    assert _should_prepend_user_documents("train", resume, "/tmp/x.jsonl") is False


def test_no_prepend_val_or_no_path() -> None:
    assert _should_prepend_user_documents("val", None, "/tmp/x.jsonl") is False
    assert _should_prepend_user_documents("train", None, "") is False
    assert _should_prepend_user_documents("train", None, None) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
