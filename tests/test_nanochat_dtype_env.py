from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest
import torch

from nanochat.common import _dtype_from_nanochat_env


def test_bf16_alias() -> None:
    assert _dtype_from_nanochat_env("bf16") == torch.bfloat16


def test_bfloat16_and_whitespace() -> None:
    assert _dtype_from_nanochat_env("  bfloat16  ") == torch.bfloat16


def test_fp16_fp32_aliases() -> None:
    assert _dtype_from_nanochat_env("fp16") == torch.float16
    assert _dtype_from_nanochat_env("FP32") == torch.float32


def test_invalid_dtype_raises() -> None:
    with pytest.raises(ValueError, match="Invalid NANOCHAT_DTYPE"):
        _dtype_from_nanochat_env("bf161")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
