import torch

from nanochat.gpt import GPT, GPTConfig


def test_set_window_pattern_recomputes_window_sizes():
    c = GPTConfig(
        sequence_len=512,
        vocab_size=256,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="SSSL",
    )
    with torch.device("meta"):
        m = GPT(c)
    short = -(-512 // 4 // 128) * 128
    assert m.window_sizes[0][0] == short
    m.set_window_pattern("L")
    assert m.config.window_pattern == "L"
    assert all(ws[0] == 512 for ws in m.window_sizes)
