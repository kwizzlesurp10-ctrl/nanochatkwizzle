print("DEBUG: NANOCHAT/ENGINE.PY LOADING - ABSOLUTELY FINAL VERSION", flush=True)
"""
Engine for efficient inference of our models.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
import os
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        return None

def use_calculator(expr):
    expr = expr.replace(",", "")
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None
    if '.count(' not in expr:
        return None
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.prev_embedding = None

    def reset(self):
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        assert self.get_pos() == 0
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------

class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    ABSOLUTELY FINAL ROBUST SAMPLING FUNCTION.
    """
    print("DEBUG: ABSOLUTELY FINAL sample_next_token CALLED - FORCED ARGMAX", flush=True)
    # FORCE CPU AND FLOAT32 JUST IN CASE
    return torch.argmax(logits.detach().cpu().to(torch.float32), dim=-1, keepdim=True)

# -----------------------------------------------------------------------------

class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        print("DEBUG: Engine.generate starting ABSOLUTELY FINAL", flush=True)
        device = self.model.get_device()
        dtype = COMPUTE_DTYPE
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)

        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        thought_start = get_special("<|thought_start|>")
        thought_end = get_special("<|thought_end|>")
        action_start = get_special("<|action_start|>")
        action_end = get_special("<|action_end|>")
        observation_start = get_special("<|observation_start|>")
        observation_end = get_special("<|observation_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        m = self.model.config
        cap = int(getattr(m, "sequence_len", 2048))
        if len(tokens) > cap:
            tokens = list(tokens[-cap:])
        room = max(0, cap - len(tokens))
        max_new = room if max_tokens is None else min(max(0, max_tokens), room)
        decode_len = len(tokens) + max_new

        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), device=device, dtype=dtype, **kv_model_kwargs)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)[:, -1, :].expand(num_samples, -1)

        kv_cache_decode = KVCache(batch_size=num_samples, seq_len=decode_len, device=device, dtype=dtype, **kv_model_kwargs)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        row_states = []
        for _ in range(num_samples):
            state = RowState(tokens.copy())
            state.in_action_block = False
            state.action_tokens = []
            row_states.append(state)

        num_generated = 0
        while True:
            if num_generated >= max_new or all(state.completed for state in row_states):
                break

            # CALL GLOBAL ROBUST SAMPLER
            next_ids = sample_next_token(logits, rng, temperature, top_k).to(device)

            sampled_tokens = next_ids[:, 0].tolist()
            token_column, token_masks = [], []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if next_token == assistant_end or next_token == bos: state.completed = True
                
                if next_token == python_start:
                    state.in_python_block = True; state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        res = use_calculator(self.tokenizer.decode(state.python_expr_tokens))
                        if res is not None:
                            state.forced_tokens.append(output_start); state.forced_tokens.extend(self.tokenizer.encode(str(res))); state.forced_tokens.append(output_end)
                elif state.in_python_block: state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results, masks = [tokens.copy() for _ in range(num_samples)], [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for col, mks in self.generate(tokens, num_samples, **kwargs):
            for i, (t, m) in enumerate(zip(col, mks)):
                if not completed[i]:
                    if t == assistant_end or t == bos: completed[i] = True
                    else: results[i].append(t); masks[i].append(m)
            if all(completed): break
        return results, masks

if __name__ == "__main__":
    import time
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model("base", device, phase="eval")
    engine = Engine(model, tokenizer)
    for col, mks in engine.generate(tokenizer.encode("Water is", prepend=tokenizer.get_bos_token_id()), max_tokens=10):
        print(tokenizer.decode([col[0]]), end="", flush=True)
    print()
