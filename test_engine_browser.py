import asyncio
import torch
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig

def test():
    tokenizer = get_tokenizer()
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=128, vocab_size=tokenizer.get_vocab_size())
    model = GPT(config)
    model.eval()
    engine = Engine(model, tokenizer)

    action_json = '{"action": "navigate", "params": {"url": "https://example.com"}}'
    bos = tokenizer.get_bos_token_id()
    
    # We want to feed action_start and action_json, then see if it generates action_end and observation_start
    # To make it deterministic, we'll manually step it.
    
    action_start = tokenizer.encode_special("<|action_start|>")
    action_end = tokenizer.encode_special("<|action_end|>")
    observation_start = tokenizer.encode_special("<|observation_start|>")
    
    tokens = [bos] + [action_start] + tokenizer.encode(action_json) + [action_end]
    
    print(f"Feeding tokens up to action_end...")
    gen = engine.generate(tokens, num_samples=1, max_tokens=50)
    
    # The first token generated should be observation_start (forced)
    token_column, token_masks = next(gen)
    token = token_column[0]
    token_str = tokenizer.decode([token])
    print(f"First gen token: {token}, mask={token_masks[0]}, str='{token_str}'")
    
    if token == observation_start:
        print("SUCCESS: observation_start was correctly injected!")
    else:
        print(f"FAILURE: expected {observation_start}, got {token}")

    # Next tokens should be the observation result (forced)
    for i in range(5):
        token_column, token_masks = next(gen)
        token = token_column[0]
        token_str = tokenizer.decode([token])
        print(f"Gen {i+1}: token={token}, mask={token_masks[0]}, str='{token_str}'")

    print("Done.")

if __name__ == "__main__":
    test()
