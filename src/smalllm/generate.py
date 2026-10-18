"""Autoregressive sampling (temperature, top-k, top-p). No KV cache: simple but O(T^2)."""

from __future__ import annotations

import argparse

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from smalllm.device import describe, pick_device, pick_dtype
from smalllm.model import ModelConfig, Transformer

TOKENIZER_REPO = "HuggingFaceTB/SmolLM2-135M"


def sample_next(
    logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None
) -> torch.Tensor:
    """logits: (B, vocab) for the last position -> (B, 1) sampled token ids."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)  # greedy
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        # Drop tokens once the cumulative probability *before* them already exceeds top_p,
        # so the first token that crosses the threshold is kept.
        drop = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


@torch.no_grad()
def generate(
    model: Transformer,
    ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        # Crop to the context window; RoPE positions restart at 0 for the cropped window.
        window = ids[:, -model.cfg.max_position_embeddings :]
        next_id = sample_next(model(window)[:, -1, :], temperature, top_k, top_p)
        ids = torch.cat((ids, next_id), dim=1)
    return ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/nano.yaml")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(device)
    print(describe(device, dtype))

    tok = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))
    # Random init unless/until weights are loaded: expect gibberish.
    model = Transformer(ModelConfig.from_yaml(args.config)).to(device=device, dtype=dtype)

    ids = torch.tensor([tok.encode(args.prompt).ids], device=device)
    out = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
