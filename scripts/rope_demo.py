"""Demo of how RoPE (rotary position embeddings) is used inside the model's attention.

Run with:  uv run python scripts/rope_demo.py
"""

import math
from pathlib import Path

import torch

from smalllm.model import ModelConfig, Transformer, apply_rope, rope_cos_sin, rotate_half

CONFIG = Path(__file__).parent.parent / "configs" / "nano.yaml"
BOOST = 20.0  # stand-in for trained, larger q/k weights (see demo_in_attention)
TOKEN = 6403  # "Once"; repeated below so that only position can tell the tokens apart

torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)


def demo_frequencies(cfg: ModelConfig) -> None:
    print("== 1. Each pair of dimensions rotates at its own speed ==")
    d = cfg.head_dim
    inv_freq = 1.0 / cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
    print(f"head_dim={d} gives {d // 2} pairs; rotation per position (radians), first/last 3:")
    print(f"  fast pairs: {inv_freq[:3].tolist()}")
    print(f"  slow pairs: {[f'{v:.2e}' for v in inv_freq[-3:].tolist()]}")
    print(f"  full turn takes {2 * math.pi / inv_freq[0].item():.1f} tokens (fastest pair)")
    print(f"  full turn takes {2 * math.pi / inv_freq[-1].item():,.0f} tokens (slowest pair)")
    print("  Fast pairs tell neighbours apart; slow pairs track long distances.")


def demo_layout() -> None:
    print("\n== 2. Which dimensions are paired (Hugging Face layout) ==")
    d = 8
    x = torch.arange(d, dtype=torch.float32).view(1, 1, 1, d)  # values 0..7 label the dimensions
    print(f"x               = {x.flatten().tolist()}")
    print(f"rotate_half(x)  = {rotate_half(x).flatten().tolist()}")
    print("  dim i is paired with dim i + head_dim/2 (0<->4, 1<->5, ...), NOT with its neighbour.")
    print("  Published weights were trained with this layout, so it must match exactly.")


def demo_one_pair(cfg: ModelConfig) -> None:
    print("\n== 3. RoPE is a 2D rotation of each pair ==")
    d = cfg.head_dim
    torch.manual_seed(0)
    x = torch.randn(1, 1, 1, d)
    pos = 5
    cos, sin = rope_cos_sin(torch.tensor([pos]), d, cfg.rope_theta)
    y = apply_rope(x, cos, sin)

    angle = pos * 1.0  # pair 0 has frequency 1.0, so it turns by `pos` radians
    a, b = x[0, 0, 0, 0].item(), x[0, 0, 0, d // 2].item()
    by_hand = (a * math.cos(angle) - b * math.sin(angle), a * math.sin(angle) + b * math.cos(angle))
    print(f"pair 0 before:            ({a:.4f}, {b:.4f})")
    print(
        f"pair 0 after (RoPE):      ({y[0, 0, 0, 0].item():.4f}, {y[0, 0, 0, d // 2].item():.4f})"
    )
    print(f"pair 0 rotated by hand:   ({by_hand[0]:.4f}, {by_hand[1]:.4f})  (angle = {angle} rad)")
    print(
        f"vector length before/after: {x.norm().item():.4f} / {y.norm().item():.4f}  (rotation keeps it)"
    )


def demo_relative(cfg: ModelConfig) -> None:
    print("\n== 4. Scores depend only on the distance between positions ==")
    d, t = cfg.head_dim, 6
    torch.manual_seed(0)
    q = torch.randn(d)  # the SAME query vector placed at every position
    k = torch.randn(d)  # the SAME key vector placed at every position
    cos, sin = rope_cos_sin(torch.arange(t), d, cfg.rope_theta)
    qs = apply_rope(q.expand(1, 1, t, d), cos, sin)[0, 0]
    ks = apply_rope(k.expand(1, 1, t, d), cos, sin)[0, 0]
    scores = qs @ ks.T  # scores[m, n]: query at position m against key at position n
    print("score(query at m, key at n); rows = m, columns = n:")
    print(scores)
    same = torch.allclose(scores[2, 1], scores[5, 4], atol=1e-4)
    print(f"score(m=2, n=1) == score(m=5, n=4), both distance 1: {same}")
    print("  Every diagonal is constant: only m - n matters, not the absolute positions.")


def demo_in_attention(cfg: ModelConfig, model: Transformer) -> None:
    print("\n== 5. Inside attention: same token repeated, so only RoPE can tell positions apart ==")
    attn = model.model.layers[0].self_attn
    t = 5
    ids = torch.full((1, t), TOKEN)
    # The block feeds attention the normalised embeddings, so do the same here.
    x = model.model.layers[0].input_layernorm(model.model.embed_tokens(ids))
    b = 1

    q = attn.q_proj(x).view(b, t, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(b, t, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    # Freshly initialised weights are tiny, so every score is ~0 and softmax is flat either way.
    # Scale q and k up (as training eventually would) so the effect of position is visible.
    q, k = q * BOOST, k * BOOST
    print(f"q: {tuple(q.shape)} (batch, heads, tokens, head_dim); k: {tuple(k.shape)}")

    cos, sin = rope_cos_sin(torch.arange(t), cfg.head_dim, cfg.rope_theta)
    scale = 1 / math.sqrt(cfg.head_dim)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool))

    def show(name: str, qq: torch.Tensor, kk: torch.Tensor) -> None:
        raw = (qq[0, 0] @ kk[0, 0].T) * scale  # head 0 (kv head 0 serves it)
        weights = raw.masked_fill(~mask, float("-inf")).softmax(-1)
        print(f"\n{name}: attention weights, head 0 (rows = query position, cols = key position)")
        print(weights)

    show("WITHOUT RoPE", q, k)
    print("  Every row is flat (uniform over the allowed tokens): identical tokens are")
    print("  indistinguishable, so attention has no idea about word order.")
    show("WITH RoPE", apply_rope(q, cos, sin), apply_rope(k, cos, sin))
    print("  Rows now differ and each diagonal repeats: attention depends on distance.")


def demo_matches_module(cfg: ModelConfig, model: Transformer) -> None:
    print("\n== 6. Doing it by hand reproduces the Attention module ==")
    attn = model.model.layers[0].self_attn
    t = 5
    torch.manual_seed(1)
    x = torch.randn(1, t, cfg.hidden_size)
    cos, sin = rope_cos_sin(torch.arange(t), cfg.head_dim, cfg.rope_theta)

    h, kvh, d = attn.n_heads, attn.n_kv_heads, attn.head_dim
    q = attn.q_proj(x).view(1, t, h, d).transpose(1, 2)
    k = attn.k_proj(x).view(1, t, kvh, d).transpose(1, 2)
    v = attn.v_proj(x).view(1, t, kvh, d).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # RoPE touches q and k, never v
    k, v = k.repeat_interleave(h // kvh, dim=1), v.repeat_interleave(h // kvh, dim=1)

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    scores = scores.masked_fill(~torch.tril(torch.ones(t, t, dtype=torch.bool)), float("-inf"))
    out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, t, -1)
    by_hand = attn.o_proj(out)

    print(
        f"max |by hand - Attention module| = {(by_hand - attn(x, cos, sin)).abs().max().item():.2e}"
    )


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig.from_yaml(CONFIG)
    model = Transformer(cfg)

    demo_frequencies(cfg)
    demo_layout()
    demo_one_pair(cfg)
    demo_relative(cfg)
    demo_in_attention(cfg, model)
    demo_matches_module(cfg, model)


if __name__ == "__main__":
    main()
