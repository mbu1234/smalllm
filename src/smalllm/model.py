"""Llama-style decoder-only transformer (SmolLM2 compatible), plain PyTorch.

Module names mirror Hugging Face Llama so published weights load with `load_state_dict`:
    model.embed_tokens, model.layers.{i}.{input_layernorm, self_attn, post_attention_layernorm, mlp},
    model.norm, lm_head
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn


@dataclass
class ModelConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    num_layers: int = 30
    num_heads: int = 9
    num_kv_heads: int = 3
    intermediate_size: int = 1536
    head_dim: int = 64
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        raw = yaml.safe_load(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Normalise in float32 for stability, then cast back before applying the weight (as HF does).
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(x.dtype)


def rope_cos_sin(positions: Tensor, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    """cos/sin tables for the given integer positions, shape (T, head_dim).

    Computed on the fly (no buffer) so nothing extra lands in the state dict. `positions` is explicit
    so that callers (and the relative-position test) can shift positions by an offset.
    """
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    freqs = positions.float()[:, None] * inv_freq.to(positions.device)[None, :]
    # HF convention: frequencies are duplicated, so dims i and i+head_dim/2 share one angle.
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    # Pairs the first half of head_dim with the second half (NOT interleaved even/odd pairs).
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.num_heads % cfg.num_kv_heads == 0
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # RoPE on q and k only
        # GQA: each kv head serves n_rep consecutive query heads (same layout as HF repeat_kv).
        n_rep = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaBackbone(nn.Module):
    """Holds embed_tokens / layers / norm so parameter names start with `model.`."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = LlamaBackbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Scale residual-output projections so the residual stream variance stays bounded with depth.
        for layer in self.model.layers:
            for proj in (layer.self_attn.o_proj, layer.mlp.down_proj):
                proj.weight.data.mul_(1 / math.sqrt(2 * cfg.num_layers))
        if cfg.tie_word_embeddings:
            # Same Parameter object: the output head and the embedding are one matrix.
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, positions: Tensor | None = None) -> Tensor:
        """input_ids: (B, T) -> logits (B, T, vocab). `positions` defaults to 0..T-1."""
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        cos, sin = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_theta)
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.model.norm(x))

    def num_parameters(self) -> int:
        # parameters() de-duplicates shared tensors, so the tied matrix is counted once.
        return sum(p.numel() for p in self.parameters())
