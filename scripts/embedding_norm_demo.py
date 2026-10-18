"""Demo of the first two pieces of the model: the token embedding and RMSNorm.

Run with:  uv run python scripts/embedding_norm_demo.py
"""

from pathlib import Path

import torch

from smalllm.model import ModelConfig, RMSNorm, Transformer

CONFIG = Path(__file__).parent.parent / "configs" / "nano.yaml"
IDS = [6403, 1980, 253, 655]  # "Once upon a time" (see tokenize_demo.py)


def rms(x: torch.Tensor) -> list[float]:
    """Root-mean-square (typical size) of each token vector, for a (1, T, hidden) tensor."""
    return [round(v, 4) for v in x.pow(2).mean(-1).sqrt()[0].tolist()]


def demo_embedding(model: Transformer, ids: torch.Tensor) -> torch.Tensor:
    print("== Embedding ==")
    table = model.model.embed_tokens
    print(f"table shape:  {tuple(table.weight.shape)}  (vocab, hidden)")

    vectors = table(ids)
    print(f"ids shape:    {tuple(ids.shape)}  ->  vectors shape: {tuple(vectors.shape)}")

    # An embedding is a plain row lookup: no arithmetic happens.
    print(
        f"lookup is just row {IDS[0]} of the table: {torch.equal(vectors[0, 0], table.weight[IDS[0]])}"
    )
    print(f"first 6 numbers of token {IDS[0]}: {[round(v, 4) for v in vectors[0, 0, :6].tolist()]}")
    print(f"rms of each token vector: {rms(vectors)}")
    return vectors


def demo_rmsnorm(cfg: ModelConfig, vectors: torch.Tensor) -> None:
    print("\n== RMSNorm ==")
    norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    # Give the four tokens wildly different scales to see the norm even them out.
    scales = torch.tensor([1.0, 10.0, 100.0, 0.01]).view(1, 4, 1)
    scaled = vectors * scales
    normed = norm(scaled)
    print(f"rms before: {rms(scaled)}")
    print(f"rms after:  {rms(normed)}")
    print("  (the last token is tiny, so eps dominates and it stays small)")

    # Same formula written out by hand: x * rsqrt(mean(x^2) + eps), weight is 1.0 at init.
    by_hand = scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    print(f"matches the formula by hand: {torch.allclose(by_hand, normed)}")

    # Only the length changes, not the direction.
    cosine = torch.cosine_similarity(scaled[0, 0], normed[0, 0], dim=0).item()
    print(f"cosine similarity before/after for token 0: {cosine:.4f}")
    print(f"initial norm weights (first 4): {norm.weight[:4].tolist()}")


def demo_step_by_step(cfg: ModelConfig, vectors: torch.Tensor) -> None:
    """RMSNorm on one real vector (token 0), one arithmetic step at a time."""
    print(f"\n== RMSNorm, step by step on the real {cfg.hidden_size}-number vector ==")
    x = vectors[0, 0].detach()  # shape (hidden,)
    h = cfg.hidden_size

    def first6(t: torch.Tensor) -> list[float]:
        return [round(v, 4) for v in t[:6].tolist()]

    print(f"x, first 6:                {first6(x)}")

    squares = x.pow(2)
    print(f"1. squared, first 6:       {[f'{v:.6f}' for v in squares[:6].tolist()]}")
    print(f"   sum of all {h} squares:  {squares.sum().item():.6f}")

    mean_square = squares.mean()
    print(f"2. mean square (sum / {h}): {mean_square.item():.8f}")

    with_eps = mean_square + cfg.rms_norm_eps
    print(f"3. + eps:                  {with_eps.item():.8f}")
    print(f"   rms = sqrt of that:     {with_eps.sqrt().item():.6f}")

    scale = torch.rsqrt(with_eps)
    print(f"4. scale = 1 / rms:        {scale.item():.4f}")
    y = x * scale
    print(f"   x * scale, first 6:     {first6(y)}")
    print(f"   y / x, first 6:         {first6(y / x)}  (the same factor for every number)")

    print("\nbefore vs after:")
    print(
        f"   rms:            {squares.mean().sqrt().item():.4f} -> {y.pow(2).mean().sqrt().item():.4f}"
    )
    print(f"   largest |value|: {x.abs().max().item():.4f} -> {y.abs().max().item():.4f}")
    print(
        f"   euclidean length: {x.norm().item():.4f} -> {y.norm().item():.4f}  (sqrt({h}) = {h**0.5:.4f})"
    )

    module_out = RMSNorm(h, cfg.rms_norm_eps)(x)
    print(f"   RMSNorm module gives the same numbers: {torch.allclose(module_out, y, atol=1e-6)}")


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig.from_yaml(CONFIG)
    model = Transformer(cfg)
    ids = torch.tensor([IDS])

    vectors = demo_embedding(model, ids)
    demo_rmsnorm(cfg, vectors)
    demo_step_by_step(cfg, vectors)


if __name__ == "__main__":
    main()
