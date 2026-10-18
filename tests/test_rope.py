import torch

from smalllm.model import apply_rope, rope_cos_sin, rotate_half

HEAD_DIM, THETA = 64, 100000.0


def scores(q: torch.Tensor, k: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor):
    cq, sq = rope_cos_sin(q_pos, HEAD_DIM, THETA)
    ck, sk = rope_cos_sin(k_pos, HEAD_DIM, THETA)
    return apply_rope(q, cq, sq) @ apply_rope(k, ck, sk).transpose(-1, -2)


def test_relative_position_property() -> None:
    """Shifting q and k positions by the same offset leaves attention scores unchanged."""
    torch.manual_seed(0)
    t = 12
    q, k = torch.randn(1, 2, t, HEAD_DIM), torch.randn(1, 2, t, HEAD_DIM)
    pos = torch.arange(t)
    base = scores(q, k, pos, pos)
    for offset in (1, 37, 1000):
        shifted = scores(q, k, pos + offset, pos + offset)
        torch.testing.assert_close(base, shifted, atol=1e-3, rtol=1e-4)


def test_absolute_positions_do_matter_when_shifted_unequally() -> None:
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 4, HEAD_DIM), torch.randn(1, 1, 4, HEAD_DIM)
    pos = torch.arange(4)
    assert not torch.allclose(scores(q, k, pos, pos), scores(q, k, pos + 5, pos), atol=1e-3)


def test_position_zero_is_identity() -> None:
    x = torch.randn(1, 1, 1, HEAD_DIM)
    cos, sin = rope_cos_sin(torch.tensor([0]), HEAD_DIM, THETA)
    torch.testing.assert_close(apply_rope(x, cos, sin), x)


def test_rope_preserves_norm() -> None:
    x = torch.randn(2, 3, 8, HEAD_DIM)
    cos, sin = rope_cos_sin(torch.arange(8), HEAD_DIM, THETA)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_rotate_half_pairs_first_and_second_half() -> None:
    """HF layout: (x1, x2) -> (-x2, x1) with halves, not interleaved even/odd pairs."""
    x = torch.arange(8.0)
    assert rotate_half(x).tolist() == [-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]
