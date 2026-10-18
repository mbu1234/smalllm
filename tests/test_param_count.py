from pathlib import Path

import pytest
import torch

from smalllm.model import ModelConfig, Transformer

CONFIGS = Path(__file__).parent.parent / "configs"


def expected_params(c: ModelConfig) -> int:
    """Closed-form count: tied embedding counted once, no biases."""
    q_out, kv_out = c.num_heads * c.head_dim, c.num_kv_heads * c.head_dim
    attn = c.hidden_size * (q_out + 2 * kv_out) + q_out * c.hidden_size
    mlp = 3 * c.hidden_size * c.intermediate_size
    per_layer = attn + mlp + 2 * c.hidden_size
    return c.vocab_size * c.hidden_size + c.num_layers * per_layer + c.hidden_size


@pytest.mark.parametrize("name", ["nano", "tiny"])
def test_param_count_matches_formula(name: str) -> None:
    cfg = ModelConfig.from_yaml(CONFIGS / f"{name}.yaml")
    assert Transformer(cfg).num_parameters() == expected_params(cfg)


def test_smollm2_135m_param_count() -> None:
    """Published SmolLM2-135M has 134,515,008 parameters (tied embeddings)."""
    cfg = ModelConfig.from_yaml(CONFIGS / "smollm2_135m.yaml")
    with torch.device("meta"):  # shapes only, no 540 MB allocation
        model = Transformer(cfg)
    assert model.num_parameters() == 134_515_008
