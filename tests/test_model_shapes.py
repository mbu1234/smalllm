from pathlib import Path

import pytest
import torch

from smalllm.device import pick_device
from smalllm.model import ModelConfig, Transformer

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="module")
def nano() -> Transformer:
    torch.manual_seed(0)
    return Transformer(ModelConfig.from_yaml(CONFIGS / "nano.yaml"))


def test_forward_shape(nano: Transformer) -> None:
    ids = torch.randint(0, nano.cfg.vocab_size, (3, 17))
    logits = nano(ids)
    assert logits.shape == (3, 17, nano.cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_initial_loss_near_uniform(nano: Transformer) -> None:
    ids = torch.randint(0, nano.cfg.vocab_size, (4, 32))
    # Independent targets: with tied embeddings, targets == inputs would start below ln(vocab)
    # because each token's own embedding scores highly against itself.
    targets = torch.randint(0, nano.cfg.vocab_size, (4, 32))
    logits = nano(ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    assert abs(loss.item() - torch.log(torch.tensor(float(nano.cfg.vocab_size))).item()) < 0.5


def test_weights_are_tied(nano: Transformer) -> None:
    assert nano.lm_head.weight is nano.model.embed_tokens.weight


def test_parameter_names_match_hf_llama(nano: Transformer) -> None:
    names = set(nano.state_dict())
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(nano.cfg.num_layers):
        p = f"model.layers.{i}."
        expected |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight"}
        expected |= {p + f"self_attn.{n}_proj.weight" for n in "qkvo"}
        expected |= {p + f"mlp.{n}_proj.weight" for n in ("gate", "up", "down")}
    assert names == expected


def test_residual_projections_are_scaled(nano: Transformer) -> None:
    layer = nano.model.layers[0]
    expected = 0.02 / (2 * nano.cfg.num_layers) ** 0.5
    assert layer.self_attn.o_proj.weight.std().item() == pytest.approx(expected, rel=0.1)
    assert layer.self_attn.q_proj.weight.std().item() == pytest.approx(0.02, rel=0.1)


def test_pick_device_cpu_override() -> None:
    assert pick_device("cpu").type == "cpu"
