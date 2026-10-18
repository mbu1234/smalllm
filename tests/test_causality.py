from pathlib import Path

import torch

from smalllm.model import ModelConfig, Transformer

CONFIGS = Path(__file__).parent.parent / "configs"


def test_future_token_does_not_change_earlier_logits() -> None:
    torch.manual_seed(0)
    model = Transformer(ModelConfig.from_yaml(CONFIGS / "nano.yaml")).eval()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 16))
    changed = ids.clone()
    changed[0, 10] = (ids[0, 10] + 1) % model.cfg.vocab_size
    with torch.no_grad():
        a, b = model(ids), model(changed)
    torch.testing.assert_close(a[:, :10], b[:, :10])
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_prefix_logits_match_full_sequence() -> None:
    torch.manual_seed(1)
    model = Transformer(ModelConfig.from_yaml(CONFIGS / "nano.yaml")).eval()
    ids = torch.randint(0, model.cfg.vocab_size, (2, 12))
    with torch.no_grad():
        full, prefix = model(ids), model(ids[:, :7])
    torch.testing.assert_close(full[:, :7], prefix, atol=1e-5, rtol=1e-4)
