# smalllm

A small Llama-style, decoder-only **base** language model in plain PyTorch, built as a learning
project. The same model code is used for training from scratch and (later) for loading published
SmolLM2-135M weights. Only the config differs. It runs on CPU by default and uses a GPU
automatically if one is available (`cuda` > `mps` > `cpu`).

## Setup

All dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
```

On Windows the default PyTorch wheel is CPU-only, which is what this project uses unless you add a
CUDA index to `pyproject.toml`.

## Running inference

`smalllm.generate` tokenizes a prompt, repeatedly samples the next token from the model, and prints
the decoded text.

```bash
uv run python -m smalllm.generate --prompt "Once upon a time" --max-new-tokens 50
```

**Weights are currently untrained (random init), so the output is gibberish.** That is expected. It
shows the full pipeline working: tokenizer, forward pass, sampling and decoding. Trained weights
will plug into the same script once they are implemented.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/nano.yaml` | Model config: `nano`, `tiny` or `smollm2_135m` |
| `--prompt` | `Once upon a time` | Text to continue |
| `--max-new-tokens` | `50` | Number of tokens to generate |
| `--temperature` | `1.0` | Higher is more random. `0` is greedy (always the most likely token) |
| `--top-k` | `50` | Sample only from the k most likely tokens |
| `--top-p` | `0.95` | Sample only from the smallest set of tokens whose probability sums to p |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `mps`. The chosen device and dtype are printed at start-up |
| `--seed` | `0` | Fixes the random init and the sampling, so runs are reproducible |

The `smollm2_135m` config builds the full 135M-parameter model with random weights (about 540 MB in
float32), so it is slower and uses more memory than `nano`.

### First run needs the network

The first run downloads `tokenizer.json` (about 2 MB) from the `HuggingFaceTB/SmolLM2-135M` repo into
`~/.cache/huggingface`. No model weights are downloaded. Later runs use the cached file.

You may see a warning about unauthenticated requests to the HF Hub. It is harmless for public repos.
To silence it, set a free token from huggingface.co/settings/tokens in the `HF_TOKEN` environment
variable.

### Using it from Python

```python
import torch
from smalllm.generate import generate
from smalllm.model import ModelConfig, Transformer

model = Transformer(ModelConfig.from_yaml("configs/nano.yaml"))
ids = torch.tensor([[1, 2, 3]])  # token ids, shape (batch, seq)
out = generate(model, ids, max_new_tokens=20, temperature=0.8, top_k=50)
```

`generate` takes and returns token ids. Use the tokenizer in `smalllm.generate.main` to convert to
and from text.

## Configs

| Config | Layers | Hidden | Heads / KV heads | Context | Purpose |
|---|---|---|---|---|---|
| `nano` | 4 | 128 | 2 / 1 | 128 | Smoke tests |
| `tiny` | 6 | 256 | 4 / 2 | 256 | Train from scratch |
| `smollm2_135m` | 30 | 576 | 9 / 3 | 2048 | Published SmolLM2-135M architecture |

All three use the SmolLM2 tokenizer, so the vocabulary is 49,152. With a vocabulary that large, the
embedding/output matrix dominates the parameter count and compute of the small models. This is
accepted so the tokenizer, and therefore the model architecture, stays compatible with the published
weights. A custom small tokenizer would be a possible later experiment, but a scratch model trained
with it could not load the published weights.

## Status

- Done: model, device selection, tests (RoPE, causality, shapes, parameter counts), sampling.
- Not yet: loading published weights, Hugging Face parity test, data pipeline, training, checkpoints.
