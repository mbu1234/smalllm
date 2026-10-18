# CLAUDE.md — Small Llama-style LLM (CPU-first, optional GPU)

## Project goal

Build a small, decoder-only, **base** language model (no chat template, no reasoning/thinking
tokens, no RLHF) in plain PyTorch that can be:

1. **Pretrained from scratch** on Project Gutenberg text, on a CPU laptop, with weights saved by me.
2. **Loaded with published weights** (SmolLM2-135M, base model) to get good-quality output.

Both must use the **same model code**. Only the config differs.

This is a learning project. Prefer clear, readable code over cleverness. Target ~150-250 lines for
the model file. Do not wrap or subclass Hugging Face models for the core implementation.

## Hard requirements

- All Python dependencies are installed and run with **uv**. Never use `pip`, `pip install`,
  `python -m venv`, `poetry`, or `conda`. Use `uv add`, `uv add --dev`, `uv sync`, `uv run`.
- Must run on **CPU by default**. Must automatically use a GPU if one is available.
  Device priority: `cuda` > `mps` > `cpu`. Allow override with `--device`.
- **Rotary positional embeddings (RoPE)**. No learned or sinusoidal absolute position embeddings.
- A **small number of standard transformer blocks** (pre-norm, residual, attention + MLP).
- Architecture must match SmolLM2 / Llama so published weights load with no conversion hacks.

## Architecture spec (Llama-style)

- Token embedding, **tied** with the output head (`lm_head.weight` is `embed_tokens.weight`)
- N blocks, each: `RMSNorm -> causal self-attention -> residual -> RMSNorm -> SwiGLU MLP -> residual`
- Final `RMSNorm`, then the output head
- Attention: **grouped-query attention** (n_kv_heads <= n_heads), no biases on projections,
  use `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`
- RoPE: applied to q and k only, using the **Hugging Face Llama convention**
  (`rotate_half`, i.e. the first half / second half of head_dim are paired, NOT interleaved pairs).
  This is essential for published weights to produce correct output.
- MLP: SwiGLU = `down_proj(silu(gate_proj(x)) * up_proj(x))`, no biases
- RMSNorm: `x * rsqrt(mean(x^2) + eps) * weight`, compute in float32

### Name parameters exactly like Hugging Face Llama

So the checkpoint can be loaded with a direct `load_state_dict`:

```
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.{gate,up,down}_proj.weight
model.norm.weight
lm_head.weight            # tied to model.embed_tokens.weight
```

Structure the modules so these names fall out naturally (e.g. a `model` submodule holding
`embed_tokens`, `layers`, `norm`). Skip any buffers like `rotary_emb.inv_freq`; compute RoPE tables
inside our own code.

## Two configs, one codebase

Put configs in `configs/` as YAML or a dataclass. **Treat the published `config.json` as the source
of truth** for the pretrained config; download it and read it rather than trusting the numbers here.

| | `smollm2_135m` (published) | `tiny` (train from scratch) |
|---|---|---|
| vocab_size | 49152 | 49152 (same tokenizer) |
| hidden_size | 576 | 256 |
| num_layers | 30 | 6 |
| num_heads / kv_heads | 9 / 3 | 4 / 2 |
| intermediate_size | 1536 | 704 |
| head_dim | 64 | 64 |
| rope_theta | check config.json (likely 100000) | same as published |
| tie_word_embeddings | true | true |
| context length | 2048 usable | 256 (raise later if CPU allows) |

Also provide a `nano` config (e.g. 4 layers, hidden 128) for quick smoke tests that finish in seconds.

Note: with a 49k vocabulary, the embedding/output matrices dominate the parameter count and compute
of the tiny model. That is accepted for now so that the tokenizer is shared with the published
model. Mention this tradeoff in the README; a custom small BPE tokenizer is a possible later
experiment but would make the scratch model incompatible with the published weights.

## Project layout

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── nano.yaml
│   ├── tiny.yaml
│   └── smollm2_135m.yaml
├── data/                   # gitignored
│   ├── raw/                # downloaded Gutenberg .txt files
│   └── processed/          # train.bin / val.bin (uint16 token ids)
├── checkpoints/            # gitignored, safetensors + config snapshot
├── src/smalllm/
│   ├── model.py            # RMSNorm, RoPE, GQA attention, SwiGLU, Transformer
│   ├── device.py           # pick_device() and dtype/autocast helpers
│   ├── data.py             # Gutenberg download/clean, tokenize, batch sampler
│   ├── train.py            # training loop
│   ├── generate.py         # sampling (temperature, top-k, top-p)
│   └── load_pretrained.py  # HF weights -> our model
└── tests/
    ├── test_model_shapes.py
    ├── test_rope.py
    ├── test_causality.py
    └── test_hf_parity.py
```

## Setup (uv)

First, **ask me my OS and whether this laptop has an NVIDIA GPU or Apple Silicon**, and check with
`nvidia-smi` (NVIDIA) or `uname -m` / `system_profiler SPDisplaysDataType` (Mac). This decides the
PyTorch install.

```bash
uv init --package smalllm        # or `uv init` if already inside the project dir
uv add torch numpy safetensors tokenizers huggingface_hub pyyaml tqdm
uv add --dev pytest transformers ruff
uv sync
```

PyTorch notes:

- **macOS**: the default PyPI wheel supports MPS. Nothing special needed.
- **Linux**: the default PyPI wheel bundles CUDA. For a CPU-only install or a specific CUDA version,
  configure a PyTorch index in `pyproject.toml`.
- **Windows**: the default PyPI wheel is **CPU-only**. For NVIDIA GPU support add the matching PyTorch
  CUDA index via `[[tool.uv.index]]` and `[tool.uv.sources]` in `pyproject.toml`.
- Consult the current uv docs page "Using uv with PyTorch" for the exact syntax before editing
  `pyproject.toml`. Do not guess index URLs.

Verify the install:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
```

`transformers` is a **dev dependency only**, used for the tokenizer in tests and for the parity test.
The model itself must not import it.

## Build order (do these in order, one milestone at a time, running tests after each)

1. **Model skeleton** (`model.py`, `device.py`). Random init, forward pass, shape tests with the `nano`
   config. Init: normal(0, 0.02), scale residual output projections (`o_proj`, `down_proj`) by
   `1/sqrt(2*n_layers)`.
2. **Tests**: RoPE correctness (relative-position property: shifting both q and k positions by the same
   offset leaves attention scores unchanged), causality (changing a future token does not change earlier
   logits), parameter count.
3. **Load published weights** (`load_pretrained.py`): download `HuggingFaceTB/SmolLM2-135M` (the **base**
   model, not `-Instruct`) with `huggingface_hub`, load `model.safetensors` into our model, handling tied
   weights.
4. **HF parity test** (`test_hf_parity.py`): load the same checkpoint into `transformers`
   `AutoModelForCausalLM` and compare logits on a fixed prompt. Require max abs diff < ~1e-3 in float32.
   **Do not proceed until this passes.** It catches RoPE convention and naming mistakes.
5. **Generation** (`generate.py`): autoregressive sampling with temperature / top-k / top-p. Start without
   a KV cache; add a simple KV cache afterwards only if generation is painfully slow. Verify the pretrained
   model produces sensible English continuation.
6. **Data** (`data.py`): see below. Overfit test first.
7. **Training loop** (`train.py`): see below. Run on `nano`, then `tiny`.
8. **Save/load own checkpoints**, resume training, and generate from a scratch-trained checkpoint using the
   same `generate.py`.

## Data: Project Gutenberg

- Download plain-text books by ID from gutenberg.org, for example
  `https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt`. Keep the list of IDs in a config/text file.
  Download politely: a handful of books, sleep between requests, cache in `data/raw/`, never re-download.
- Clean: strip the Project Gutenberg header and footer (text outside the
  `*** START OF THE PROJECT GUTENBERG EBOOK ... ***` / `*** END OF ... ***` markers), normalise line
  endings, collapse the hard-wrapped single newlines within paragraphs into spaces but keep blank lines
  as paragraph breaks.
- Tokenize with the **SmolLM2 tokenizer** (`tokenizers` library, load `tokenizer.json` from the hub).
  Append an EOS token between books. Store as `uint16` numpy arrays in `data/processed/`.
- Split by **contiguous chunk** (last ~5-10% for validation), not random lines, to avoid leakage.
- Batches: random windows of `block_size + 1` tokens from the train array.

Dataset size guidance (print token counts after processing):

- **One book** (~100-300k tokens): use it **only** as a pipeline sanity check. The model should be able to
  drive training loss near zero (memorise it). It will not generalise.
- **~20-100 books** from one era/author style (~5-30M tokens): the realistic target. Expect train loss
  to fall well below the starting loss (about ln(49152) ≈ 10.8) and generations that mimic the style of
  the prose, with locally plausible grammar but little coherence.

## Training loop requirements

- AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only, none on norms/embeddings), peak LR ~1e-3
  for `tiny` / `nano`, linear warmup then cosine decay to 10% of peak, grad clip 1.0.
- Device handling: `cuda` -> bf16 autocast if supported else fp16 with GradScaler; `mps` -> fp32 to start;
  `cpu` -> fp32. No `torch.compile` by default (flag `--compile`, off on CPU/MPS).
- Set `torch.set_num_threads()` sensibly on CPU (default to physical core count, flag to override).
- Config flags: batch size, gradient accumulation steps, block size, max steps, eval interval, eval
  batches, seed.
- Log train loss, val loss, tokens/sec, and elapsed time. Print perplexity for val.
- Checkpoint every N steps and at the end: `model.safetensors` + `config.yaml` + `optimizer.pt` + `state.json`
  (step, RNG seeds). Support `--resume`.
- First run `nano` for ~200 steps to confirm loss falls and tokens/sec is reasonable on this machine,
  and **report the measured speed and an estimate of total training time for `tiny`** before launching a
  long run.

## Coding conventions

- Python 3.11+, type hints on public functions, `ruff` for lint/format (`uv run ruff check`, `uv run ruff format`).
- Run everything with `uv run ...`, e.g. `uv run pytest`, `uv run python -m smalllm.train --config configs/tiny.yaml`.
- Keep functions small. No global state. Seed everything for reproducibility.
- Comment the *why* for anything non-obvious (especially RoPE layout and weight tying).
- Add `data/`, `checkpoints/`, `.venv/` to `.gitignore`. Commit `uv.lock`.
- Do not download anything large (weights, datasets) without telling me the size first.

## Things NOT to do

- Do not add chat templates, instruction tuning, system prompts, tool use, or "thinking" behaviour.
- Do not pull in training frameworks (Lightning, Trainer, Accelerate, DeepSpeed) or use
  `transformers` models inside the core code.
- Do not use learned absolute position embeddings or ALiBi.
- Do not change the architecture away from the published model (e.g. add biases, change the MLP, or switch
  to interleaved RoPE) without asking, since that breaks weight loading.
- Do not silently fall back from GPU to CPU or the reverse; always print the chosen device and dtype at start-up.

## Getting started (what I want you to do now)

1. Ask me my OS and GPU details, then run the setup steps above.
2. Scaffold the layout and write `pyproject.toml`, configs, and `.gitignore`.
3. Implement milestone 1 and 2, run the tests, and show me the output before moving on.
4. Continue through the build order, pausing after milestone 4 (HF parity) and after the first
   `nano` training run to report results.

## Useful commands (fill in as the project grows)

```bash
uv sync
uv run pytest -q
uv run python -m smalllm.load_pretrained --model HuggingFaceTB/SmolLM2-135M
uv run python -m smalllm.generate --pretrained HuggingFaceTB/SmolLM2-135M --prompt "Once upon a time"
uv run python -m smalllm.data --books configs/books.txt
uv run python -m smalllm.train --config configs/nano.yaml
uv run python -m smalllm.train --config configs/tiny.yaml --device auto
uv run python -m smalllm.generate --checkpoint checkpoints/tiny --prompt "It was a dark"
```