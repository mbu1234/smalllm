"""Minimal tokenizer demo: text -> token ids -> text.

Run with:  uv run python scripts/tokenize_demo.py
"""

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "HuggingFaceTB/SmolLM2-135M"
SENTENCE = "Once upon a time"


def load_tokenizer() -> Tokenizer:
    # Downloads tokenizer.json on the first run, then reuses the cached copy
    # in ~/.cache/huggingface. The file holds the vocabulary and the BPE merge rules.
    path = hf_hub_download(REPO, "tokenizer.json")
    return Tokenizer.from_file(path)


def encode(tok: Tokenizer, text: str) -> list[int]:
    """Text -> list of token ids."""
    return tok.encode(text).ids


def decode(tok: Tokenizer, ids: list[int]) -> str:
    """List of token ids -> text."""
    return tok.decode(ids)


def main() -> None:
    tok = load_tokenizer()

    ids = encode(tok, SENTENCE)
    print(f"text:    {SENTENCE!r}")
    print(f"ids:     {ids}")

    # Show what each id stands for. 'Ġ' is how the tokenizer writes a leading space.
    for token_id in ids:
        print(f"  {token_id:>6} -> {tok.id_to_token(token_id)!r}")

    text = decode(tok, ids)
    print(f"decoded: {text!r}")
    assert text == SENTENCE, "round trip should reproduce the original text"


if __name__ == "__main__":
    main()
