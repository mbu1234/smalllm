"""Print N token ids and their tokens, starting at a given id.

Run with:  uv run python scripts/show_vocab.py [N] [START]      (defaults: N = 200, START = 0)
"""

import sys

from tokenize_demo import load_tokenizer


def main() -> None:
    # Tokens contain characters like 'Ġ' that the default Windows console encoding can't print.
    sys.stdout.reconfigure(encoding="utf-8")
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    tok = load_tokenizer()
    # with_added_tokens=True includes special tokens such as <|endoftext|> (id 0).
    vocab = tok.get_vocab(with_added_tokens=True)  # dict: token string -> id
    for token, token_id in sorted(vocab.items(), key=lambda item: item[1])[start : start + count]:
        print(f"{token_id:>6}  {token!r}")


if __name__ == "__main__":
    main()
