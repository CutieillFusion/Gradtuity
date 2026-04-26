"""
GPT-2 dataset prep: download codelion 50/30/20 mix (finePDFs / DCLM / FineWeb-Edu),
select up to a token budget, concatenate, shuffle, and export tokens.bin + meta.json.

Run as a script: `uv run python -m gpt2.dataset` (or `python gpt2/dataset.py`).
Importable: `from gpt2.dataset import load_and_mix, export_tokens` — no side effects.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Iterable

import numpy as np
import tiktoken
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import get_token
from tqdm import tqdm

EOS_TOKEN_ID = 50256
BATCH_SIZE_TOKENIZE = 4096  # for selection when no token_count column
BATCH_SIZE_EXPORT = 2048


def _make_encode_batch(encoding: tiktoken.Encoding) -> Callable[[Iterable[str]], list[list[int]]]:
    """Return a function that tokenizes a list of strings (handles None and missing encode_batch)."""
    encode_kw = {"disallowed_special": ()}
    encode_batch_fn = getattr(encoding, "encode_batch", None)

    def _encode(texts: Iterable[str]) -> list[list[int]]:
        clean = [t or "" for t in texts]
        if encode_batch_fn is not None:
            return encode_batch_fn(clean, **encode_kw)
        return [encoding.encode(t, **encode_kw) for t in clean]

    return _encode


def select_until_tokens_fast(
    dataset: Dataset,
    target_tokens: int,
    encoding: tiktoken.Encoding,
    *,
    desc: str = "Selecting",
    use_token_count_column: bool = False,
    text_col: str = "text",
    batch_size: int = BATCH_SIZE_TOKENIZE,
) -> tuple[Dataset, int]:
    """
    Fast selection: vectorized cumsum/searchsorted when token_count exists;
    otherwise batch slices + encoding.encode_batch(texts).
    """
    if use_token_count_column and "token_count" in dataset.column_names:
        counts = np.array([c or 0 for c in dataset["token_count"]], dtype=np.int64)
        csum = np.cumsum(counts)
        idx = np.searchsorted(csum, target_tokens, side="left")
        n = min(int(idx) + 1, len(dataset))
        cumul = int(csum[n - 1])
        return dataset.select(range(n)), cumul

    encode = _make_encode_batch(encoding)
    cumul = 0
    n = 0
    pbar = tqdm(total=len(dataset), desc=desc, unit=" rows")

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, end))
        texts = list(batch[text_col])

        token_ids = encode(texts)
        lens = np.array([len(x) for x in token_ids], dtype=np.int64)
        batch_sum = int(lens.sum())

        if cumul + batch_sum >= target_tokens:
            prefix = np.cumsum(lens)
            k = int(np.searchsorted(prefix, target_tokens - cumul, side="left")) + 1
            k = min(k, len(lens))
            n = start + k
            cumul += int(prefix[k - 1])
            pbar.update(k)
            pbar.set_postfix(tokens=f"{cumul:,}", target=f"{target_tokens:,}")
            break

        cumul += batch_sum
        n = end
        pbar.update(len(texts))
        pbar.set_postfix(tokens=f"{cumul:,}", target=f"{target_tokens:,}")

    pbar.close()
    return dataset.select(range(n)), cumul


def load_and_mix(
    total_tokens: int = 1_000_000_000,
    *,
    mix: tuple[float, float, float] = (0.5, 0.3, 0.2),
    seed: int = 42,
    encoding: tiktoken.Encoding | None = None,
) -> tuple[Dataset, dict[str, int]]:
    """
    Download codelion finePDFs / DCLM-baseline / FineWeb-Edu, select up to mix * total_tokens
    from each, concatenate and shuffle. Returns (dataset, token_counts_per_source).
    """
    assert get_token(), "No HF token set"
    encoding = encoding or tiktoken.get_encoding("gpt2")

    target_fp = int(total_tokens * mix[0])
    target_dclm = int(total_tokens * mix[1])
    target_fw = int(total_tokens * mix[2])

    finepdfs = load_dataset("codelion/finepdfs-1B", split="train").shuffle(seed=seed)
    dclm = load_dataset("codelion/dclm-baseline-1B", split="train").shuffle(seed=seed)
    fineweb_edu = load_dataset("codelion/fineweb-edu-1B", split="train").shuffle(seed=seed)

    train_fp, tokens_fp = select_until_tokens_fast(
        finepdfs, target_fp, encoding, desc="finePDFs", use_token_count_column=True
    )
    train_dclm, tokens_dclm = select_until_tokens_fast(
        dclm, target_dclm, encoding, desc="DCLM"
    )
    train_fw, tokens_fw = select_until_tokens_fast(
        fineweb_edu, target_fw, encoding, desc="FineWeb-Edu"
    )

    combined = concatenate_datasets([train_fp, train_dclm, train_fw]).shuffle(seed=seed)
    counts = {
        "finepdfs": tokens_fp,
        "dclm": tokens_dclm,
        "fineweb_edu": tokens_fw,
        "total": tokens_fp + tokens_dclm + tokens_fw,
    }
    return combined, counts


def export_tokens(
    dataset: Dataset,
    output_dir: str,
    encoding: tiktoken.Encoding | None = None,
    *,
    text_col: str = "text",
    batch_size: int = BATCH_SIZE_EXPORT,
) -> tuple[str, str]:
    """
    Tokenize `dataset[text_col]` with `encoding` and write a flat uint16 token stream
    to `output_dir/tokens.bin`, with EOS between documents and `meta.json` alongside.
    Returns (tokens_path, meta_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    tokens_path = os.path.join(output_dir, "tokens.bin")
    meta_path = os.path.join(output_dir, "meta.json")

    encoding = encoding or tiktoken.get_encoding("gpt2")
    encode = _make_encode_batch(encoding)
    n_docs = len(dataset)
    num_tokens = 0

    with open(tokens_path, "wb") as f:
        for start in tqdm(range(0, n_docs, batch_size), desc="Export tokens", unit=" batch"):
            end = min(start + batch_size, n_docs)
            batch = dataset.select(range(start, end))
            token_ids = encode(list(batch[text_col]))

            # Flatten with EOS between documents (skip after the very last document)
            chunks: list[np.ndarray] = []
            for i, ids in enumerate(token_ids):
                arr = np.asarray(ids, dtype=np.uint32)
                np.minimum(arr, 65535, out=arr)
                chunks.append(arr.astype(np.uint16))
                if start + i < n_docs - 1:
                    chunks.append(np.array([EOS_TOKEN_ID], dtype=np.uint16))
            if chunks:
                flat = np.concatenate(chunks)
                flat.tofile(f)
                num_tokens += int(flat.size)

    meta = {
        "num_tokens": num_tokens,
        "vocab_size": encoding.n_vocab,
        "dtype": "uint16",
        "eos_token_id": EOS_TOKEN_ID,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))
    return tokens_path, meta_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the 50/30/20 GPT-2 training corpus")
    ap.add_argument("--total_tokens", type=int, default=1_000_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <repo>/gpt2/data/50-30-20-mix",
    )
    args = ap.parse_args()

    # Avoid SSL errors when SSL_CERT_FILE points to a missing file (e.g. on HPC)
    os.environ.pop("SSL_CERT_FILE", None)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gpt2", "data", "50-30-20-mix",
    )

    encoding = tiktoken.get_encoding("gpt2")
    dataset, counts = load_and_mix(args.total_tokens, seed=args.seed, encoding=encoding)
    print(dataset)
    print(
        f"finePDFs={counts['finepdfs']:,}  DCLM={counts['dclm']:,}  "
        f"FineWeb-Edu={counts['fineweb_edu']:,}  total={counts['total']:,}"
    )

    print(f"Saving HF dataset to {output_dir} (Arrow shards)...")
    dataset.save_to_disk(output_dir)

    print("Exporting flat uint16 token stream + meta.json...")
    tokens_path, meta_path = export_tokens(dataset, output_dir, encoding)
    print(f"Wrote {tokens_path} and {meta_path}")


if __name__ == "__main__":
    main()
