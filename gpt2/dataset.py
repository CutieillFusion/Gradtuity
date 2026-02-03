import os

import numpy as np
import tiktoken
from huggingface_hub import get_token
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# Avoid SSL errors when SSL_CERT_FILE points to a missing file (e.g. on HPC)
os.environ.pop("SSL_CERT_FILE", None)

# On clusters, put HF caches on fast local disk (not NFS):
#   export HF_HOME=/scratch/$USER/hf
#   export HF_DATASETS_CACHE=$HF_HOME/datasets
#   export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

token = get_token()
assert token, "No token set"

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)

# GPT-2 tokenizer (tiktoken) for counting tokens on DCLM and FineWeb-Edu
print("Loading GPT-2 encoding (tiktoken)...")
gpt2_encoding = tiktoken.get_encoding("gpt2")
print("  encoding: gpt2")

# Load samples from our collection (codelion pre-training dataset collection)
finepdfs = load_dataset("codelion/finepdfs-1B", split="train")
dclm = load_dataset("codelion/dclm-baseline-1B", split="train")
fineweb_edu = load_dataset("codelion/fineweb-edu-1B", split="train")

# 50-30-20 mix by *token count* (article: 500M + 300M + 200M = 1B tokens)
# https://huggingface.co/blog/codelion/optimal-dataset-mixing
TOTAL_TOKENS_TARGET = 1_000_000_000 # 1B; use less for testing
target_fp = int(TOTAL_TOKENS_TARGET * 0.5)   # 500M finePDFs
target_dclm = int(TOTAL_TOKENS_TARGET * 0.3)  # 300M DCLM-baseline
target_fw = int(TOTAL_TOKENS_TARGET * 0.2)    # 200M FineWeb-Edu

seed = 42

# Set False to skip the expensive final shuffle (shuffle in training instead).
SHUFFLE_AFTER_CONCAT = False

BATCH_SIZE_TOKENIZE = 4096  # batch size for encode_batch when no token_count column


def select_until_tokens_fast(
    dataset,
    target_tokens,
    encoding,
    *,
    desc="Selecting",
    use_token_count_column=False,
    text_col="text",
    batch_size=BATCH_SIZE_TOKENIZE,
):
    """
    Fast selection: vectorized cumsum/searchsorted when token_count exists;
    otherwise batch slices + encoding.encode_batch(texts).
    """
    if use_token_count_column and "token_count" in dataset.column_names:
        # Vectorized: one pass over column, then numpy
        counts = np.array([c or 0 for c in dataset["token_count"]], dtype=np.int64)
        csum = np.cumsum(counts)
        idx = np.searchsorted(csum, target_tokens, side="left")
        n = min(int(idx) + 1, len(dataset))
        cumul = int(csum[n - 1])
        return dataset.select(range(n)), cumul

    # Batch slice + batch tokenize (no per-row dataset[i])
    # disallowed_special=() so literal "<|endoftext|>" etc. in data encode as normal text
    cumul = 0
    n = 0
    pbar = tqdm(total=len(dataset), desc=desc, unit=" rows")
    _encode_kw = {"disallowed_special": ()}
    encode_batch_fn = getattr(encoding, "encode_batch", None)

    def _encode_batch(texts):
        texts = [t or "" for t in texts]
        if encode_batch_fn is not None:
            return encode_batch_fn(texts, **_encode_kw)
        return [encoding.encode(t, **_encode_kw) for t in texts]

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, end))
        texts = batch[text_col]

        token_ids = _encode_batch(list(texts))
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


# Shuffle each source (reproducible), then take documents until token targets
train_fp = finepdfs.shuffle(seed=seed)
train_dclm = dclm.shuffle(seed=seed)
train_fw = fineweb_edu.shuffle(seed=seed)

# finePDFs has token_count → vectorized. DCLM and FineWeb-Edu use batch encode_batch.
train_fp, tokens_fp = select_until_tokens_fast(
    train_fp, target_fp, gpt2_encoding, desc="finePDFs", use_token_count_column=True
)
train_dclm, tokens_dclm = select_until_tokens_fast(
    train_dclm, target_dclm, gpt2_encoding, desc="DCLM", batch_size=BATCH_SIZE_TOKENIZE
)
train_fw, tokens_fw = select_until_tokens_fast(
    train_fw, target_fw, gpt2_encoding, desc="FineWeb-Edu", batch_size=BATCH_SIZE_TOKENIZE
)

# Static mixing: concatenate; optionally shuffle (expensive on large data)
dataset = concatenate_datasets([train_fp, train_dclm, train_fw])
if SHUFFLE_AFTER_CONCAT:
    dataset = dataset.shuffle(seed=seed)

print(dataset)
total_tokens = tokens_fp + tokens_dclm + tokens_fw
print(f"finePDFs tokens: {tokens_fp:,}  DCLM tokens: {tokens_dclm:,}  FineWeb-Edu tokens: {tokens_fw:,}")
print(f"Total token count: {total_tokens:,}")

# Save the final concatenated dataset to disk
OUTPUT_DIR = os.path.join(_root, "gpt2", "data", "50-30-20-mix")
print(f"Saving dataset to {OUTPUT_DIR}...")
dataset.save_to_disk(OUTPUT_DIR)
print("Done.")
