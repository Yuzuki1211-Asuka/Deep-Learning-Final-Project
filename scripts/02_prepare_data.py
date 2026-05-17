import os
import pickle
import argparse
import urllib.request
from collections import Counter


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok"

FILES = {
    "train.en": "train.lc.norm.tok.en",
    "train.de": "train.lc.norm.tok.de",
    "val.en": "val.lc.norm.tok.en",
    "val.de": "val.lc.norm.tok.de",
    "test.en": "test_2016_flickr.lc.norm.tok.en",
    "test.de": "test_2016_flickr.lc.norm.tok.de",
}


def download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"[Skip] {path} already exists.")
        return

    print(f"[Download] {url}")
    urllib.request.urlretrieve(url, path)
    print(f"[Saved] {path}")


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def tokenize(sentence):
    return sentence.strip().split()


def read_parallel(src_path, trg_path, max_len):
    src_lines = read_lines(src_path)
    trg_lines = read_lines(trg_path)

    assert len(src_lines) == len(trg_lines), (
        f"Line number mismatch: {src_path} has {len(src_lines)}, "
        f"{trg_path} has {len(trg_lines)}"
    )

    pairs = []

    for src, trg in zip(src_lines, trg_lines):
        src_tokens = tokenize(src)
        trg_tokens = tokenize(trg)

        if len(src_tokens) == 0 or len(trg_tokens) == 0:
            continue

        if len(src_tokens) > max_len or len(trg_tokens) > max_len:
            continue

        pairs.append((src_tokens, trg_tokens))

    return pairs


def build_vocab(token_sequences, min_freq, max_vocab):
    counter = Counter()

    for tokens in token_sequences:
        counter.update(tokens)

    stoi = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    itos = list(SPECIAL_TOKENS)

    for token, freq in counter.most_common():
        if freq < min_freq:
            continue

        if token in stoi:
            continue

        if len(itos) >= max_vocab:
            break

        stoi[token] = len(itos)
        itos.append(token)

    return stoi, itos, counter


def encode_tokens(tokens, vocab, add_bos=False, add_eos=True):
    ids = []

    if add_bos:
        ids.append(BOS_IDX)

    for token in tokens:
        ids.append(vocab.get(token, UNK_IDX))

    if add_eos:
        ids.append(EOS_IDX)

    return ids


def encode_pairs(pairs, src_vocab, trg_vocab):
    encoded = []

    for src_tokens, trg_tokens in pairs:
        src_ids = encode_tokens(src_tokens, src_vocab, add_bos=False, add_eos=True)
        trg_ids = encode_tokens(trg_tokens, trg_vocab, add_bos=True, add_eos=True)
        encoded.append((src_ids, trg_ids, src_tokens, trg_tokens))

    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/multi30k")
    parser.add_argument("--save-path", default="data/processed/multi30k_en_de.pkl")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--max-len", type=int, default=80)
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print("=" * 80)
    print("Step 1: Download Multi30k tokenized files")
    print("=" * 80)

    local_paths = {}

    for key, filename in FILES.items():
        url = f"{BASE_URL}/{filename}"
        path = os.path.join(args.raw_dir, filename)
        download_file(url, path)
        local_paths[key] = path

    print("=" * 80)
    print("Step 2: Read parallel corpus")
    print("=" * 80)

    train_pairs = read_parallel(local_paths["train.en"], local_paths["train.de"], args.max_len)
    val_pairs = read_parallel(local_paths["val.en"], local_paths["val.de"], args.max_len)
    test_pairs = read_parallel(local_paths["test.en"], local_paths["test.de"], args.max_len)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")

    print("=" * 80)
    print("Step 3: Build vocabularies from training set")
    print("=" * 80)

    src_token_sequences = [src for src, _ in train_pairs]
    trg_token_sequences = [trg for _, trg in train_pairs]

    src_vocab, src_itos, src_counter = build_vocab(
        src_token_sequences,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
    )

    trg_vocab, trg_itos, trg_counter = build_vocab(
        trg_token_sequences,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
    )

    print(f"Source vocab size: {len(src_itos)}")
    print(f"Target vocab size: {len(trg_itos)}")

    print("Most common English tokens:", src_counter.most_common(10))
    print("Most common German tokens: ", trg_counter.most_common(10))

    print("=" * 80)
    print("Step 4: Encode text into token ids")
    print("=" * 80)

    train_data = encode_pairs(train_pairs, src_vocab, trg_vocab)
    val_data = encode_pairs(val_pairs, src_vocab, trg_vocab)
    test_data = encode_pairs(test_pairs, src_vocab, trg_vocab)

    data = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab,
        "src_itos": src_itos,
        "trg_itos": trg_itos,
        "pad_idx": PAD_IDX,
        "unk_idx": UNK_IDX,
        "bos_idx": BOS_IDX,
        "eos_idx": EOS_IDX,
        "special_tokens": SPECIAL_TOKENS,
        "config": {
            "min_freq": args.min_freq,
            "max_vocab": args.max_vocab,
            "max_len": args.max_len,
        },
    }

    with open(args.save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[Saved] {args.save_path}")

    print("=" * 80)
    print("Example")
    print("=" * 80)

    src_ids, trg_ids, src_tokens, trg_tokens = train_data[0]
    print("English tokens:", src_tokens)
    print("German tokens: ", trg_tokens)
    print("English ids:   ", src_ids)
    print("German ids:    ", trg_ids)

    print("=" * 80)
    print("Data preparation finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
