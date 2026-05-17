import os
import sys
import csv
import pickle
import argparse
import inspect

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from transformer.Models import Transformer


def build_transformer(config, data_info):
    init_params = set(inspect.signature(Transformer.__init__).parameters.keys())

    kwargs = {
        "n_src_vocab": data_info["src_vocab_size"],
        "n_trg_vocab": data_info["trg_vocab_size"],
        "src_pad_idx": data_info["pad_idx"],
        "trg_pad_idx": data_info["pad_idx"],
        "d_word_vec": config["d_word_vec"],
        "d_model": config["d_model"],
        "d_inner": config["d_inner"],
        "n_layers": config["n_layers"],
        "n_head": config["n_head"],
        "d_k": config["d_k"],
        "d_v": config["d_v"],
        "dropout": config["dropout"],
        "n_position": config["n_position"],
        "trg_emb_prj_weight_sharing": False,
        "emb_src_trg_weight_sharing": False,
        "scale_emb_or_prj": "none",
    }

    filtered_kwargs = {
        key: value for key, value in kwargs.items()
        if key in init_params
    }

    return Transformer(**filtered_kwargs)


def ids_to_tokens(ids, itos, special_ids):
    tokens = []
    for idx in ids:
        idx = int(idx)
        if idx in special_ids:
            continue
        if 0 <= idx < len(itos):
            tokens.append(itos[idx])
        else:
            tokens.append("<unk>")
    return tokens


def greedy_decode(model, src_ids, bos_idx, eos_idx, max_len, device):
    model.eval()

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    ys = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(src, ys)

        # 原仓库输出形状是 [batch_size * trg_len, trg_vocab_size]
        vocab_size = logits.size(-1)
        logits = logits.view(1, ys.size(1), vocab_size)

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()

        ys = torch.cat(
            [
                ys,
                torch.tensor([[next_token]], dtype=torch.long, device=device)
            ],
            dim=1,
        )

        if next_token == eos_idx:
            break

    return ys.squeeze(0).tolist()


def encode_sentence(sentence, src_vocab, eos_idx, unk_idx):
    tokens = sentence.strip().lower().split()
    ids = [src_vocab.get(tok, unk_idx) for tok in tokens]
    ids.append(eos_idx)
    return ids, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="data/processed/multi30k_en_de.pkl")
    parser.add_argument("--save-path", default="results/baseline_256_3l_predictions.csv")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--sentence", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Greedy Translation")
    print("=" * 80)
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint["config"]
    data_info = checkpoint["data_info"]

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    src_vocab = data["src_vocab"]
    trg_vocab = data["trg_vocab"]
    src_itos = data["src_itos"]
    trg_itos = data["trg_itos"]

    pad_idx = data["pad_idx"]
    unk_idx = data["unk_idx"]
    bos_idx = data["bos_idx"]
    eos_idx = data["eos_idx"]

    special_ids = {pad_idx, unk_idx, bos_idx, eos_idx}

    model = build_transformer(config, data_info).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    rows = []

    if args.sentence is not None:
        src_ids, src_tokens = encode_sentence(args.sentence, src_vocab, eos_idx, unk_idx)

        pred_ids = greedy_decode(
            model=model,
            src_ids=src_ids,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            max_len=args.max_len,
            device=device,
        )

        pred_tokens = ids_to_tokens(pred_ids, trg_itos, special_ids)

        print("Source sentence:", args.sentence)
        print("Source tokens:  ", " ".join(src_tokens))
        print("Prediction:     ", " ".join(pred_tokens))
        return

    test_data = data["test"]
    n = min(args.num_samples, len(test_data))

    print(f"Generating predictions for {n} test samples...")

    for i in range(n):
        src_ids, trg_ids, src_tokens, trg_tokens = test_data[i]

        pred_ids = greedy_decode(
            model=model,
            src_ids=src_ids,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            max_len=args.max_len,
            device=device,
        )

        pred_tokens = ids_to_tokens(pred_ids, trg_itos, special_ids)

        src_sentence = " ".join(src_tokens)
        gold_sentence = " ".join(trg_tokens)
        pred_sentence = " ".join(pred_tokens)

        rows.append({
            "id": i + 1,
            "source_en": src_sentence,
            "target_de": gold_sentence,
            "prediction_de": pred_sentence,
        })

        print("-" * 80)
        print(f"Sample {i + 1}")
        print("EN:   ", src_sentence)
        print("GOLD: ", gold_sentence)
        print("PRED: ", pred_sentence)

    with open(args.save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "source_en", "target_de", "prediction_de"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print(f"Predictions saved to: {args.save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
