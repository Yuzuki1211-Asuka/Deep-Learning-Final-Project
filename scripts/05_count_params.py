import os
import sys
import csv
import json
import argparse
import inspect
from collections import defaultdict

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


def count_params(parameters, trainable_only=False):
    total = 0
    for p in parameters:
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def classify_parameter(name):
    """
    根据原仓库 Transformer 的参数名做粗粒度分类。
    用于课程报告中的模块参数量分析。
    """
    lower = name.lower()

    if "src_word_emb" in lower:
        return "Source Embedding"
    if "trg_word_emb" in lower:
        return "Target Embedding"
    if "position_enc" in lower or "position" in lower:
        return "Positional Encoding"
    if "trg_word_prj" in lower:
        return "Output Projection"

    if "slf_attn" in lower or "enc_attn" in lower:
        return "Multi-Head Attention"

    if "pos_ffn" in lower:
        return "Feed Forward Network"

    if "encoder" in lower:
        return "Other Encoder Parameters"
    if "decoder" in lower:
        return "Other Decoder Parameters"

    return "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-prefix", default="baseline_256_3l")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    config = checkpoint["config"]
    data_info = checkpoint["data_info"]

    model = build_transformer(config, data_info)
    model.load_state_dict(checkpoint["model_state_dict"])

    total_params = count_params(model.parameters(), trainable_only=False)
    trainable_params = count_params(model.parameters(), trainable_only=True)

    print("=" * 80)
    print("Model Parameter Statistics")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Source vocab size: {data_info['src_vocab_size']}")
    print(f"Target vocab size: {data_info['trg_vocab_size']}")
    print(f"d_model: {config['d_model']}")
    print(f"n_layers: {config['n_layers']}")
    print(f"n_head: {config['n_head']}")
    print(f"d_inner: {config['d_inner']}")
    print("-" * 80)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 80)

    module_stats = defaultdict(lambda: {"total": 0, "trainable": 0})

    for name, param in model.named_parameters():
        category = classify_parameter(name)
        module_stats[category]["total"] += param.numel()
        if param.requires_grad:
            module_stats[category]["trainable"] += param.numel()

    preferred_order = [
        "Source Embedding",
        "Target Embedding",
        "Positional Encoding",
        "Multi-Head Attention",
        "Feed Forward Network",
        "Other Encoder Parameters",
        "Other Decoder Parameters",
        "Output Projection",
        "Other",
    ]

    rows = []

    print("Parameters by module:")
    print("-" * 80)
    print(f"{'Module':35s} {'Total':>15s} {'Trainable':>15s}")
    print("-" * 80)

    for module_name in preferred_order:
        if module_name not in module_stats:
            continue

        total = module_stats[module_name]["total"]
        trainable = module_stats[module_name]["trainable"]

        rows.append({
            "module": module_name,
            "total_params": total,
            "trainable_params": trainable,
        })

        print(f"{module_name:35s} {total:15,} {trainable:15,}")

    print("-" * 80)

    # 额外统计 Encoder / Decoder 整体参数
    extra_rows = []

    if hasattr(model, "encoder"):
        enc_total = count_params(model.encoder.parameters(), trainable_only=False)
        enc_trainable = count_params(model.encoder.parameters(), trainable_only=True)
        extra_rows.append({
            "module": "Encoder Total",
            "total_params": enc_total,
            "trainable_params": enc_trainable,
        })

    if hasattr(model, "decoder"):
        dec_total = count_params(model.decoder.parameters(), trainable_only=False)
        dec_trainable = count_params(model.decoder.parameters(), trainable_only=True)
        extra_rows.append({
            "module": "Decoder Total",
            "total_params": dec_total,
            "trainable_params": dec_trainable,
        })

    if hasattr(model, "trg_word_prj"):
        out_total = count_params(model.trg_word_prj.parameters(), trainable_only=False)
        out_trainable = count_params(model.trg_word_prj.parameters(), trainable_only=True)
        extra_rows.append({
            "module": "Output Projection Total",
            "total_params": out_total,
            "trainable_params": out_trainable,
        })

    print("\nMajor component totals:")
    print("-" * 80)
    print(f"{'Module':35s} {'Total':>15s} {'Trainable':>15s}")
    print("-" * 80)

    for row in extra_rows:
        print(f"{row['module']:35s} {row['total_params']:15,} {row['trainable_params']:15,}")

    all_rows = rows + extra_rows + [
        {
            "module": "Model Total",
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
    ]

    csv_path = f"results/{args.save_prefix}_param_stats.csv"
    json_path = f"results/{args.save_prefix}_param_stats.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "total_params", "trainable_params"])
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "checkpoint": args.checkpoint,
        "config": config,
        "src_vocab_size": data_info["src_vocab_size"],
        "trg_vocab_size": data_info["trg_vocab_size"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "module_stats": all_rows,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"CSV saved to:  {csv_path}")
    print(f"JSON saved to: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
