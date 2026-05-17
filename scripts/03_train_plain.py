import os
import sys
import time
import math
import csv
import pickle
import random
import argparse
import inspect
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from transformer.Models import Transformer


@dataclass
class TrainConfig:
    data_path: str = "data/processed/multi30k_en_de.pkl"
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"

    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4

    d_model: int = 256
    d_word_vec: int = 256
    d_inner: int = 512
    n_layers: int = 3
    n_head: int = 8
    d_k: int = 32
    d_v: int = 32
    dropout: float = 0.1
    n_position: int = 100

    max_grad_norm: float = 1.0
    num_workers: int = 2
    seed: int = 42
    use_amp: bool = True
    save_name: str = "baseline"


class TranslationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_ids, trg_ids, src_tokens, trg_tokens = self.samples[idx]
        return {
            "src_ids": src_ids,
            "trg_ids": trg_ids,
            "src_tokens": src_tokens,
            "trg_tokens": trg_tokens,
        }


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_sequences(sequences, pad_idx):
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)

    for i, seq in enumerate(sequences):
        batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    return batch


def make_collate_fn(pad_idx):
    def collate_fn(batch):
        src_ids = [item["src_ids"] for item in batch]
        trg_ids = [item["trg_ids"] for item in batch]

        src_batch = pad_sequences(src_ids, pad_idx)
        trg_batch = pad_sequences(trg_ids, pad_idx)

        return {
            "src": src_batch,
            "trg": trg_batch,
            "src_tokens": [item["src_tokens"] for item in batch],
            "trg_tokens": [item["trg_tokens"] for item in batch],
        }

    return collate_fn


def build_transformer(config, src_vocab_size, trg_vocab_size, pad_idx):
    init_params = set(inspect.signature(Transformer.__init__).parameters.keys())

    kwargs = {
        "n_src_vocab": src_vocab_size,
        "n_trg_vocab": trg_vocab_size,
        "src_pad_idx": pad_idx,
        "trg_pad_idx": pad_idx,
        "d_word_vec": config.d_word_vec,
        "d_model": config.d_model,
        "d_inner": config.d_inner,
        "n_layers": config.n_layers,
        "n_head": config.n_head,
        "d_k": config.d_k,
        "d_v": config.d_v,
        "dropout": config.dropout,
        "n_position": config.n_position,
        "trg_emb_prj_weight_sharing": False,
        "emb_src_trg_weight_sharing": False,
        "scale_emb_or_prj": "none",
    }

    filtered_kwargs = {
        key: value for key, value in kwargs.items()
        if key in init_params
    }

    return Transformer(**filtered_kwargs)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, config, pad_idx):
    model.train()

    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader, start=1):
        src = batch["src"].to(device)
        trg = batch["trg"].to(device)

        # trg: [B, L]
        # decoder input:  [<bos>, token1, token2, ...]
        # training target: [token1, token2, ..., <eos>]
        trg_input = trg[:, :-1]
        trg_gold = trg[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        use_amp_now = config.use_amp and device.type == "cuda"

        with torch.amp.autocast(device_type="cuda", enabled=use_amp_now):
            logits = model(src, trg_input)
            loss = criterion(logits, trg_gold.contiguous().view(-1))

        if use_amp_now:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        non_pad_tokens = trg_gold.ne(pad_idx).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

        if step % 100 == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - start_time
            print(
                f"  step {step:4d}/{len(dataloader)} | "
                f"loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                f"time {elapsed:.1f}s"
            )

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, pad_idx):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        src = batch["src"].to(device)
        trg = batch["trg"].to(device)

        trg_input = trg[:, :-1]
        trg_gold = trg[:, 1:]

        logits = model(src, trg_input)
        loss = criterion(logits, trg_gold.contiguous().view(-1))

        non_pad_tokens = trg_gold.ne(pad_idx).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


def save_checkpoint(path, model, optimizer, config, data_info, epoch, valid_loss, best_valid_loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "data_info": data_info,
        "epoch": epoch,
        "valid_loss": valid_loss,
        "best_valid_loss": best_valid_loss,
    }

    torch.save(checkpoint, path)


def write_log_header(log_path):
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "valid_loss",
            "train_ppl",
            "valid_ppl",
            "epoch_time_sec",
        ])


def append_log(log_path, epoch, train_loss, valid_loss, epoch_time):
    train_ppl = math.exp(min(train_loss, 20))
    valid_ppl = math.exp(min(valid_loss, 20))

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{valid_loss:.6f}",
            f"{train_ppl:.6f}",
            f"{valid_ppl:.6f}",
            f"{epoch_time:.2f}",
        ])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="data/processed/multi30k_en_de.pkl")
    parser.add_argument("--save-name", default="baseline")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-word-vec", type=int, default=256)
    parser.add_argument("--d-inner", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-k", type=int, default=32)
    parser.add_argument("--d-v", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-position", type=int, default=100)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    return TrainConfig(
        data_path=args.data_path,
        save_name=args.save_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        d_inner=args.d_inner,
        n_layers=args.n_layers,
        n_head=args.n_head,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=args.dropout,
        n_position=args.n_position,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=not args.no_amp,
    )


def main():
    config = parse_args()
    set_seed(config.seed)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Training Transformer on Multi30k")
    print("=" * 80)
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    print("=" * 80)
    print("Loading data:", config.data_path)

    with open(config.data_path, "rb") as f:
        data = pickle.load(f)

    train_data = data["train"]
    val_data = data["val"]
    src_itos = data["src_itos"]
    trg_itos = data["trg_itos"]
    pad_idx = data["pad_idx"]

    src_vocab_size = len(src_itos)
    trg_vocab_size = len(trg_itos)

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples:   {len(val_data)}")
    print(f"Source vocab:  {src_vocab_size}")
    print(f"Target vocab:  {trg_vocab_size}")
    print(f"PAD index:     {pad_idx}")

    collate_fn = make_collate_fn(pad_idx)

    train_loader = DataLoader(
        TranslationDataset(train_data),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        TranslationDataset(val_data),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print("=" * 80)
    print("Building model")

    model = build_transformer(
        config=config,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        pad_idx=pad_idx,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    log_path = os.path.join(config.result_dir, f"{config.save_name}_train_log.csv")
    best_ckpt_path = os.path.join(config.checkpoint_dir, f"{config.save_name}_best.pt")
    last_ckpt_path = os.path.join(config.checkpoint_dir, f"{config.save_name}_last.pt")

    write_log_header(log_path)

    data_info = {
        "src_vocab_size": src_vocab_size,
        "trg_vocab_size": trg_vocab_size,
        "pad_idx": pad_idx,
        "src_itos": src_itos,
        "trg_itos": trg_itos,
    }

    best_valid_loss = float("inf")

    print("=" * 80)
    print("Start training")
    print("=" * 80)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        print(f"Epoch {epoch}/{config.epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            pad_idx=pad_idx,
        )

        valid_loss = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            pad_idx=pad_idx,
        )

        epoch_time = time.time() - epoch_start

        train_ppl = math.exp(min(train_loss, 20))
        valid_ppl = math.exp(min(valid_loss, 20))

        print(
            f"Epoch {epoch} finished | "
            f"train_loss {train_loss:.4f} | valid_loss {valid_loss:.4f} | "
            f"train_ppl {train_ppl:.2f} | valid_ppl {valid_ppl:.2f} | "
            f"time {epoch_time:.1f}s"
        )

        append_log(log_path, epoch, train_loss, valid_loss, epoch_time)

        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            config=config,
            data_info=data_info,
            epoch=epoch,
            valid_loss=valid_loss,
            best_valid_loss=best_valid_loss,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                config=config,
                data_info=data_info,
                epoch=epoch,
                valid_loss=valid_loss,
                best_valid_loss=best_valid_loss,
            )

            print(f"  New best checkpoint saved to {best_ckpt_path}")

        print("-" * 80)

    print("=" * 80)
    print("Training finished")
    print(f"Best valid loss: {best_valid_loss:.4f}")
    print(f"Log saved to: {log_path}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
