import os
import sys
import inspect
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from transformer.Models import Transformer


def build_transformer():
    init_params = set(inspect.signature(Transformer.__init__).parameters.keys())

    kwargs = {
        "n_src_vocab": 1000,
        "n_trg_vocab": 1200,
        "src_pad_idx": 0,
        "trg_pad_idx": 0,
        "d_word_vec": 256,
        "d_model": 256,
        "d_inner": 512,
        "n_layers": 3,
        "n_head": 8,
        "d_k": 32,
        "d_v": 32,
        "dropout": 0.1,
        "n_position": 100,
        "trg_emb_prj_weight_sharing": False,
        "emb_src_trg_weight_sharing": False,
        "scale_emb_or_prj": "none",
    }

    filtered_kwargs = {
        key: value for key, value in kwargs.items()
        if key in init_params
    }

    print("Transformer init parameters used:")
    for key, value in filtered_kwargs.items():
        print(f"  {key}: {value}")

    return Transformer(**filtered_kwargs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Device:", device)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("=" * 60)

    model = build_transformer().to(device)
    model.eval()

    batch_size = 4
    src_len = 12
    trg_len = 10
    src_vocab_size = 1000
    trg_vocab_size = 1200

    src_seq = torch.randint(1, src_vocab_size, (batch_size, src_len)).to(device)
    trg_seq = torch.randint(1, trg_vocab_size, (batch_size, trg_len)).to(device)

    with torch.no_grad():
        output = model(src_seq, trg_seq)

    print("=" * 60)
    print("src_seq shape:", src_seq.shape)
    print("trg_seq shape:", trg_seq.shape)
    print("output shape:", output.shape)

    expected_shape = (batch_size * trg_len, trg_vocab_size)
    print("expected shape:", expected_shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if output.shape == expected_shape:
        print("Model forward check passed.")
    else:
        print("Warning: output shape is different from expected shape.")

    print("=" * 60)


if __name__ == "__main__":
    main()
