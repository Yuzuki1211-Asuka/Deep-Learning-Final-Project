# Deep Learning Final Project: Transformer Reproduction and Analysis

## 1. Project Overview

本项目是《深度学习》课程期末 Project，主题为 **Transformer 论文阅读、代码复现与实验分析**。

本项目基于论文 **Attention Is All You Need** 中提出的 Transformer 架构，参考开源 PyTorch 项目 `attention-is-all-you-need-pytorch`，完成了 Transformer 的核心代码复现、数据集训练、实验结果分析、模型参数统计和预测样例展示。

本项目重点不是追求最优翻译指标，而是理解 Transformer 的核心结构、代码实现和训练流程。项目内容包括：

- 阅读并理解 Transformer 模型结构；
- 复现 Transformer 的核心模块；
- 在公开数据集 Multi30k 上进行真实训练；
- 绘制训练集和验证集 Loss 曲线；
- 给出测试集翻译预测样例；
- 统计模型参数量；
- 分析模型规模、训练时间和模型效果之间的关系；
- 使用 GitHub 管理和展示项目成果。

---

## 2. Group Members

| Name | Student ID | Responsibility |
|---|---|---|
| 请填写姓名 | 请填写学号 | 论文阅读、代码复现、实验训练、结果分析、README 整理 |

> 如果是小组项目，请在此处补充所有成员信息和分工。

---

## 3. Reference Project

本项目参考了以下开源项目：

```text
https://github.com/jadore801120/attention-is-all-you-need-pytorch
```

主要参考内容包括：

- Transformer 模型整体结构；
- Scaled Dot-Product Attention；
- Multi-Head Attention；
- Encoder Layer；
- Decoder Layer；
- Positional Encoding；
- Padding Mask 和 Subsequent Mask；
- Transformer Encoder / Decoder 代码组织方式。

在参考原项目的基础上，本项目进行了课程项目化改造：

- 保留 `transformer/` 中的核心模型结构；
- 重新编写数据处理脚本；
- 重新编写训练与验证脚本；
- 增加 Loss 曲线绘制脚本；
- 增加模型参数统计脚本；
- 增加测试集翻译预测脚本；
- 完成三组超参数对比实验。

---

## 4. Project Structure

```text
Deep-Learning-Final-Project/
│
├── transformer/
│   ├── Constants.py
│   ├── Modules.py
│   ├── SubLayers.py
│   ├── Layers.py
│   ├── Models.py
│   ├── Optim.py
│   └── Translator.py
│
├── scripts/
│   ├── 01_check_model.py
│   ├── 02_prepare_data.py
│   ├── 03_train_plain.py
│   ├── 04_plot_loss.py
│   ├── 05_count_params.py
│   └── 06_translate_plain.py
│
├── figures/
│   ├── baseline_256_3l_loss_curve.png
│   ├── small_128_2l_loss_curve.png
│   └── deeper_256_4l_loss_curve.png
│
├── results/
│   ├── baseline_256_3l_train_log.csv
│   ├── baseline_256_3l_param_stats.csv
│   ├── baseline_256_3l_predictions.csv
│   ├── small_128_2l_train_log.csv
│   ├── small_128_2l_param_stats.csv
│   ├── small_128_2l_predictions.csv
│   ├── deeper_256_4l_train_log.csv
│   ├── deeper_256_4l_param_stats.csv
│   └── deeper_256_4l_predictions.csv
│
├── data/
│   ├── raw/
│   └── processed/
│
├── checkpoints/
├── configs/
├── report/
├── PPT/
├── poster/
├── requirements.txt
├── .gitignore
└── README.md
```

说明：

- `transformer/`：原始 Transformer 模型核心代码；
- `scripts/`：本项目新增的实验脚本；
- `figures/`：Loss 曲线图片；
- `results/`：训练日志、参数统计和预测样例；
- `data/`：数据集目录，实际数据未上传到 GitHub；
- `checkpoints/`：模型权重目录，`.pt` 文件未上传到 GitHub。

由于数据文件和模型权重较大，`data/raw/`、`data/processed/*.pkl` 和 `checkpoints/*.pt` 已通过 `.gitignore` 排除。运行脚本后可以自动重新生成数据和模型文件。

---

## 5. Transformer Model Introduction

Transformer 是一种完全基于注意力机制的序列建模结构。与 RNN、LSTM、GRU 等循环结构不同，Transformer 不依赖递归计算，而是通过 Self-Attention 在序列内部直接建模 token 之间的关系，因此具有更强的并行计算能力。

本项目复现的 Transformer 包含以下核心模块：

### 5.1 Input Embedding

代码位置：

```text
transformer/Models.py
```

Input Embedding 将输入 token id 映射为连续向量表示。

在本项目中，源语言和目标语言分别有独立的 embedding：

```text
src_word_emb
trg_word_emb
```

---

### 5.2 Positional Encoding

代码位置：

```text
transformer/Models.py
```

由于 Transformer 不使用 RNN 或 CNN，因此需要额外加入位置信息。Positional Encoding 用于向 token embedding 中注入序列位置信息，使模型能够感知词语顺序。

---

### 5.3 Scaled Dot-Product Attention

代码位置：

```text
transformer/Modules.py
```

核心公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

其中：

- `Q`：Query；
- `K`：Key；
- `V`：Value；
- `sqrt(d_k)`：缩放因子，用于避免点积结果过大；
- `mask`：用于屏蔽 padding token 或未来 token。

---

### 5.4 Multi-Head Attention

代码位置：

```text
transformer/SubLayers.py
```

Multi-Head Attention 将输入映射到多个注意力头，每个头独立计算注意力，然后将多个头的结果拼接并线性映射。

主要流程：

```text
Input
→ Linear projections for Q, K, V
→ Split into multiple heads
→ Scaled Dot-Product Attention
→ Concatenate heads
→ Final linear projection
→ Residual connection
→ Layer normalization
```

---

### 5.5 Position-wise Feed Forward Network

代码位置：

```text
transformer/SubLayers.py
```

Feed Forward Network 对每个位置的表示进行非线性变换，通常由两层线性层和 ReLU 激活函数组成。

---

### 5.6 Encoder Layer

代码位置：

```text
transformer/Layers.py
```

Encoder Layer 结构：

```text
Self-Attention
→ Add & Norm
→ Feed Forward Network
→ Add & Norm
```

---

### 5.7 Decoder Layer

代码位置：

```text
transformer/Layers.py
```

Decoder Layer 结构：

```text
Masked Self-Attention
→ Add & Norm
→ Encoder-Decoder Attention
→ Add & Norm
→ Feed Forward Network
→ Add & Norm
```

---

### 5.8 Mask Mechanism

代码位置：

```text
transformer/Models.py
```

本项目包含两类 mask：

| Mask | Function |
|---|---|
| Padding Mask | 屏蔽 `<pad>` token，避免模型关注无效位置 |
| Subsequent Mask | 屏蔽未来 token，防止 Decoder 在训练时提前看到后续词 |

---

## 6. Environment

实验环境如下：

```text
GPU: NVIDIA GeForce RTX 4090
GPU Memory: 24 GB
Python: 3.12.3
PyTorch: 2.5.1+cu124
CUDA available: True
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果缺少绘图或数据处理依赖，可以执行：

```bash
pip install pandas matplotlib
```

---

## 7. Dataset

本项目使用 **Multi30k 英德翻译数据集**。

数据处理脚本：

```text
scripts/02_prepare_data.py
```

处理后的数据规模如下：

| Split | Number of Samples |
|---|---:|
| Train | 29,000 |
| Validation | 1,014 |
| Test | 1,000 |

词表规模如下：

| Vocabulary | Size |
|---|---:|
| Source English Vocabulary | 5,921 |
| Target German Vocabulary | 7,859 |

特殊 token：

| Token | Meaning |
|---|---|
| `<pad>` | padding token |
| `<unk>` | unknown token |
| `<bos>` | beginning of sentence |
| `<eos>` | end of sentence |

生成的数据文件：

```text
data/processed/multi30k_en_de.pkl
```

该文件未上传到 GitHub，可以通过数据预处理脚本重新生成。

---

## 8. How to Run

### 8.1 Check Model Forward Pass

```bash
python scripts/01_check_model.py
```

Expected output:

```text
src_seq shape: torch.Size([4, 12])
trg_seq shape: torch.Size([4, 10])
output shape: torch.Size([40, 1200])
Model forward check passed.
```

This step verifies that the Transformer model can perform a forward pass correctly.

---

### 8.2 Prepare Dataset

```bash
python scripts/02_prepare_data.py
```

Expected output:

```text
Train pairs: 29000
Val pairs:   1014
Test pairs:  1000
Source vocab size: 5921
Target vocab size: 7859
```

---

### 8.3 Train Baseline Model

```bash
python scripts/03_train_plain.py \
  --save-name baseline_256_3l \
  --batch-size 64 \
  --epochs 10 \
  --lr 1e-4 \
  --d-model 256 \
  --d-word-vec 256 \
  --d-inner 512 \
  --n-layers 3 \
  --n-head 8 \
  --d-k 32 \
  --d-v 32 \
  --dropout 0.1 \
  --no-amp
```

说明：

本项目训练时使用 `--no-amp`。原因是原始 attention mask 中使用了 `-1e9`，在 FP16 混合精度训练下可能导致 overflow。因此本实验采用 FP32 训练。

---

### 8.4 Plot Loss Curve

```bash
python scripts/04_plot_loss.py \
  --log-file results/baseline_256_3l_train_log.csv \
  --save-path figures/baseline_256_3l_loss_curve.png \
  --title "Baseline Transformer Loss Curve"
```

---

### 8.5 Count Model Parameters

```bash
python scripts/05_count_params.py \
  --checkpoint checkpoints/baseline_256_3l_best.pt \
  --save-prefix baseline_256_3l
```

This script reports:

- total parameters;
- trainable parameters;
- embedding parameters;
- multi-head attention parameters;
- feed forward network parameters;
- encoder parameters;
- decoder parameters;
- output projection parameters.

---

### 8.6 Generate Translation Examples

```bash
python scripts/06_translate_plain.py \
  --checkpoint checkpoints/baseline_256_3l_best.pt \
  --data-path data/processed/multi30k_en_de.pkl \
  --save-path results/baseline_256_3l_predictions.csv \
  --num-samples 20
```

You can also translate a custom sentence:

```bash
python scripts/06_translate_plain.py \
  --checkpoint checkpoints/baseline_256_3l_best.pt \
  --data-path data/processed/multi30k_en_de.pkl \
  --sentence "a man is riding a bicycle ."
```

---

## 9. Experimental Settings

本项目共完成三组模型规模对比实验：

| Experiment | d_model | Layers | Heads | d_inner | Batch Size | Epochs | Learning Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| small_128_2l | 128 | 2 | 4 | 256 | 64 | 10 | 1e-4 |
| baseline_256_3l | 256 | 3 | 8 | 512 | 64 | 10 | 1e-4 |
| deeper_256_4l | 256 | 4 | 8 | 512 | 64 | 10 | 1e-4 |

Loss function:

```text
CrossEntropyLoss(ignore_index=pad_idx)
```

Optimizer:

```text
Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
```

Decoding method:

```text
Greedy decoding
```

---

## 10. Experimental Results

### 10.1 Baseline Training Results

Baseline configuration:

```text
d_model = 256
n_layers = 3
n_head = 8
d_inner = 512
batch_size = 64
epochs = 10
learning_rate = 1e-4
```

Training results:

| Metric | Initial | Final |
|---|---:|---:|
| Train Loss | 5.6549 | 3.3808 |
| Validation Loss | 4.8124 | 3.1889 |
| Validation Perplexity | 123.03 | 24.26 |

The training and validation losses decreased steadily, indicating that the reproduced Transformer model successfully learned useful sequence-to-sequence mappings on the Multi30k English-German translation task.

Loss curve:

```text
figures/baseline_256_3l_loss_curve.png
```

---

### 10.2 Comparison of Model Scales

| Experiment | d_model | Layers | Heads | Parameters | Best Valid Loss | Final Valid PPL | Time / Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| small_128_2l | 128 | 2 | 4 | 3,429,760 | 3.4261 | 30.76 | about 18.9s |
| baseline_256_3l | 256 | 3 | 8 | 9,485,056 | 3.1889 | 24.26 | about 24s |
| deeper_256_4l | 256 | 4 | 8 | 10,799,872 | 3.3309 | 27.96 | about 29.3s |

Analysis:

1. `small_128_2l` has the fewest parameters and the fastest training speed, but its validation loss is the highest. This suggests that a smaller model has weaker representation ability.
2. `baseline_256_3l` achieves the lowest validation loss and the best validation perplexity among the three settings. It provides the best trade-off between model size, training time, and performance.
3. `deeper_256_4l` has more parameters and requires more training time, but it does not outperform the baseline within 10 epochs. This suggests that a deeper model is not always better under limited training epochs and a simple learning rate schedule.

---

## 11. Parameter Analysis

### 11.1 Total Parameters

| Experiment | Total Parameters | Trainable Parameters |
|---|---:|---:|
| small_128_2l | 3,429,760 | 3,429,760 |
| baseline_256_3l | 9,485,056 | 9,485,056 |
| deeper_256_4l | 10,799,872 | 10,799,872 |

---

### 11.2 Baseline Parameter Breakdown

| Module | Parameters |
|---|---:|
| Source Embedding | 1,515,776 |
| Target Embedding | 2,011,904 |
| Multi-Head Attention | 2,363,904 |
| Feed Forward Network | 1,580,544 |
| Encoder Total | 3,094,528 |
| Decoder Total | 4,378,624 |
| Output Projection | 2,011,904 |
| Total | 9,485,056 |

Analysis:

- The source embedding parameter count is determined by source vocabulary size and `d_model`.
- The target embedding and output projection contain many parameters because the German target vocabulary is larger.
- Multi-Head Attention parameters mainly come from Q, K, V projections and the output projection.
- Feed Forward Network parameters are determined by `d_model` and `d_inner`.
- The Decoder has more parameters than the Encoder because each decoder layer contains both masked self-attention and encoder-decoder attention.

---

## 12. Translation Examples

Some test set predictions from the baseline model are shown below.

| English Input | German Reference | Model Prediction |
|---|---|---|
| a man in an orange hat staring at something . | ein mann mit einem orangefarbenen hut , der etwas anstarrt . | ein mann in einem blauen hemd hält sich auf dem boden . |
| a boston terrier is running on lush green grass in front of a white fence . | ein boston terrier läuft über saftig-grünes gras vor einem weißen zaun . | ein fährt auf einem auf einem . |
| a girl in karate uniform breaking a stick with a front kick . | ein mädchen in einem karateanzug bricht ein brett mit einem tritt . | ein mädchen in einem steht auf einem . |
| five people wearing winter jackets and helmets stand in the snow , with snowmobiles in the background . | fünf leute in winterjacken und mit helmen stehen im schnee mit schneemobilen im hintergrund . | mehrere personen in und stehen auf dem boden und schauen sich auf dem boden . |

Analysis:

- The model can generate German-like sentence structures.
- It learns some high-frequency words and simple sentence patterns.
- It still struggles with long sentences and detailed visual descriptions.
- Some predictions contain repeated phrases or generic templates.
- Possible reasons include limited epochs, greedy decoding, no beam search, and relatively simple learning rate scheduling.

---

## 13. Problems and Solutions

### 13.1 AMP Overflow Problem

During early training, the following error occurred:

```text
RuntimeError: value cannot be converted to type at::Half without overflow
```

Reason:

The attention mask in the original implementation uses:

```python
-1e9
```

This value may overflow under FP16 mixed precision training.

Solution:

Use FP32 training by adding:

```bash
--no-amp
```

---

### 13.2 Compatibility with Old TorchText APIs

The original project depends on older `torchtext` APIs, which may not be compatible with the current Python and PyTorch environment.

Solution:

This project keeps the model implementation and rewrites the following parts:

- data preprocessing;
- dataloader construction;
- training loop;
- validation loop;
- loss logging;
- parameter counting;
- prediction generation.

This makes the project easier to run, explain, and analyze in the course setting.

---

## 14. What We Have Completed

This project has completed the following requirements:

| Requirement | Status |
|---|---|
| Transformer core code reproduction | Completed |
| Input Embedding | Completed |
| Positional Encoding | Completed |
| Scaled Dot-Product Attention | Completed |
| Multi-Head Attention | Completed |
| Feed Forward Network | Completed |
| Encoder Layer | Completed |
| Decoder Layer | Completed |
| Padding Mask and Subsequent Mask | Completed |
| Dataset preprocessing | Completed |
| Vocabulary construction | Completed |
| Model training | Completed |
| Validation | Completed |
| Loss curve visualization | Completed |
| Prediction examples | Completed |
| Parameter counting | Completed |
| Hyperparameter comparison | Completed |
| GitHub project organization | Completed |

---

## 15. Limitations and Future Work

Although the project has completed the main course requirements, there are still several possible improvements:

1. Add BLEU score evaluation;
2. Add beam search decoding;
3. Use the original Transformer learning rate warmup schedule;
4. Train for more epochs;
5. Add attention weight visualization;
6. Try larger datasets such as IWSLT;
7. Improve tokenizer and vocabulary construction;
8. Compare more hyperparameters such as dropout, batch size, and learning rate.

---

## 16. Conclusion

In this project, we reproduced the Transformer model based on a PyTorch implementation and adapted it for a deep learning course project. We completed the core Transformer modules, including Scaled Dot-Product Attention, Multi-Head Attention, Encoder, Decoder, Positional Encoding, Feed Forward Network, and mask mechanisms.

We trained the model on the Multi30k English-German translation dataset and obtained real training and validation results. The baseline model achieved a best validation loss of 3.1889 after 10 epochs. We also compared three different model sizes and analyzed the relationship between parameter count, training time, and model performance.

The experimental results show that a larger model generally has stronger representation ability, but a deeper model does not necessarily perform better under limited training epochs. The baseline model provides the best balance among parameter count, training speed, and validation performance in our experiments.

This project helped us understand the internal structure of Transformer, the implementation of attention mechanisms, the training process of sequence-to-sequence models, and the practical relationship between model scale and experimental performance.

---

## 17. References

1. Vaswani et al. *Attention Is All You Need*. 2017.
2. Reference implementation: `jadore801120/attention-is-all-you-need-pytorch`
3. Multi30k dataset
4. PyTorch documentation
