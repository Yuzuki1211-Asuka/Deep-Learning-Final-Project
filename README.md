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
| 杨子木 | 2024214749 | 代码复现、实验训练、结果分析、README 整理 |
| 申婧 | 2024214709 | 论文阅读、报告撰写 |
| 康馨文 |202421474 | 海报制作、PPT编辑 |


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

训练过程中 loss 持续下降，验证集 perplexity 明显降低，说明模型能够学习英德翻译任务中的序列映射关系。

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

实验分析：

1. small_128_2l 参数量最少，训练速度最快，但验证集 loss 最高，说明模型表达能力较弱。 
2. baseline_256_3l 取得最低的验证集 loss，说明该配置在模型规模、训练时间和效果之间较为平衡。 
3. deeper_256_4l 参数量更多，训练时间更长，但 10 个 epoch 内没有超过 baseline，说明在训练轮数有限和学习率策略较简单的情况下，模型变深不一定带来更好效果。

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

- Embedding 层参数量与词表大小和 d_model 直接相关； 
- Multi-Head Attention 参数量主要来自 Q、K、V 和输出投影矩阵； 
- Feed Forward Network 参数量与 d_model 和 d_inner 有关； 
- Decoder 参数量大于 Encoder，因为 Decoder 除了 self-attention，还包含 encoder-decoder attention。

---

## 12. Translation Examples

部分测试集预测结果如下:

| English Input | German Reference | Model Prediction |
|---|---|---|
| a man in an orange hat staring at something . | ein mann mit einem orangefarbenen hut , der etwas anstarrt . | ein mann in einem blauen hemd hält sich auf dem boden . |
| a boston terrier is running on lush green grass in front of a white fence . | ein boston terrier läuft über saftig-grünes gras vor einem weißen zaun . | ein fährt auf einem auf einem . |
| a girl in karate uniform breaking a stick with a front kick . | ein mädchen in einem karateanzug bricht ein brett mit einem tritt . | ein mädchen in einem steht auf einem . |
| five people wearing winter jackets and helmets stand in the snow , with snowmobiles in the background . | fünf leute in winterjacken und mit helmen stehen im schnee mit schneemobilen im hintergrund . | mehrere personen in und stehen auf dem boden und schauen sich auf dem boden . |

结果分析:

- 模型能够生成德语形式的句子，并学到部分常见词汇和短语； 
- 对于简单结构，模型能生成较合理的结果； 
- 对于复杂句子或细节描述，模型容易生成高频模板句； 
- 由于只训练 10 个 epoch，并且使用 greedy decoding，没有使用 beam search，因此预测质量仍有提升空间。

---

## 13. Problems and Solutions

### 13.1 AMP 半精度训练溢出

训练时曾出现如下错误:

```text
RuntimeError: value cannot be converted to type at::Half without overflow
```

原因:

原始 attention mask 使用了:

```python
-1e9
```

该数值在 FP16 半精度下超出表示范围.

解决方法:

使用 FP32 训练:

```bash
--no-amp
```

---

### 13.2 Compatibility with Old TorchText APIs

原仓库的数据处理和训练流程依赖较旧版本的 torchtext，在当前 Python 3.12 和 PyTorch 2.5 环境下可能不兼容。

解决方法:

我们组保留 transformer/ 模型代码，重写数据处理、训练、绘图、参数统计和预测脚本，这样既保留了 Transformer 核心结构，又提高了代码可运行性和可解释性。

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

尽管该项目已满足课程的主要要求，但仍有以下几点改进空间：

1. 添加 BLEU 分数评估；
2. 添加束搜索解码；
3. 使用原始 Transformer 的学习率预热方案；
4. 增加训练轮数；
5. 添加注意力权重可视化；
6. 尝试使用更大的数据集，例如 IWSLT；
7. 改进分词器和词汇表构建；
8. 比较更多超参数，例如dropout、批量大小和学习率。

---

## 16. Conclusion

在本项目中，我们基于 PyTorch 实现重现了 Transformer 模型，并将其改编为深度学习课程项目。我们完成了 Transformer 的核心模块，包括缩放点积注意力、多头注意力、编码器、解码器、位置编码、前馈网络以及掩码机制。

我们使用Multi30k英德翻译数据集对模型进行了训练，并获得了实际的训练和验证结果。基准模型在经过10个训练周期后，验证损失达到了3.1889的最佳值。此外，我们还对比了三种不同规模的模型，并分析了参数数量、训练时间与模型性能之间的关系。

实验结果表明，更大的模型通常具有更强的表征能力，但在训练轮数有限的情况下，更深的模型并不一定表现更好。在我们的实验中，基线模型在参数数量、训练速度和验证性能之间实现了最佳平衡。

该项目帮助我们理解了Transformer的内部结构、注意力机制的实现、序列到序列模型的训练过程，以及模型规模与实验性能之间的实际关系。

---

## 17. References

1. Vaswani et al. *Attention Is All You Need*. 2017.
2. Reference implementation: `jadore801120/attention-is-all-you-need-pytorch`
3. Multi30k dataset
4. PyTorch documentation
