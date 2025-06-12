# 📘 学习日志要点总结：总结了 Transformer 模型（以 Vision Transformer 为例）的实现，以及如何构建并训练一个用于图像分类的模型框架（以 MogaNet 为例）

> 📅 日期：2025年6月12日
> 👤 作者：杨轼晟

## 📚 学习目标

今天的学习围绕两个核心主题：

1. **理解并实现 Vision Transformer (ViT)**
2. **构建完整的图像分类训练流程，支持自定义数据集和网络（MogaNet）**

---

## ✅ 一、实现 Vision Transformer (ViT)

### 🔍 核心组件解析

* **Patch Embedding**：

  * 输入序列按固定大小进行分块（`patch_size`），每个 patch 展开并线性映射为嵌入向量。
* **Position Embedding + Class Token**：

  * 增加位置编码，加入 `cls_token` 表示整个序列的摘要信息。
* **Transformer Encoder**：

  * 使用多个多头自注意力（Multi-head Self-Attention）+ 前馈网络（Feed Forward）层堆叠。
* **分类头（MLP Head）**：

  * 取 `cls_token` 的输出作为分类输入，输出 logits。

### ✅ ViT 流程结构

```text
Input → Patch Embedding → [Class Token + Positional Embedding] → Transformer Encoder × N → MLP Head → Logits
```

### 📦 测试代码样例

```python
v = ViT(
    seq_len=256, patch_size=16, num_classes=1000,
    dim=1024, depth=6, heads=8, mlp_dim=2048
)
time_series = torch.randn(4, 3, 256)
logits = v(time_series)
```

---

## ✅ 二、图像分类训练流程（MogaNet）

### 🛠️ 数据预处理与增强

* 使用 `torchvision.transforms` 构建训练和验证增强管道
* 包含 `Resize`, `Crop`, `ColorJitter`, `Normalize` 等操作

### 📁 自定义数据加载

* 使用 `ImageFolder` 支持标准结构的数据集
* 自定义 `RGBImageFolder` 确保图像为 RGB 格式

### 🧠 构建模型

* 使用 `MogaNet` 架构作为分类主干网络
* 支持多 GPU 训练（`nn.DataParallel`）

### 🌀 训练与验证循环

* 包含训练 (`train_one_epoch`) 和验证 (`evaluate`) 两个阶段
* 使用 `CrossEntropyLoss` 与 `Adam` 优化器
* 学习率调度器：`CosineAnnealingLR`

### 📊 训练可视化

* 利用 `matplotlib` 绘图展示训练准确率与损失
* 保存图表为 `training_plot.png`

---




## 🔚 总结

* 学会了使用 PyTorch 手动实现 Transformer 模块（包括 Attention、FeedForward、LayerNorm 等）
* 熟悉了自定义数据集的加载、增强、模型训练流程
* 理解了 Transformer 在图像处理中的应用方式（ViT）
* 完成了基于 MogaNet 的图像分类完整 pipeline


