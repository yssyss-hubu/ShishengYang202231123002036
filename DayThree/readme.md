

# 📘 学习日志要点总结：CNN模型基础 + MogaNet训练自定义数据集

> 📅 日期：2025年6月11日
> 👤 作者：杨轼晟

---

##  一、CNN模型对比解析

###  1. ResNet（残差网络）

* 提出者：He et al.（2015）
* 特点：

  * 使用残差连接（Skip Connection）解决深层网络退化问题
  * 常见结构：ResNet18、34、50、101
* 优点：

  * 易于训练深层网络
  * 表达能力强

---

###  2. GoogLeNet（Inception v1）

* 提出者：Szegedy et al.（2014）
* 特点：

  * Inception 模块，多尺度卷积（1x1, 3x3, 5x5）
  * 网络较深但参数少
* 优点：

  * 多路径提取特征
  * 计算效率高

---

###  3. MobileNet（V1/V2/V3）

* 提出者：Google（2017-2019）
* 特点：

  * 轻量级网络，适合移动端
  * 使用深度可分离卷积（Depthwise Separable Conv）
  * MobileNetV2：倒残差结构（Inverted Residual）
  * MobileNetV3：结合 NAS 和注意力机制
* 优点：

  * 参数少、推理快
  * 性能与效率平衡

---

###  4. MogaNet（多阶门控聚合网络）

* 提出者：华为 Noah’s Ark Lab（2022）
* 特点：

  * 多阶特征聚合 + Gated 模块
  * 模拟 Transformer 的表达能力
  * 使用 Drop Path 和 LayerScale 稳定训练
* 优点：

  * 结合 CNN 推理效率与 Transformer 精度
  * 高效处理视觉任务

---

##  二、使用 MogaNet 训练自定义数据集

###  1. 数据集格式要求（适用于 `ImageFolder`）

```bash
dataset/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

---

###  2. 训练配置示例

```python
config = {
    'data_dir': './dataset',
    'arch': 'tiny',  # 支持 tiny / small / base / large
    'num_classes': 10,
    'batch_size': 96,
    'epochs': 5,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
```

---

###  3. 构建 MogaNet 模型

```python
from moganet import MogaNet

model = MogaNet(
    arch='tiny',
    num_classes=10,
    drop_path_rate=0.1,
    attn_force_fp32=True,
    fork_feat=False
).to(config['device'])
```

---

###  4. 加载数据集

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = datasets.ImageFolder('./dataset/train', transform=transform)
val_dataset = datasets.ImageFolder('./dataset/val', transform=transform)
```

---

###  5. 训练主循环简化版

```python
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=96)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

##  三、注意事项

* ✅ `num_workers` 和 `batch_size` 需根据硬件调整
* ✅ `timm` 新版需替换导入路径，如：

  ```python
  from timm.models.layers import DropPath
  ```
* ✅ Windows 用户多进程时需加入：

```python
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
```

---

##  四、依赖安装

```bash
pip install torch torchvision timm tqdm
```

---

##  五、推荐可视化工具

* TensorBoard：训练过程实时监控
* wandb：更丰富的训练追踪
* matplotlib：手动绘图分析 loss 和 acc 曲线

---



