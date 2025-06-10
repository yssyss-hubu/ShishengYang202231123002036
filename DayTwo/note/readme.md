# 学习日志：PyTorch 配置与卷积神经网络理论实战

> 日期：2025年6月10日
> 作者：杨轼晟
> 简介：本次学习内容涵盖 PyTorch 环境配置，卷积神经网络（CNN）理论，简单神经网络代码实战，TensorBoard基础使用，深度学习基本概念，以及经典网络 AlexNet 的搭建与理解。

---

## 目录

1. PyTorch 环境搭建与基础工具
2. 数据处理与加载（Dataset 和 DataLoader）
3. 深度学习基础知识回顾
4. 卷积神经网络（CNN）理论详解
5. 简单卷积神经网络代码实战示例
6. AlexNet 网络详解与 PyTorch 实现
7. 训练流程监控与 TensorBoard 使用
8. 模型保存与加载
9. GPU 加速与推理流程

---

## 1. PyTorch 环境搭建与基础工具

### 环境搭建

```bash
# 创建 conda 环境
conda create -n pytorch python=3.9

# 激活环境
conda activate pytorch

# 查看 GPU 驱动
nvidia-smi

# 安装 Jupyter Notebook 支持
conda install nb_conda
```

### 验证 GPU 是否可用

```python
import torch
print(torch.cuda.is_available())  # True 表示 GPU 可用
```

### Python 常用工具

* `dir(obj)`：查看对象属性与方法
* `help(obj)`：查看对象的帮助文档

---

## 2. 数据处理与加载

### Dataset

* 提供数据样本及标签
* 返回数据集大小

### DataLoader

* 批量加载数据
* 支持多线程加速数据读取
* 用于训练中的批量处理

---

## 3. 深度学习基础知识回顾

* **神经网络层次构成**：输入层、隐藏层、输出层
* **激活函数**：ReLU、Sigmoid、Tanh 等
* **损失函数**：衡量模型预测与真实标签的差距（如交叉熵、MSE）
* **优化器**：如 SGD、Adam
* **正则化**：Dropout、权重衰减，防止过拟合

---

## 4. 卷积神经网络（CNN）理论详解

### 4.1 卷积层 (Convolutional Layer)

* 作用：提取输入的局部特征
* 关键参数：输入通道数、输出通道数、卷积核大小、步长、填充





### 4.2 池化层 (Pooling Layer)

* 作用：降采样，降低特征图尺寸和计算量
* 类型：最大池化（MaxPool）、平均池化（AvgPool）
* 关键参数：池化窗口大小、步长

---

## 5. 简单卷积神经网络代码实战示例

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 16 * 16, 10)  # 假设输入为32x32

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 测试
model = SimpleCNN()
input_tensor = torch.randn(4, 3, 32, 32)
output = model(input_tensor)
print(output.shape)  # [4, 10]
```

---

## 6. AlexNet 网络详解与 PyTorch 实现

### 6.1 AlexNet 介绍

* 由 Alex Krizhevsky 等人于2012年提出
* 使用 5 层卷积层，3 层池化层，3 个全连接层
* 引入 ReLU 激活和 Dropout 正则化
* 解决大规模图像分类任务，ImageNet 成功案例

### 6.2 AlexNet 结构示意

| 层     | 类型                   | 参数示例                 | 输出尺寸（近似）  |
| ----- | -------------------- | -------------------- | --------- |
| Conv1 | Conv + ReLU          | 96个11x11卷积核，步长4      | 55x55x96  |
| Pool1 | MaxPool              | 3x3核，步长2             | 27x27x96  |
| Conv2 | Conv + ReLU          | 256个5x5卷积核，padding=2 | 27x27x256 |
| Pool2 | MaxPool              | 3x3核，步长2             | 13x13x256 |
| Conv3 | Conv + ReLU          | 384个3x3卷积核，padding=1 | 13x13x384 |
| Conv4 | Conv + ReLU          | 384个3x3卷积核，padding=1 | 13x13x384 |
| Conv5 | Conv + ReLU          | 256个3x3卷积核，padding=1 | 13x13x256 |
| Pool3 | MaxPool              | 3x3核，步长2             | 6x6x256   |
| FC1   | 全连接 + ReLU + Dropout | 4096个节点              | 4096      |
| FC2   | 全连接 + ReLU + Dropout | 4096个节点              | 4096      |
| FC3   | 全连接                  | 类别数                  | 类别数       |

### 6.3 PyTorch 实现代码

```python
import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 测试
if __name__ == "__main__":
    model = AlexNet(num_classes=10)
    input_tensor = torch.randn(4, 3, 224, 224)
    output = model(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)
```

---

## 7. 训练流程监控与 TensorBoard 使用

### 使用 TensorBoard 监控训练

```bash
tensorboard --logdir=logs#地址
```

### 训练时查看 Loss 和 Accuracy

* 通过打印 loss 数值监控训练
* 利用 TensorBoard 可视化指标曲线

---

## 8. 模型保存与加载

```python
# 保存模型参数
# torch.save(model.state_dict(), "model.pth")
# 
# # 加载模型参数
# model.load_state_dict(torch.load("model.pth"))
```

---

## 9. GPU 加速与推理流程

```python
# 将模型和数据移到 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# inputs = inputs.to(device)
# labels = labels.to(device)
```

### 推理示例

```python
# model.eval()
# with torch.no_grad():
#     outputs = model(inputs)
#     _, preds = torch.max(outputs, 1)
```

---

# 总结

* PyTorch 是深度学习实践的强大工具，掌握其环境搭建和数据处理是基础。
* 理解卷积层和池化层的工作原理，有助于设计合理的 CNN 模型。
* AlexNet 作为经典网络，学习其结构和实现有助于更深入理解 CNN。
* 训练过程中合理监控和保存模型，使用 GPU 加速训练与推理，是深度学习工作流的关键环节。


