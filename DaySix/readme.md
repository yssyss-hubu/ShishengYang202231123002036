#  学习日志：对比标准 YOLOv8 模型与自定义 YOLO12 模型在同一任务下的表现差异
> 日期：2025年6月15日  
>  作者：杨轼晟  
> 简介：本次实验旨在基于 Ultralytics 的 YOLO 框架，实现交通标志检测任务，并对比标准 YOLOv8 模型与自定义 YOLO12 模型在同一任务下的表现差异。


---



## 📁 项目结构

```
├── train.py               # 主训练脚本
├── coco8.yaml             # 示例数据配置文件
├── yolov8n.pt             # YOLOv8 预训练权重
├── yolov8n.yaml           # YOLOv8 模型结构定义
├── yolo12.yaml            # 自定义 YOLO12 模型结构定义
├── custom_blocks.py       # 自定义模块 A2C2f, C3k2
├── runs/                  # 保存训练结果
└── README.md              # 项目说明文档
```

---

##  模型结构对比

| 项目    | YOLOv8n (官方)           | YOLO12n (自定义)             |
| ----- | ---------------------- | ------------------------- |
| 架构    | 标准 C2f + SPPF + Detect | 自定义 A2C2f + C3k2 + Detect |
| 层数    | 225 层                  | \~272 层（根据配置 scale=n）     |
| 参数量   | 3.18M                  | 2.60M                     |
| 推理计算量 | 9.0 GFLOPs             | 6.7 GFLOPs                |
| 输入尺寸  | 640x640                | 640x640                   |
| 分类数   | 83 类                   | 83 类（通过 `nc=83` 指定）       |

---

## 🔧 YOLO12 自定义模块说明

* **C3k2**：一种类似 C3 的结构，引入跨层连接，并优化计算复杂度。
* **A2C2f**：自研注意力增强残差模块，融合 SE/block + C2 结构，有助于改善小目标检测效果。

定义位于：

```python
ultralytics/nn/modules/custom_blocks.py
```

使用时需在 `train.py` 中注册模块：

```python
from ultralytics.nn.modules.custom_blocks import A2C2f, C3k2
globals()['A2C2f'] = A2C2f
globals()['C3k2'] = C3k2
```

---

##  训练参数

统一训练配置如下：

```yaml
imgsz: 640
epochs: 20
batch: 16
device: "cuda"
optimizer: auto
pretrained: yolov8n.pt
```

---

##  模型训练命令

YOLOv8：

```python
model = YOLO("yolov8n.yaml").load("yolov8n.pt")
model.train(data="your_data.yaml", epochs=20, imgsz=640, batch=16, device="cuda")
```

YOLO12：

```python
model = YOLO("ultralytics/cfg/models/12/yolo12.yaml").load("yolov8n.pt")
model.train(data="your_data.yaml", epochs=20, imgsz=640, batch=16, device="cuda")
```

---

##  性能评估（示例）

| 模型      | 精度 (mAP\@0.5) | 训练时间  | 参数量   | 文件大小   |
| ------- | ------------- | ----- | ----- | ------ |
| YOLOv8n | 62.5%         | 约10分钟 | 3.18M | 6.2 MB |
| YOLO12n | 64.3%         | 约8分钟  | 2.60M | 5.0 MB |

> 注：准确率和训练时间基于 COCO8 或自定义交通标志数据集，仅供参考。

---

##  总结

| 对比项     | YOLOv8n   | YOLO12n（自定义）   |
| ------- | --------- | -------------- |
| 参数优化    | 常规结构      | 精简参数、引入注意力机制   |
| 小目标检测能力 | 普通        | 更强（因 A2C2f 模块） |
| 易用性     | 官方支持，开箱即用 | 需自定义注册模块       |
| 推理速度    | 较快        | 更快（计算量更低）      |




