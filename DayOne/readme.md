#  学习日志：Python + Git + Anaconda + PyTorch 环境配置

> 日期：2025年6月9日  
>  作者：杨轼晟  
> 简介：本次学习覆盖了 Python 基础语法、Git 常用命令、Anaconda 环境管理与 PyTorch 框架安装，遥感图像数据处理的入门理解和实战代码实现。

---

##  一、Python 基本语法

学习内容包括：

-  变量与类型（int, float, str, bool, list, dict 等）
-  控制结构（`if` 条件判断、`for` 和 `while` 循环）
-  函数定义与调用（使用 `def` 关键字）
-  注释写法（`#` 单行注释，`''' '''` 多行注释）
-  常见内置函数：`print()`、`len()`、`range()` 等

---

## 🔧 二、Git 基础命令

用于版本控制和项目管理：

```bash
# 初始化本地仓库
git init

# 克隆远程仓库
git clone <repo_url>

# 查看状态和提交历史
git status
git log

# 添加文件并提交更改
git add <filename>
git commit -m "提交说明"

# 推送到远程仓库（如 GitHub）
git push origin main
````

---

##  三、Anaconda 安装与虚拟环境配置

> Anaconda 是用于管理 Python 环境和库的工具，推荐用于科学计算和深度学习项目。

### 安装步骤：

1. 前往官网下载安装：[https://www.anaconda.com](https://www.anaconda.com)
2. 创建并激活虚拟环境：

   ```bash
   conda create -n myenv python=3.10
   conda activate myenv
   ```
3. 安装常用库：

   ```bash
   conda install numpy pandas matplotlib
   ```

---

##  四、PyTorch 安装

PyTorch 是一个用于深度学习的开源框架。



### GPU（CUDA）版本安装：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 安装验证：

```python
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())  # 如果输出 True，说明 GPU 可用
```

---


##  五、遥感影像处理 - 多波段 TIFF 转 RGB 图像

本次学习了如何使用 Python 结合 `rasterio` 和 `PIL` 库处理卫星影像数据，实现多波段 TIFF 文件到 RGB 图片的转换。关键点包括：

### 1. 影像数据范围缩放
- 遥感影像数据通常数值范围是 0-10000，需要压缩到 0-255 以符合普通图像显示标准。
- 使用了 `np.clip` 和线性缩放函数 `scale_to_255()`。

### 2. 多波段数据读取与合成
- 使用 `rasterio` 读取 TIFF 文件的单波段或多波段数据。
- 支持两种输入模式：
  - 多波段 TIFF 文件（`create_rgb_from_multiband_tif`）
  - 分别保存的单波段 TIFF 文件（`create_rgb_from_separate_tifs`）

### 3. 真彩色图像的归一化处理
- 归一化处理 (`normalize=True`) 将每个波段数据映射到 0-255 的范围，提升图像显示效果。

### 4. 代码核心函数说明
```python
def scale_to_255(arr, max_val=10000):
    """将原始值从 0–10000 压缩到 0–255"""
    arr_scaled = np.clip((arr / max_val) * 255, 0, 255)
    return arr_scaled.astype(np.uint8)

def save_image_from_bands(red, green, blue, output_path, normalize=True, max_val=10000):
    # 根据 normalize 选择归一化或线性缩放处理
    # 最终保存 RGB 图片到指定路径
    
---


## 六、 今日总结

| 模块        | 状态 | 说明            |
| --------- | -- | ------------- |
| Python 基础 | ✅  | 掌握变量、流程控制、函数等 |
| Git 命令    | ✅  | 熟悉基本的版本控制操作   |
| Anaconda  | ✅  | 安装成功，创建了虚拟环境  |
| PyTorch   | ✅  | 已成功安装并验证环境    |
|图像数据处理入门|✅|遥感图像数据处理的入门理解和实战代码实现|



