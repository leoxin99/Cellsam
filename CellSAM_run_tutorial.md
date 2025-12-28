# Cell-SAM 个人电脑部署与运行教程

本教程旨在指导如何在个人电脑（Windows/Linux/macOS）上部署并运行 Cell-SAM（一种用于细胞分割的通用基础模型）。

## 1. 系统要求 (Prerequisites)

在开始之前，请确保你的电脑满足以下要求：

*   **操作系统**: Windows 10/11, Linux, 或 macOS。
*   **Python**: 版本 **>= 3.10**。
*   **硬件**: 推荐使用 NVIDIA 显卡 (GPU) 以获得更快的推理速度。如果没有显卡，也可以使用 CPU，但速度会较慢。
*   **Git**: 需要安装 [Git](https://git-scm.com/) 以从 GitHub 下载代码。

## 2. 环境配置 (Environment Setup)

推荐使用 **Conda** 来管理 Python 环境，以避免依赖冲突。

### 2.1 安装 Anaconda 或 Miniconda
如果你还没有安装 Conda，请前往 [Miniconda官网](https://docs.conda.io/en/latest/miniconda.html) 下载并安装适合你系统的版本。

### 2.2 创建虚拟环境
打开终端（Windows 上是 Anaconda Prompt 或 PowerShell），运行以下命令创建一个名为 `cellsam` 的环境：

```bash
conda create -n cellsam python=3.10 -y
conda activate cellsam
```

### 2.3 安装 PyTorch
**关键步骤**：请根据你的硬件情况安装合适的 PyTorch 版本。

*   **NVIDIA 显卡用户 (推荐)**:
    前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 查找适合你 CUDA 版本的安装命令。通常如下（以 CUDA 11.8 为例）：
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

*   **无显卡/CPU 用户**:
    ```bash
    pip install torch torchvision torchaudio
    ```

## 3. 安装 Cell-SAM

Cell-SAM 提供了两种安装方式：基础版（仅代码）和带图形界面版（Napari 插件）。**对于个人用户，强烈推荐安装带图形界面的 Napari 版本。**

### 选项 A：安装带图形界面 (Napari UI) - 推荐
Napari 是一个强大的多维图像查看器，Cell-SAM 提供了插件支持，可以通过点击操作进行分割。

```bash
pip install "cellSAM[napari] @ git+https://github.com/vanvalenlab/cellSAM@master"
```

*注意：如果遇到报错提示缺少 git，请确保已安装 git 并添加到环境变量。*

### 选项 B：仅安装核心库 (Python API)
如果你是开发者，只需要在 Python 脚本中调用，可以使用此命令：

```bash
pip install git+https://github.com/vanvalenlab/cellSAM.git
```

## 4. 运行 Cell-SAM

### 方法一：使用图形界面 (GUI)

安装了 Napari 版本后，可以直接通过命令行启动：

```bash
cellsam napari
```

**操作步骤**:
1.  启动后会打开一个 Napari 窗口。
2.  将你的细胞图像拖入窗口。
3.  在右侧菜单中找到 Cell-SAM 插件。
4.  点击运行（Run）或相关按钮即可自动分割图像。

### 方法二：使用 Python 脚本

你可以编写一个简单的 Python 脚本来批量处理图像。

创建一个名为 `run_cellsam.py` 的文件，填入以下代码：

```python
import numpy as np
from cellSAM import segment_cellular_image
import matplotlib.pyplot as plt
import cv2  # 如果没有安装 cv2，请运行 pip install opencv-python

# 1. 加载图像 (替换为你自己的图片路径)
# 支持 .jpg, .png, .tif 等格式
image_path = "path/to/your/image.tif" 
img = cv2.imread(image_path)
if img is None:
    print(f"无法读取图像: {image_path}")
    exit()

# Cell-SAM 期望图像格式为 (H, W, C) 或 (H, W)
# 确保如果是 RGB 图像，通道序正确
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("正在进行分割，请稍候...")

# 2. 运行分割
# device='cuda' 如果你有显卡，否则使用 'cpu'
device = 'cuda' 
# device = 'cpu'

# mask 是分割结果，embedding 是图像的 embedding
mask, embedding, _ = segment_cellular_image(img, device=device)

# 3. 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(mask, cmap='nipy_spectral') # 使用彩色显示不同的细胞ID
plt.axis('off')

plt.tight_layout()
plt.show()

print("分割完成！")
```

运行脚本：
```bash
python run_cellsam.py
```

## 5. 常见问题 (Troubleshooting)

1.  **CUDA Out of Memory**: 如果显存不足，尝试在调用 `segment_cellular_image` 时减小图像尺寸，或者在 CPU 上运行（虽然会很慢）。
2.  **安装失败**: 
    *   确保 `pip` 是最新的：`pip install --upgrade pip`
    *   Windows 用户如果遇到编译错误，可能需要安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 中的 "Desktop development with C++"。

## 6. 资源

*   **GitHub**: [https://github.com/vanvalenlab/cellSAM](https://github.com/vanvalenlab/cellSAM)
*   **官方教程**: [https://vanvalenlab.github.io/cellSAM/tutorial](https://vanvalenlab.github.io/cellSAM/tutorial)
