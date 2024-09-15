
# YOLOv5-Enhancements

本仓库基于 YOLOv5，添加了多种增强模型，如 MobileNetV3、BiFPN、C3STR、DIOU 等。这些增强主要集中在改进模型结构和性能，并保留 YOLOv5 的基本工作流程。

## 概述

`models/enhancements` 目录中包含了多个 YOLOv5 的改进模型，旨在提升模型的推理速度、检测精度，以及对复杂场景的适应性。用户可以根据需求选择不同的模型架构。

### 主要特点

- **新增 MobileNetV3 变体**：通过将 MobileNetV3 集成到 YOLOv5 中，实现更轻量级的模型。
- **BiFPN 加强的特征融合**：部分模型集成了 BiFPN，提升了特征融合能力。
- **C3STR 改进模型**：引入了 C3STR 架构，以增强模型的特征提取能力。
- **DIOU 损失函数**：集成了 DIOU 损失函数，提升了边框回归的效果。

## 文件结构

```
YOLOv5-Enhancements/
│
├── models/                       # 核心 YOLOv5 模型和增强部分
│   ├── enhancements/             # 增强和自定义模型
│   │   ├── yolov5s.yaml          # YOLOv5s 基础模型
│   │   ├── yolov5s-diou.yaml     # YOLOv5s + DIOU 损失函数
│   │   ├── yolov5s-mobilenetv3.yaml # YOLOv5s + MobileNetV3 架构
│   │   ├── yolov5s-mobilenetv3-bifpn.yaml # YOLOv5s + MobileNetV3 + BiFPN
│   │   ├── yolov5s-mobilenetv3-bifpn-C3STR.yaml # YOLOv5s + MobileNetV3 + BiFPN + C3STR
│   ├── hub/        
│   ├── shufflenet/               # ShuffleNet 模型相关文件
│   │   ├── CFNet.py              # 自定义特征网络
│   │   ├── MobileNetV3.py        # MobileNetV3 集成
│   │   ├── Shuffle.py            # ShuffleNet 集成
│   │   ├── SwinTransformer.py    # SwinTransformer 集成
│   │   └── ...                   # 其他增强模型
│   ├── yolov5l.yaml             
│   ├── yolov5m.yaml             
│   ├── yolov5n.yaml             
│   ├── yolov5s.yaml             
│   ├── yolov5x.yaml              
│   └── ...                    
│
├── detect.py                     # 推理脚本
├── export.py                     # 模型导出脚本
├── hubconf.py                    # Hub 模型配置文件
├── train.py                      # 模型训练脚本
├── val.py                        # 模型验证脚本
├── requirements.txt              # Python 依赖包文件
└── README.md                     # 本 README 文件
```

## 安装

1. **克隆仓库**：
    ```bash
    git clone https://github.com/your-username/YOLOv5-Enhancements.git
    cd YOLOv5-Enhancements
    ```

2. **安装依赖**：
    本项目使用与 YOLOv5 相同的依赖项，您可以通过以下命令安装：
    ```bash
    pip install -r requirements.txt
    ```

## 使用说明

可以像使用原始 YOLOv5 一样使用此仓库。以下是一些使用示例：

### 训练

使用增强配置文件训练模型，例如要训练 `yolov5s-mobilenetv3-bifpn-C3STR.yaml` 模型，可以运行：

```bash
python train.py --img 640 --batch 16 --epochs 100 --data data/coco.yaml --cfg models/enhancements/yolov5s-mobilenetv3-bifpn-C3STR.yaml --weights '' --name yolov5s_mobilenetv3_bifpn_C3STR
```

### 推理

使用自定义模型进行推理，例如：

```bash
python detect.py --weights path/to/your/custom/weights.pt --img 640 --conf 0.25 --source path/to/your/test/images
```

## 增强模型

在 `models/enhancements` 文件夹中，你可以找到多种增强版的 YOLOv5 模型：

- **YOLOv5s-DIOU**：使用 DIOU 损失函数，优化边框回归效果。
- **YOLOv5s-MobileNetV3**：通过引入 MobileNetV3，减小模型大小，提升推理速度。
- **YOLOv5s-MobileNetV3-BiFPN**：结合了 MobileNetV3 和 BiFPN，进一步优化特征融合。
- **YOLOv5s-MobileNetV3-BiFPN-C3STR**：集成了 MobileNetV3、BiFPN 和 C3STR 架构，最大化性能提升。

## 贡献

欢迎提交您的贡献！如果您有新的模型增强或优化思路，欢迎 Fork 本仓库并提交 Pull Request。遇到任何问题，您可以在 [issues 页面](https://github.com/your-username/YOLOv5-Enhancements/issues) 提交问题。

