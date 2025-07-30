# Dino AI - L3级自适应性能智能体

这是一个基于深度学习和离线强化学习的AI智能体，旨在精通谷歌Chrome恐龙游戏。项目采用了模块化的DMAA（Detect-Model-Act-Adapt）架构，实现了从感知到决策的端到端智能。

## 核心技术栈

*   **感知层 (Perception)**:
    *   **目标检测**: `YOLOv8` 用于快速定位玩家和障碍物。
    *   **精细分类**: `EfficientNetV2` 用于准确识别障碍物的具体类型。
*   **世界建模层 (World Modeling)**:
    *   **状态跟踪**: `Unscented Kalman Filter (UKF)` 用于平滑、准确地估计游戏速度和障碍物距离。
    *   **状态向量**: 将所有关键信息整合成一个标准化的状态向量，作为决策模型的输入。
*   **决策规划层 (Decision Making)**:
    *   **离线强化学习**: 使用 `d3rlpy` 库和 `DiscreteCQL` 算法，从专家数据中学习一个最优决策策略模型。
*   **核心框架**:
    *   **模型格式**: 所有模型统一使用 `ONNX` 格式进行高效推理。
    *   **环境管理**: 使用 `uv` 和 `requirements.in` 进行现代化、可复现的环境管理。

## 项目结构

项目遵循“高内聚、低耦合、模块化”的设计原则：

-   `run_bot.py`: 项目主入口，负责启动和运行AI智能体。
-   `src/`: 包含所有运行时核心模块（感知、控制、世界建模等）。
-   `training_pipelines/`: 包含了三个独立的、用于训练所有AI模型的“一键式”流水线。
-   `data/`: 存放所有训练所需的数据集。
-   `models/`: 存放所有最终训练好的、用于部署的模型文件。

## 快速开始

**1. 环境搭建**

本项目使用 `uv` 进行环境管理。请确保你已安装 `uv`。

```bash
# 1. 创建并激活虚拟环境
uv venv

# 2. 安装所有依赖
uv pip install -r requirements.in
```

**2. 模型训练 (流水线)**

你需要按顺序运行训练流水线来生成所有必要的模型。

```bash
# 假设你已经准备好了YOLO和分类器的标注数据

# 流水线 1: 训练目标检测模型
python training_pipelines/1_detection_pipeline/train_detector.py

# 流水线 2: 训练图像分类模型

# 合成数据
python training_pipelines/2_classification_pipeline/1_generate_dataset.py

# 训练分类器
python training_pipelines/2_classification_pipeline/2_train_classifier.py

# 流水线 3: 训练决策策略模型

# 采集你自己的专家数据
python training_pipelines/3_policy_pipeline/1_collect_data.py

# 将采集的数据处理成IQL格式
python training_pipelines/3_policy_pipeline/2_process_data.py

# 训练并导出最终的ONNX决策模型
python training_pipelines/3_policy_pipeline/3_train_and_export.py
```

**3. 运行AI**

当所有模型都已成功生成并放置在 `models/` 对应的子目录下后，即可启动AI。

```bash
python run_bot.py
```

## 未来的工作

*   [ ] 在README中添加AI运行的动态GIF演示图。
*   [ ] 对超参数进行进一步调优，冲击更高的分数。
*   [ ] 探索L4级别的混合驱动安全框架。


