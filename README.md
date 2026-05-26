# 基于AI视觉的汽车白车身焊点质量检测系统

《智能生产与制造服务技术》课程项目 | 第13组

## 项目结构

```
F:\ai-tools\weld_inspection\
├── 00_check_env.py               # 环境依赖检查
├── 01_generate_data.py           # 焊点缺陷图像合成 (2000张)
├── 02_train.py                   # YOLOv8 模型训练
├── 03_evaluate.py                # 模型评估与可视化
├── 04_demo.py                    # Streamlit 交互式演示系统
├── check_env.py                  # 环境检测脚本
├── README.md                     # 本文件
│
├── output/                       # 所有输出
│   ├── dataset/                  # 合成数据集
│   │   ├── train/                # 训练集 (1407张)
│   │   │   ├── images/
│   │   │   └── labels/           # YOLO格式标注
│   │   ├── val/                  # 验证集 (292张)
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── test/                 # 测试集 (301张)
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── data.yaml             # YOLO数据集配置
│   │   └── preview/              # 数据集预览
│   │
│   ├── runs/                     # 训练结果
│   │   └── weld_detection/
│   │       ├── weights/
│   │       │   ├── best.pt       # 最佳模型
│   │       │   └── last.pt       # 最后一轮模型
│   │       ├── confusion_matrix.png
│   │       ├── results.csv
│   │       └── ...
│   │
│   └── figures/                  # 报告用图
│       ├── confusion_matrix.png  # 混淆矩阵
│       ├── detection_examples.png # 检测示例
│       ├── evaluation_summary.png # 评估摘要
│       └── evaluation_metrics.json # 指标数据
```

## 快速使用

### 1. 环境准备

```bash
cd F:\ai-tools\weld_inspection
py 00_check_env.py   # 检查依赖
```

如果缺少依赖：
```bash
pip install ultralytics opencv-python streamlit pyyaml
```

### 2. 生成数据集

```bash
py 01_generate_data.py
```
生成 2000 张焊点图像（5类×400张），自动划分 train/val/test。

**缺陷类型:**
| 标签 | 类型 | 说明 |
|------|------|------|
| 0 | good | 正常焊点 |
| 1 | cold_weld | 虚焊（焊核偏小） |
| 2 | overburn | 过烧（烧穿/飞溅） |
| 3 | crack | 裂纹 |
| 4 | shrinkage | 缩孔（气孔/凹陷） |

### 3. 训练模型

```bash
py 02_train.py
```
训练 YOLOv8n 模型（CPU约5-6小时完成50轮）。

**参数:** epochs=50, batch=8, imgsz=640
**硬件:** CPU模式（有GPU自动使用GPU）

### 4. 评估模型

```bash
py 03_evaluate.py
```
生成混淆矩阵、检测示例图、分类指标报告。

### 5. 启动演示系统

```bash
streamlit run 04_demo.py
```
浏览器打开 http://localhost:8501

**演示系统功能:**
- 焊点检测演示（单张/批量）
- 检测结果分析（置信度分布）
- 模型评估报告（指标、混淆矩阵）
- 缺陷类型详解

## 技术方案

### 图像合成原理
- 钢板纹理背景（高斯噪声 + 拉丝效果）
- 焊点核心（同心圆 + 电极压痕 + 表面高光）
- 缺陷特征（裂纹、缩孔、飞溅、熔合不良）
- 真实模拟（光照不均、JPEG压缩、镜头噪声）

### 检测模型
- **YOLOv8n** - 301万参数，轻量级实时检测
- 640×640 输入，端到端检测
- 支持 Mosaic/Mixup 数据增强
- AdamW 优化器，CIoU Loss

### 检测流程
```
图像采集 → 预处理 → YOLOv8推理 → 质量判定 → 数据记录
```

## 结课提交说明

本项目包含完整的原理验证系统，可根据需要运行生成以下结课材料：

1. **合成数据集** - output/dataset/ (2000张焊点图像)
2. **训练模型** - output/runs/weld_detection/weights/best.pt
3. **评估图表** - output/figures/ (混淆矩阵、检测示例、评估摘要)
4. **演示系统** - Streamlit交互界面
5. **实验数据** - output/figures/evaluation_metrics.json

## 扩展建议

- **真实数据**：替换为产线实际焊点图像进行微调
- **模型升级**：换用 YOLOv8m/l/x 提升精度
- **边缘部署**：导出 ONNX/TensorRT 部署到 Jetson
- **多模态**：结合超声波检测信号
- **数字孪生**：与 MES 系统对接实现质量追溯
