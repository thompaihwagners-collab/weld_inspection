<<<<<<< HEAD
# 基于AI视觉的汽车白车身焊点质量检测系统

## 项目概述
基于 YOLOv8 的焊点缺陷检测系统，支持5类焊点检测：
- ✅ **good** — 合格焊点
- ❄️ **cold_weld** — 冷焊（未熔合）
- 🔥 **overburn** — 过烧
- 💥 **crack** — 裂纹
- 🕳️ **shrinkage** — 缩孔

## 项目结构
```
├── 01_generate_data.py   # 合成数据集生成器
├── 02_train.py           # 模型训练脚本
├── 03_evaluate.py        # 评估脚本（混淆矩阵、分类报告）
├── 04_demo.py            # Streamlit 交互式演示系统
├── requirements.txt      # 依赖包列表
├── README.md             # 本文件
└── output/
    ├── dataset/          # 数据集（本地生成）
    └── runs/             # 训练结果（本地生成）
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 生成数据
```bash
python 01_generate_data.py
```

### 3. 训练模型
```bash
python 02_train.py
```

### 4. 评估结果
```bash
python 03_evaluate.py
```

### 5. 启动演示
```bash
streamlit run 04_demo.py
```

## 在线演示
本项目可部署到 Streamlit Cloud：
1. 把代码推到 GitHub 仓库
2. 打开 https://streamlit.io/cloud
3. 选择仓库 → 部署即可获得公开链接

## 技术栈
- **目标检测**: YOLOv8 (Ultralytics)
- **数据合成**: OpenCV + NumPy
- **演示框架**: Streamlit
- **可视化**: Matplotlib + Seaborn

## 模型性能
在2000张合成数据集上训练50轮后，测试集结果：
- mAP50: 99.5%
- Precision: 99.9%
- Recall: 100%
=======
# weld_inspection
《智能生产与制造服务技术》课程项目 | 第13组————基于AI视觉的汽车白车身焊点质量检测系统
>>>>>>> 9403b186872cb1c99df1ad85126b7806083fd52e
