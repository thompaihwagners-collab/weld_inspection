"""
焊点缺陷检测评估工具
====================
生成混淆矩阵、分类报告、检测结果可视化
"""
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import Counter
import json

# === 配置 ===
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output' / 'figures'
DATASET_DIR = BASE_DIR / 'output' / 'dataset'
MODEL_PATH = BASE_DIR / 'output' / 'runs' / 'weld_detection' / 'weights' / 'best.pt'

DEFECT_NAMES = ['Good (正常)', 'Cold Weld (虚焊)', 'Overburn (过烧)', 
                'Crack (裂纹)', 'Shrinkage (缩孔)']
DEFECT_SHORT = ['Good', 'ColdWeld', 'Overburn', 'Crack', 'Shrinkage']


def load_model():
    """加载训练好的模型"""
    if not MODEL_PATH.exists():
        print(f"⚠ 模型不存在: {MODEL_PATH}")
        print("  请先运行 02_train.py 完成训练")
        return None
    print(f"加载模型: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def evaluate_test_set(model):
    """在测试集上评估"""
    test_img_dir = DATASET_DIR / 'test' / 'images'
    test_lbl_dir = DATASET_DIR / 'test' / 'labels'
    
    img_files = sorted(test_img_dir.glob('*.jpg'))
    print(f"测试集: {len(img_files)} 张")
    
    # 逐张预测
    y_true = []
    y_pred = []
    y_pred_conf = []
    
    for img_path in img_files:
        # 真实标签
        label_path = test_lbl_dir / (img_path.stem + '.txt')
        if label_path.exists():
            with open(label_path) as f:
                cls_id = int(f.read().strip().split()[0])
            y_true.append(cls_id)
        else:
            continue
        
        # 预测
        results = model(str(img_path), imgsz=640, conf=0.25, verbose=False)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # 取置信度最高的检测
            top_idx = boxes.conf.argmax().item()
            pred_cls = int(boxes.cls[top_idx].item())
            pred_conf = boxes.conf[top_idx].item()
            y_pred.append(pred_cls)
            y_pred_conf.append(pred_conf)
        else:
            y_pred.append(-1)  # 未检测到
            y_pred_conf.append(0)
    
    return y_true, y_pred, y_pred_conf


def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    n_classes = len(DEFECT_NAMES)
    
    # 过滤掉-1（未检测到）
    valid_indices = [i for i, p in enumerate(y_pred) if p >= 0]
    y_true_f = [y_true[i] for i in valid_indices]
    y_pred_f = [y_pred[i] for i in valid_indices]
    
    cm = confusion_matrix(y_true_f, y_pred_f, labels=range(n_classes))
    
    # 计算百分比
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=DEFECT_SHORT, yticklabels=DEFECT_SHORT,
                ax=ax, vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Welding Spot Defect Detection - Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(str(path), dpi=200)
    print(f"  混淆矩阵: {path}")
    plt.close()
    
    return cm, cm_normalized


def plot_detection_examples(model, n_samples=12):
    """绘制检测结果示例"""
    test_dir = DATASET_DIR / 'test' / 'images'
    img_files = sorted(test_dir.glob('*.jpg'))
    random_indices = np.random.choice(len(img_files), min(n_samples, len(img_files)), replace=False)
    
    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for idx, img_idx in enumerate(random_indices):
        img_path = img_files[img_idx]
        
        # 原始图像
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 预测
        results = model(str(img_path), imgsz=640, conf=0.25, verbose=False)
        
        # 绘制结果
        result_img = results[0].plot(line_width=2, font_size=8)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(result_img_rgb)
        axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('YOLOv8 Weld Defect Detection - Test Results', fontsize=14, y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / 'detection_examples.png'
    plt.savefig(str(path), dpi=200, bbox_inches='tight')
    print(f"  检测示例: {path}")
    plt.close()


def compute_detailed_metrics(y_true, y_pred, y_pred_conf):
    """计算详细指标"""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # 过滤无效预测
    valid = [i for i, p in enumerate(y_pred) if p >= 0]
    detected = len(valid)
    missed = len(y_pred) - detected
    
    y_true_f = [y_true[i] for i in valid]
    y_pred_f = [y_pred[i] for i in valid]
    confs = [y_pred_conf[i] for i in valid]
    
    # 各类指标
    accuracy = accuracy_score(y_true_f, y_pred_f)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_f, y_pred_f, labels=range(len(DEFECT_NAMES)), zero_division=0
    )
    
    print("\n" + "=" * 60)
    print("DETAILED EVALUATION REPORT")
    print("=" * 60)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 66)
    
    metrics_by_class = {}
    for i in range(len(DEFECT_NAMES)):
        metrics_by_class[DEFECT_SHORT[i]] = {
            'precision': round(precision[i], 4),
            'recall': round(recall[i], 4),
            'f1': round(f1[i], 4),
            'support': int(support[i]),
        }
        print(f"{DEFECT_NAMES[i]:<20} {precision[i]:.4f}      {recall[i]:.4f}      {f1[i]:.4f}      {int(support[i]):<10}")
    
    print("-" * 66)
    print(f"{'Overall':<20} {'':<12} {'':<12} {accuracy:.4f}      {detected:<10}")
    print(f"{'未检测到':<20} {'':<12} {'':<12} {'':<12} {missed:<10}")
    print(f"平均置信度: {np.mean(confs):.4f}")
    
    # 汇总指标
    summary = {
        'accuracy': round(accuracy, 4),
        'detected': detected,
        'missed': missed,
        'total': len(y_pred),
        'avg_confidence': round(float(np.mean(confs)), 4),
        'per_class': metrics_by_class,
    }
    return summary


def generate_summary_card(summary, cm_normalized):
    """生成总结信息卡"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    # 构建表格文本
    text = "WELD DEFECT DETECTION - 系统评估总结\n"
    text += "=" * 50 + "\n"
    text += f"测试集总样本: {summary['total']}\n"
    text += f"检出样本: {summary['detected']} | 漏检: {summary['missed']}\n"
    text += f"综合准确率: {summary['accuracy']:.2%}\n"
    text += f"平均检测置信度: {summary['avg_confidence']:.2%}\n\n"
    
    text += "分类正确率:\n"
    diag = [cm_normalized[i][i] for i in range(len(DEFECT_SHORT))]
    for i, name in enumerate(DEFECT_SHORT):
        text += f"  {name:<12} {diag[i]:.1f}%\n"
    
    text += "\n指标来源: YOLOv8n on Synthetic Weld Dataset"
    
    ax.text(0.05, 0.5, text, fontsize=11, verticalalignment='center',
            fontfamily='monospace')
    
    path = OUTPUT_DIR / 'evaluation_summary.png'
    plt.savefig(str(path), dpi=200, bbox_inches='tight')
    print(f"  评估摘要: {path}")
    plt.close()


def main():
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = load_model()
    if model is None:
        return
    
    # 评估
    print("\n[1/5] 评估测试集...")
    y_true, y_pred, y_pred_conf = evaluate_test_set(model)
    
    print(f"\n[2/5] 计算详细指标...")
    summary = compute_detailed_metrics(y_true, y_pred, y_pred_conf)
    
    print(f"\n[3/5] 绘制混淆矩阵...")
    cm, cm_norm = plot_confusion_matrix(y_true, y_pred)
    
    print(f"\n[4/5] 绘制检测示例...")
    plot_detection_examples(model, n_samples=8)
    
    print(f"\n[5/5] 生成评估摘要...")
    generate_summary_card(summary, cm_norm)
    
    # 保存指标到JSON
    metrics_path = OUTPUT_DIR / 'evaluation_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  指标JSON: {metrics_path}")
    
    print("\n✅ 评估完成！所有结果在 output/figures/")
    print(f"\n{OUTPUT_DIR}")

if __name__ == '__main__':
    main()
