"""
YOLOv8 焊点缺陷检测训练 - 单进程模式
======================================
用单进程+CPU加载模型，避免多进程DLL加载问题
"""
import os
# 完全禁用多进程
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

from ultralytics import YOLO
from pathlib import Path

DATASET_YAML = str(Path(__file__).parent / 'output' / 'dataset' / 'data.yaml')
OUTPUT_DIR = str(Path(__file__).parent / 'output' / 'runs')
EPOCHS = 50

def main():
    print("=" * 60)
    print("焊点缺陷检测 - YOLOv8n CPU训练")
    print("=" * 60)
    print(f"数据集: {DATASET_YAML}")
    print(f"训练轮数: {EPOCHS}")
    print("设备: cpu (GPU因页面文件限制暂不可用)")
    print()
    
    model = YOLO('yolov8n.pt')
    
    print("开始训练...")
    print("=" * 60)
    
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        batch=8,
        imgsz=640,
        device='cpu',
        project=OUTPUT_DIR,
        name='weld_detection',
        exist_ok=True,
        patience=10,
        lr0=0.001,
        augment=True,
        mosaic=0.5,
        workers=0,      # 必须=0，否则多进程DLL问题
        verbose=True,
        cache=False,
    )
    
    print("\n✅ 训练完成！")
    print(f"\n最佳模型: {OUTPUT_DIR}/weld_detection/weights/best.pt")

if __name__ == '__main__':
    main()
