"""
YOLOv8 焊点缺陷检测训练（GPU 优化版）
======================================
"""
from ultralytics import YOLO
import torch
import os
from pathlib import Path

# 设置内存分配策略 - 减少碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 多进程使用 spawn 模式避免 DLL 加载问题
os.environ['PYTHON_MULTIPROCESSING'] = 'spawn'

DATASET_YAML = str(Path(__file__).parent / 'output' / 'dataset' / 'data.yaml')
OUTPUT_DIR = str(Path(__file__).parent / 'output' / 'runs')
EPOCHS = 50

def main():
    print("=" * 60)
    print("焊点缺陷检测 - YOLOv8n GPU训练")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  GPU: {gpu_name} ({mem/1e9:.0f}GB VRAM)")
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    print(f"设备: {device}")
    print(f"数据集: {DATASET_YAML}")
    print(f"训练轮数: {EPOCHS}, 批次: 4")
    print()
    
    model = YOLO('yolov8n.pt')
    
    print("开始训练...")
    print("=" * 60)
    
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        batch=4,               # 显存只有6GB，batch=4安全
        imgsz=640,
        device=device,
        project=OUTPUT_DIR,
        name='weld_detection',
        exist_ok=True,
        patience=10,
        lr0=0.001,
        augment=True,
        mosaic=0.5,
        mixup=0.0,             # 关掉mixup省显存
        workers=0,             # 用0避免多进程 DLL 问题
        verbose=True,
        cache=False,
        amp=True,              # 混合精度
    )
    
    print("\n✅ 训练完成！")
    
    # 用单GPU batch=1验证，避免OOM
    model.val(
        data=DATASET_YAML,
        batch=1,
        device=device,
        project=OUTPUT_DIR,
        name='weld_detection',
        exist_ok=True,
    )

if __name__ == '__main__':
    main()
