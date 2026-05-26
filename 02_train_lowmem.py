"""
YOLOv8 焊点缺陷检测训练 - 无绘图省内存
======================================
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import gc, psutil
gc.collect()
print(f"可用内存: {psutil.virtual_memory().available/1e9:.1f}GB / {psutil.virtual_memory().total/1e9:.0f}GB")

from ultralytics import YOLO
from pathlib import Path

DATASET_YAML = str(Path(__file__).parent / 'output' / 'dataset' / 'data.yaml')
OUTPUT_DIR = str(Path(__file__).parent / 'output' / 'runs')

def main():
    model = YOLO('yolov8n.pt')
    
    model.train(
        data=DATASET_YAML,
        epochs=15,
        batch=2,
        imgsz=320,
        device='cpu',
        project=OUTPUT_DIR,
        name='weld_detection',
        exist_ok=True,
        patience=5,
        lr0=0.001,
        augment=False,
        mosaic=0.0,
        workers=0,
        verbose=True,
        cache=False,
        plots=False,        # 关掉绘图节省内存
        deterministic=False,
    )
    
    print("\n✅ 训练完成！")

if __name__ == '__main__':
    main()
