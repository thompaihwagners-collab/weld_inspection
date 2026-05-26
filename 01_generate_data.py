"""
焊点缺陷图像合成器
==================
模拟白车身电阻点焊(RSW)焊点的4类缺陷 + 正常焊点
用于验证AI视觉检测原理方案

缺陷类型:
  0 - 正常焊点 (Good)
  1 - 虚焊 (Cold Weld) - 焊核偏小/熔合不足
  2 - 过烧 (Overburn) - 烧穿/喷溅
  3 - 裂纹 (Crack) - 焊点表面/周边裂纹
  4 - 缩孔 (Shrinkage) - 表面缩孔/气孔
"""
import cv2
import numpy as np
import os
import random
import json
from pathlib import Path

# ===================== 配置 =====================
OUTPUT_DIR = Path(__file__).parent / 'output' / 'dataset'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_SAMPLES = 2000  # 总样本数（每类400张）
IMG_SIZE = 640

# 缺陷配置
DEFECT_TYPES = ['good', 'cold_weld', 'overburn', 'crack', 'shrinkage']
DEFECT_LABELS = {name: i for i, name in enumerate(DEFECT_TYPES)}

# 焊点外观参数
WELD_CORE_RADIUS_RANGE = (50, 90)       # 正常焊核半径范围
WELD_NUGGET_COLOR = (80, 100, 120)       # 焊核颜色（BGR，暗灰蓝色）
WELD_HAZE_COLOR = (100, 120, 140)        # 热影响区颜色
STEEL_COLOR = (130, 150, 170)            # 钢板底色
STEEL_TEXTURE_VAR = 15                    # 钢板纹理方差

random.seed(42)
np.random.seed(42)

def create_steel_background(h, w):
    """创建带纹理的钢板背景"""
    base = np.full((h, w, 3), STEEL_COLOR, dtype=np.uint8)
    # 添加随机纹理
    noise = np.random.randint(-STEEL_TEXTURE_VAR, STEEL_TEXTURE_VAR, (h, w, 3), dtype=np.int16)
    textured = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # 添加轻微的划痕/拉丝效果
    for _ in range(random.randint(5, 15)):
        y = random.randint(0, h-1)
        x1, x2 = random.randint(0, w//3), random.randint(w//3*2, w-1)
        cv2.line(textured, (x1, y), (x2, y), 
                 tuple(int(c - random.randint(3, 8)) for c in STEEL_COLOR), 
                 random.randint(1, 2))
    return textured

def draw_weld_nugget(img, cx, cy, radius, defect_type):
    """绘制焊点核心"""
    h, w = img.shape[:2]
    
    # 1. 热影响区 (HAZ) - 最外层
    haz_radius = int(radius * 1.6)
    haz_color = tuple(c + random.randint(-5, 5) for c in WELD_HAZE_COLOR)
    cv2.circle(img, (cx, cy), haz_radius, haz_color, -1)
    
    # 2. 焊核本体
    if defect_type == 'cold_weld':
        # 虚焊 - 焊核偏小、颜色偏暗
        core_r = int(radius * (0.4 + random.uniform(0, 0.15)))
        core_color = tuple(int(c - random.randint(15, 30)) for c in WELD_NUGGET_COLOR)
    elif defect_type == 'overburn':
        # 过烧 - 焊核偏大、边缘烧蚀、颜色发白
        core_r = int(radius * (1.2 + random.uniform(0, 0.2)))
        core_color = tuple(int(min(c + random.randint(10, 25), 255)) for c in WELD_NUGGET_COLOR)
    else:
        core_r = radius
        core_color = tuple(c + random.randint(-5, 5) for c in WELD_NUGGET_COLOR)
    
    cv2.circle(img, (cx, cy), core_r, core_color, -1)
    
    # 3. 焊核表面纹理 - 圆形同心纹
    for r in range(core_r, 0, -max(1, core_r // 8)):
        shade = random.randint(-3, 3)
        circ_color = tuple(int(np.clip(c + shade, 0, 255)) for c in core_color)
        cv2.circle(img, (cx, cy), r, circ_color, 1)
    
    # 4. 焊核中心压痕（电极留下的压痕）
    indent_r = int(core_r * 0.3)
    indent_color = tuple(int(c - 5) for c in core_color)
    cv2.circle(img, (cx, cy), indent_r, indent_color, -1)
    # 压痕中心高光
    highlight = tuple(min(c + 15, 255) for c in indent_color)
    cv2.circle(img, (cx - indent_r//3, cy - indent_r//3), 
               max(1, indent_r // 2), highlight, -1)
    
    return img, core_r, haz_radius

def add_defect_artifacts(img, cx, cy, core_r, defect_type):
    """根据缺陷类型添加特征性伪影"""
    h, w = img.shape[:2]
    
    if defect_type == 'cold_weld':
        # 虚焊特征：熔合边界不连续、有间隙感
        # 在焊核边缘增加暗色断续环
        for angle in np.linspace(0, 2*np.pi, random.randint(8, 16)):
            a = random.uniform(0.7, 0.95) * core_r
            px = int(cx + a * np.cos(angle))
            py = int(cy + a * np.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), 2, (60, 70, 85), -1)
        # 焊核亮度不均匀（熔合不良区）
        inner_r = int(core_r * 0.5)
        x_off, y_off = random.randint(-inner_r//2, inner_r//2), random.randint(-inner_r//2, inner_r//2)
        if 0 <= cx+x_off < w and 0 <= cy+y_off < h:
            cv2.circle(img, (cx+x_off, cy+y_off), int(inner_r*0.6), 
                      (55, 65, 80), -1)
    
    elif defect_type == 'overburn':
        # 过烧特征：烧蚀飞溅、边缘毛刺
        # 飞溅颗粒
        for _ in range(random.randint(10, 25)):
            angle = random.uniform(0, 2*np.pi)
            dist = int(core_r * random.uniform(1.3, 2.5))
            px = int(cx + dist * np.cos(angle))
            py = int(cy + dist * np.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                size = random.randint(2, 5)
                spatter_color = tuple(np.random.randint(180, 255, 3).tolist())
                cv2.circle(img, (px, py), size, spatter_color, -1)
        # 烧蚀边缘 - 不规则白色烧蚀带
        for _ in range(random.randint(3, 6)):
            angle = random.uniform(0, 2*np.pi)
            d = int(core_r * random.uniform(0.8, 1.3))
            px, py = int(cx + d*np.cos(angle)), int(cy + d*np.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), random.randint(3, 8), 
                          (200, 220, 240), -1)
    
    elif defect_type == 'crack':
        # 裂纹特征：从焊核边缘向外辐射的裂纹
        num_cracks = random.randint(2, 5)
        for _ in range(num_cracks):
            angle = random.uniform(0, 2*np.pi)
            length = int(core_r * random.uniform(0.8, 2.0))
            start_x = int(cx + core_r * 0.8 * np.cos(angle))
            start_y = int(cy + core_r * 0.8 * np.sin(angle))
            # 裂纹线——曲折的暗线
            pts = [(start_x, start_y)]
            for step in range(1, 6):
                frac = step / 5
                px = int(start_x + length * frac * np.cos(angle + random.uniform(-0.3, 0.3)))
                py = int(start_y + length * frac * np.sin(angle + random.uniform(-0.3, 0.3)))
                pts.append((px, py))
            for i in range(len(pts)-1):
                if all(0 <= p < max(h, w) for p in pts[i]):
                    if all(0 <= p < max(h, w) for p in pts[i+1]):
                        cv2.line(img, pts[i], pts[i+1], 
                                (40, 45, 50) if i % 2 == 0 else (55, 60, 70), 
                                random.randint(1, 2))
        # 焊核内部裂纹
        inner_angle = random.uniform(0, 2*np.pi)
        inner_len = int(core_r * random.uniform(0.3, 0.7))
        start_x = int(cx + random.uniform(-core_r*0.3, core_r*0.3))
        start_y = int(cy + random.uniform(-core_r*0.3, core_r*0.3))
        cv2.line(img, (start_x, start_y), 
                (int(start_x + inner_len*np.cos(inner_angle)), 
                 int(start_y + inner_len*np.sin(inner_angle))),
                (35, 40, 45), 2)
    
    elif defect_type == 'shrinkage':
        # 缩孔特征：焊核表面凹坑/气孔
        num_holes = random.randint(2, 4)
        for _ in range(num_holes):
            angle = random.uniform(0, 2*np.pi)
            dist = random.uniform(0, core_r * 0.6)
            px = int(cx + dist * np.cos(angle))
            py = int(cy + dist * np.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                hole_r = random.randint(3, max(3, core_r // 4))
                # 缩孔 - 暗色圆形凹陷
                cv2.circle(img, (px, py), hole_r, 
                          (45, 52, 60), -1)
                # 缩孔边缘的亮边
                cv2.circle(img, (px, py), hole_r, 
                          (110, 125, 140) if random.random() > 0.5 else (70, 82, 95), 
                          1)
        # 少量的小气孔
        for _ in range(random.randint(1, 5)):
            angle = random.uniform(0, 2*np.pi)
            dist = random.uniform(0, core_r * 0.5)
            px = int(cx + dist * np.cos(angle))
            py = int(cy + dist * np.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), 1, (55, 60, 68), -1)
    
    return img

def add_realistic_noise(img):
    """添加真实拍摄噪声"""
    # 光照不均匀
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    # 模拟环形光效果
    cx_light = w//2 + random.randint(-w//6, w//6)
    cy_light = h//2 + random.randint(-h//6, h//6)
    dist = np.sqrt((X - cx_light)**2 + (Y - cy_light)**2)
    max_dist = np.sqrt((w//2)**2 + (h//2)**2)
    vignette = 1.0 - 0.3 * (dist / max_dist)
    img_float = img.astype(np.float32)
    for c in range(3):
        img_float[:,:,c] *= vignette
    
    # 高斯噪声
    noise = np.random.normal(0, random.randint(2, 5), (h, w, 3))
    img_float += noise
    
    # JPEG压缩伪影（模拟低端工业相机）
    img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(75, 95)]
    _, enc = cv2.imencode('.jpg', img_uint8, encode_param)
    img_uint8 = cv2.imdecode(enc, 1)
    
    return img_uint8

def generate_one_sample(cls_id, img_size=640):
    """生成一张焊点图像"""
    h, w = img_size, img_size
    defect_type = DEFECT_TYPES[cls_id]
    
    # 背景钢板
    img = create_steel_background(h, w)
    
    # 焊点位置（随机偏移中心）
    cx, cy = w//2 + random.randint(-w//8, w//8), h//2 + random.randint(-h//8, h//8)
    base_radius = random.randint(WELD_CORE_RADIUS_RANGE[0], WELD_CORE_RADIUS_RANGE[1])
    
    # 绘制焊点
    img, core_r, haz_r = draw_weld_nugget(img, cx, cy, base_radius, defect_type)
    
    # 添加缺陷特征
    if defect_type != 'good':
        img = add_defect_artifacts(img, cx, cy, core_r, defect_type)
    
    # 添加噪声
    img = add_realistic_noise(img)
    
    # YOLO格式标注: class x_center y_center width height (归一化)
    x_center = cx / w
    y_center = cy / h
    bbox_w = (haz_r * 2) / w
    bbox_h = (haz_r * 2) / h
    label = f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
    
    return img, label, defect_type

def generate_dataset():
    """生成完整数据集"""
    # 清空/创建目录
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            d = OUTPUT_DIR / split / subdir
            d.mkdir(parents=True, exist_ok=True)
            # 清空旧文件
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
    
    # 生成样本
    samples_per_class = NUM_SAMPLES // len(DEFECT_TYPES)
    total = 0
    class_counts = {}
    
    print(f"开始生成 {NUM_SAMPLES} 张焊点图像...")
    print(f"缺陷类型: {', '.join(DEFECT_TYPES)}")
    print(f"每类 {samples_per_class} 张\n")
    
    for cls_id, defect_type in enumerate(DEFECT_TYPES):
        for i in range(1, samples_per_class + 1):
            img, label, dtype = generate_one_sample(cls_id)
            total += 1
            
            # 分配到 train/val/test
            r = random.random()
            if r < TRAIN_RATIO:
                split = 'train'
            elif r < TRAIN_RATIO + VAL_RATIO:
                split = 'val'
            else:
                split = 'test'
            
            img_name = f"{defect_type}_{i:04d}"
            img_path = OUTPUT_DIR / split / 'images' / f"{img_name}.jpg"
            label_path = OUTPUT_DIR / split / 'labels' / f"{img_name}.txt"
            
            cv2.imwrite(str(img_path), img)
            with open(label_path, 'w') as f:
                f.write(label + '\n')
            
            if i % 500 == 0:
                print(f"  [{defect_type}] 已生成 {i}/{samples_per_class}")
        
        class_counts[defect_type] = samples_per_class
    
    # 统计
    print(f"\n✅ 数据集生成完成！总计 {total} 张")
    for split in ['train', 'val', 'test']:
        imgs_dir = OUTPUT_DIR / split / 'images'
        lbls_dir = OUTPUT_DIR / split / 'labels'
        n_img = len(list(imgs_dir.glob('*.jpg')))
        n_lbl = len(list(lbls_dir.glob('*.txt')))
        print(f"  {split}: {n_img} images, {n_lbl} labels")
    
    # 生成 data.yaml
    data_yaml = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': str(OUTPUT_DIR / 'train' / 'images'),
        'val': str(OUTPUT_DIR / 'val' / 'images'),
        'test': str(OUTPUT_DIR / 'test' / 'images'),
        'nc': len(DEFECT_TYPES),
        'names': DEFECT_TYPES
    }
    yaml_path = OUTPUT_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        import yaml
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"\n  data.yaml 已保存到 {yaml_path}")
    
    # 生成示例预览
    preview_dir = OUTPUT_DIR / 'preview'
    preview_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, defect_name in enumerate(DEFECT_TYPES):
        cls_id = idx
        row, col = divmod(idx, 3)
        if row >= 2:
            break
        # 找一张示例
        for f in (OUTPUT_DIR / 'train' / 'images').glob(f"{defect_name}_0001.jpg"):
            ax = axes[row, col]
            img = cv2.imread(str(f))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"{cls_id}: {defect_name}", fontsize=10)
            ax.axis('off')
    
    for idx in range(len(DEFECT_TYPES), 6):
        row, col = divmod(idx, 3)
        if row < 2:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    preview_path = preview_dir / 'dataset_preview.png'
    plt.savefig(str(preview_path), dpi=150)
    print(f"\n  预览图: {preview_path}")
    plt.close()
    
    return class_counts

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import yaml
    
    generate_dataset()
