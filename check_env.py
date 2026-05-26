"""检查环境依赖"""
import subprocess, sys, pkg_resources

required = {
    'torch': 'PyTorch',
    'ultralytics': 'YOLOv8',
    'opencv-python': 'OpenCV',
    'matplotlib': 'matplotlib',
    'numpy': 'numpy',
    'streamlit': 'Streamlit',
    'Pillow': 'Pillow',
    'scikit-learn': 'scikit-learn'
}

missing = []
for pkg, name in required.items():
    try:
        v = pkg_resources.get_distribution(pkg).version
        print(f'  ✅ {name} ({pkg}): v{v}')
    except:
        missing.append(pkg)
        print(f'  ❌ {name} ({pkg}): 未安装')

if missing:
    print(f'\n缺少 {len(missing)} 个依赖:')
    print(f'pip install {" ".join(missing)}')
else:
    print('\n✅ 所有依赖就绪！')

# CUDA check
try:
    import torch
    print(f'  CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
except:
    pass
