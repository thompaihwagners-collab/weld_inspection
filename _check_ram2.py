"""检查到底还有多少真实可用内存"""
import psutil

mem = psutil.virtual_memory()
print(f"Total:  {mem.total/1e9:.1f}GB")
print(f"Available: {mem.available/1e9:.1f}GB")
print(f"Used:   {mem.used/1e9:.1f}GB")
print(f"Free:   {mem.free/1e9:.1f}GB")
print(f"Percent: {mem.percent}%")
print()
print(f"Available for use: {mem.available/1e6:.1f}MB")

# 用numpy试试能不能分配
import numpy as np
try:
    arr = np.zeros((100, 100), dtype=np.float32)
    print(f"Small numpy alloc: OK ({arr.nbytes/1e3:.0f}KB)")
except:
    print("Small numpy alloc: FAILED")
    
try:
    arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
    print(f"3MB numpy alloc: OK ({arr.nbytes/1e6:.1f}MB)")
except:
    print("3MB numpy alloc: FAILED")

try:
    arr = np.zeros((3000, 3000, 3), dtype=np.uint8)
    print(f"27MB numpy alloc: OK ({arr.nbytes/1e6:.1f}MB)")
except:
    print("27MB numpy alloc: FAILED")
