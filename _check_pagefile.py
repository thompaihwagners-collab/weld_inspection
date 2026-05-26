import subprocess
# 获取页面文件大小
result = subprocess.run(
    'powershell -Command "Get-CimInstance Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize | Format-Table -AutoSize"',
    capture_output=True, text=True, shell=True
)
print("页面文件设置:")
print(result.stdout)

result2 = subprocess.run(
    'powershell -Command "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory, TotalVirtualMemory"',
    capture_output=True, text=True, shell=True
)
print("\n系统内存信息:")
print(result2.stdout)

import psutil
mem = psutil.virtual_memory()
print(f"物理内存: {mem.total/1e9:.1f}GB")
print(f"可用内存: {mem.available/1e9:.1f}GB")

swap = psutil.swap_memory()
print(f"\n页面文件总计: {swap.total/1e9:.1f}GB")
print(f"已使用: {swap.used/1e9:.1f}GB")
print(f"空闲: {swap.free/1e9:.1f}GB")
