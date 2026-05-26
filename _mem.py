import psutil
m = psutil.virtual_memory()
print(f"Total: {m.total/1e9:.1f}GB, Available: {m.available/1e9:.1f}GB, Used: {m.percent}%")
