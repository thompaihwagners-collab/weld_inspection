import psutil
for p in psutil.process_iter(['name', 'create_time']):
    try:
        if 'code' in p.info['name'].lower():
            print(f"VS Code running! PID={p.pid}, started at {p.info['create_time']}")
            break
    except:
        pass
else:
    print("VS Code not running")
    
print(f"\nRAM: {psutil.virtual_memory().available/1e9:.2f}GB available")
