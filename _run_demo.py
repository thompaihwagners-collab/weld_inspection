import subprocess, psutil, time

# Check if streamlit already running
for p in psutil.process_iter(['name','cmdline','pid']):
    try:
        cmd = ' '.join(p.info['cmdline'] or [])
        if 'streamlit' in cmd.lower() and '04_demo' in cmd:
            print(f"ALREADY RUNNING PID={p.pid}")
            break
    except: pass
else:
    # Start it fresh with stdin pipe
    proc = subprocess.Popen(
        ['python', '-m', 'streamlit', 'run', 'F:\\ai-tools\\weld_inspection\\04_demo.py',
         '--browser.gatherUsageStats=false'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd='F:\\ai-tools\\weld_inspection', shell=True, text=True
    )
    print(f"Started streamlit PID={proc.pid}, sending newline...")
    proc.stdin.write('\n')
    proc.stdin.flush()
    time.sleep(10)
    
    # Read first output lines
    import select
    print("Output so far:")
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"  {line.rstrip()}")
        if 'You can now view' in line or 'Network URL' in line or 'Local URL' in line:
            break

print("Done checking")
