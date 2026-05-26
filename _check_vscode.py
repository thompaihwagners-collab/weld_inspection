import subprocess
r = subprocess.run('tasklist /FI "IMAGENAME eq Code.exe"', capture_output=True, text=True, shell=True, timeout=5)
print(r.stdout)
if 'Code.exe' in r.stdout:
    print(">>> VS Code IS RUNNING <<<")
else:
    print(">>> VS Code NOT RUNNING <<<")
