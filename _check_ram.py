"""查看内存占用最多的进程"""
import subprocess, json

result = subprocess.run(
    'powershell "Get-Process | Sort-Object -Property WorkingSet64 -Descending | Select-Object -First 20 Name, @{N=\'MB\';E={[math]::Round($_.WorkingSet64/1MB,1)}} | Format-Table -AutoSize"',
    capture_output=True, text=True, shell=True
)
print(result.stdout)
