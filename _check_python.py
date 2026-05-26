"""查看Python进程详情"""
import subprocess
result = subprocess.run(
    'powershell "Get-Process python | Select-Object Id, @{N=\'MB\';E={[math]::Round($_.WorkingSet64/1MB,1)}}, CommandLine | Format-Table -AutoSize -Wrap"',
    capture_output=True, text=True, shell=True
)
print(result.stdout)
