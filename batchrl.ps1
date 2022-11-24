param([Int32]$num_processes=12)

for($i = 0; $i -lt $num_processes; $i++)
{
    start powershell {python run_sb3.py ; Read-Host}
    #invoke-expression 'cmd /c start powershell {echo python run_sb3.py; Read-Host}'
}

Write-Host "Started $i processes"

Pause

 