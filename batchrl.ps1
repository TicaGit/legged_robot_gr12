param([Int32]$num_processes=12)
$data = @(0.05, 0.01, 0.02, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)


for($i = 0; $i -lt $num_processes; $i++)
{
    $argument = $data[$i]

    Start-Process powershell -ArgumentList "python run_sb3.py --y_offset_weight $argument"
    #invoke-expression 'cmd /c start powershell {echo "python run_sb3.py"; Read-Host}'
    Write-Host "Learning Rate of: " $data[$i]
}

Write-Host "Started $i processes"

Pause

 