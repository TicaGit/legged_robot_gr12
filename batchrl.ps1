param([Int32]$num_processes=4)
$data = @(0.05, 0.01, 0.1, 0.5, 1, 5, 10, 50, 0.02, 0.2, 2, 20, 100, 0.005, 0.002)


for($i = 0; $i -lt $num_processes; $i++)
{
    $argument = $data[$i]

    Start-Process powershell -ArgumentList "python run_sb3.py --y_offset_weight $argument"
    #invoke-expression 'cmd /c start powershell {echo "python run_sb3.py"; Read-Host}'
    Write-Host "Parameter: " $data[$i]
    Start-Sleep -Milliseconds 500
}

Write-Host "Started $i processes"

Pause

 