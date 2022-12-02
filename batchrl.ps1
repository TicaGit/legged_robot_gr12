param([Int32]$num_processes=12)
$data = @(10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001)


for($i = 0; $i -lt $num_processes; $i++)
{
    $argument = $data[$i]

    Start-Process powershell -ArgumentList "python run_sb3.py $argument"
    #invoke-expression 'cmd /c start powershell {echo "python run_sb3.py"; Read-Host}'
    Write-Host "Learning Rate of: " $data[$i]
}

Write-Host "Started $i processes"

Pause

 