# Round-2 gate, chained behind the round-2 retrain (issue #3).
# Waits for "saved best" in regime_train_r2.log (fresh file, no staleness
# concern), then runs the unchanged two-part gate against the r2 checkpoint.
$repo = "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
$trainlog = "$repo\training\nn\regime_train_r2.log"
$gatelog = "$repo\training\nn\regime_gate_r2.log"
$deadline = (Get-Date).AddHours(14)
$ready = $false
while ((Get-Date) -lt $deadline) {
    if (Test-Path $trainlog) {
        if (Select-String -Path $trainlog -Pattern "saved best" -Quiet) {
            $ready = $true
            break
        }
        if (Select-String -Path $trainlog -Pattern "ABORT" -Quiet) {
            "ABORT: trainer aborted upstream - gate not run" |
                Out-File $gatelog -Encoding utf8
            exit 1
        }
    }
    Start-Sleep -Seconds 120
}
if (-not $ready) {
    "ABORT: trainer never produced 'saved best' before deadline" |
        Out-File $gatelog -Encoding utf8
    exit 1
}
$env:REGIME_CKPT = "$repo\training\regime_qnet_r2.pth"
& python "$repo\training\nn\regime_gate.py" `
    --scenario-pairs 300 --anchor-games 200 `
    --csv "$repo\training\regime_gate_r2_games.csv" `
    *> $gatelog
