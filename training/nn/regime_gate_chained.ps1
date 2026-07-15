# Phase C gate, chained behind the retrain (issue #3):
# collector (regime_r1e) -> trainer (regime_train_chained.ps1) -> this.
# Waits for a FRESH "saved best" in regime_train.log (written AFTER this
# watcher started — yesterday's log already contains the phrase), then runs
# the full pre-registered two-part gate. Log: regime_gate.log
$repo = "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
$trainlog = "$repo\training\nn\regime_train.log"
$start = Get-Date
$deadline = $start.AddHours(10)
$fresh = $false
while ((Get-Date) -lt $deadline) {
    if ((Test-Path $trainlog) -and ((Get-Item $trainlog).LastWriteTime -gt $start)) {
        if (Select-String -Path $trainlog -Pattern "saved best" -Quiet) {
            $fresh = $true
            break
        }
    }
    Start-Sleep -Seconds 120
}
if (-not $fresh) {
    "trainer never produced a fresh 'saved best' - aborting gate chain" |
        Out-File "$repo\training\nn\regime_gate.log" -Encoding utf8
    exit 1
}
& python "$repo\training\nn\regime_gate.py" `
    --scenario-pairs 300 --anchor-games 200 `
    *> "$repo\training\nn\regime_gate.log"
