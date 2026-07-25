# ACAN post-collection pipeline (S3) — waits for the corpus, then runs the
# pre-registered CONTROLLED loss ablation on identical data/seed.
#
# The ablation matters: an uncontrolled first look suggested the listwise
# ranking loss HURTS (loose precision 0.330 -> 0.140, training MSE rising),
# with a plausible mechanism -- the listwise target is the search's overall
# argmax, which IS the heuristic's pick ~45% of the time, so ranking largely
# teaches conformity rather than override discrimination. But those two runs
# saw different corpus sizes, so it is not yet a result. This settles it.

$ErrorActionPreference = "Stop"
$nn = $PSScriptRoot
$corpus = Join-Path $nn "acan_corpus"

# wait for all 16 workers to write their outcomes file
$deadline = (Get-Date).AddHours(4)
while ((Get-ChildItem "$corpus\*.outcomes.json" -ErrorAction SilentlyContinue).Count -lt 16) {
    if ((Get-Date) -gt $deadline) { Write-Output "TIMEOUT waiting for collection"; break }
    Start-Sleep -Seconds 60
}
$recs = (Get-ChildItem "$corpus\*.jsonl" | Get-Content | Measure-Object -Line).Lines
Write-Output "collection complete: $recs records"

# ablation arm A: MSE only (magnitude, no ranking pressure)
Write-Output "`n=== ARM A: MSE-only (--rank-weight 0) ==="
python "$nn\train_acan.py" --epochs 25 --rank-weight 0 `
    --out "$nn\acan_mse.pth" 2>&1 | Tee-Object "$nn\acan_mse.log"

# ablation arm B: MSE + listwise ranking
Write-Output "`n=== ARM B: MSE + listwise ranking (--rank-weight 1) ==="
python "$nn\train_acan.py" --epochs 25 --rank-weight 1 `
    --out "$nn\acan_rank.pth" 2>&1 | Tee-Object "$nn\acan_rank.log"

Write-Output "`nDONE -- compare the two GATE tables (loose_p and exact precision)."
