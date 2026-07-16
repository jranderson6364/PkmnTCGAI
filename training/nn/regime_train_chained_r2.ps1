# Round-2 training, chained behind the round-2 collection (issue #3).
# NEW vs round 1: aborts unless the collection's TOTAL win rate is >= 3%
# (the pre-registered contrast precondition — round 1 trained on a corpus
# with ZERO wins and only found out at the gate).
$repo = "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
$collectlog = "$repo\training\nn\regime_r2_collect.log"
$trainlog = "$repo\training\nn\regime_train_r2.log"
$deadline = (Get-Date).AddHours(10)
while ((Get-Date) -lt $deadline) {
    $collectors = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
        Where-Object { $_.CommandLine -like "*regime_collect*" }
    if (-not $collectors) { break }
    Start-Sleep -Seconds 60
}
$total = Select-String -Path $collectlog -Pattern "TOTAL games=\d+ winrate=([0-9.]+)%" |
    Select-Object -Last 1
if (-not $total) {
    "ABORT: collection never printed a TOTAL line" | Out-File $trainlog -Encoding utf8
    exit 1
}
$wr = [double]$total.Matches[0].Groups[1].Value
if ($wr -lt 3.0) {
    "ABORT: contrast precondition failed - TOTAL winrate $wr% < 3%" |
        Out-File $trainlog -Encoding utf8
    exit 1
}
& python "$repo\training\nn\train_dmc.py" `
    --data "$repo\training\regime_r2*.pkl.gz" `
    --no-init --big --seed 0 --epochs 6 `
    --out "$repo\training\regime_qnet_r2.pth" `
    *> $trainlog
