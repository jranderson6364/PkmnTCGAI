# Phase C training, chained behind the Phase B collection (issue #3).
# Polls until no python3.11 process is running regime_collect, verifies the
# corpus exists, then trains the regime Q-net detached. Log: regime_train.log
$repo = "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
$corpus = "$repo\training\regime_r1.pkl.gz"
$deadline = (Get-Date).AddHours(6)
while ((Get-Date) -lt $deadline) {
    $collectors = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
        Where-Object { $_.CommandLine -like "*regime_collect*" }
    if (-not $collectors) { break }
    Start-Sleep -Seconds 60
}
if (-not (Test-Path $corpus)) {
    "collection never produced $corpus - aborting training chain" |
        Out-File "$repo\training\nn\regime_train.log" -Encoding utf8
    exit 1
}
& python "$repo\training\nn\train_dmc.py" `
    --data "$repo\training\regime_r1*.pkl.gz" `
    --no-init --big --seed 0 --epochs 6 `
    --out "$repo\training\regime_qnet.pth" `
    *> "$repo\training\nn\regime_train.log"
