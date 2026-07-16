# Round-2 collection (issue #3, the one pre-registered iteration):
# one-step-deviation Source A + eps=0 Source B. See report-log 2026-07-15.
$repo = "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
$log = "$repo\training\nn\regime_r2_collect.log"
Start-Process -FilePath "python" `
  -ArgumentList "training/nn/regime_collect.py --continuations 1000 --deviate-once --fresh-games 2000 --opponents main,lucario,abomasnow,starmie --eps 0 --verify-seats --out training/regime_r2.pkl.gz" `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $log `
  -RedirectStandardError "$repo\training\nn\regime_r2_collect.err.log" `
  -WindowStyle Hidden
Write-Host "launched; log: $log"
