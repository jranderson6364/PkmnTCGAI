# Phase B remainder (issue #3): the three Source B fresh-game blocks that
# died un-flushed in the night-2 starmie crash (main/lucario/abomasnow,
# 500 games each — starmie was re-collected separately as regime_r1d).
# Detached per the 2026-07-12 infra lesson. Chain regime_train_chained.ps1
# AFTER confirming this process is up.
$repo = "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
$log = "$repo\training\nn\regime_r1e_collect.log"
Start-Process -FilePath "python" `
  -ArgumentList "training/nn/regime_collect.py --continuations 0 --fresh-games 1500 --opponents main,lucario,abomasnow --eps 0.25 --verify-seats --out training/regime_r1e.pkl.gz" `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $log `
  -RedirectStandardError "$repo\training\nn\regime_r1e_collect.err.log" `
  -WindowStyle Hidden
Write-Host "launched; log: $log"
