# Phase B overnight haul (issue #3): regime Q-net training corpus.
# Detached per the 2026-07-12 infra lesson (Start-Process survives the CLI
# harness's background-task lifecycle). Monitor: Get-Process python + log tail.
$repo = "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
$log = "$repo\training\nn\regime_r1_collect.log"
Start-Process -FilePath "python" `
  -ArgumentList "training/nn/regime_collect.py --continuations 1000 --fresh-games 2000 --eps 0.25 --verify-seats --out training/regime_r1.pkl.gz" `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $log `
  -RedirectStandardError "$repo\training\nn\regime_r1_collect.err.log" `
  -WindowStyle Hidden
Write-Host "launched; log: $log"
