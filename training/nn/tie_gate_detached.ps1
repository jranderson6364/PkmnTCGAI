# Learned tie-breaker gate (report-log 2026-07-16 pre-registration).
$repo = "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
Start-Process -FilePath "python" `
  -ArgumentList "training/nn/tie_gate.py --mirror 400 --anchor 200" `
  -WorkingDirectory $repo `
  -RedirectStandardOutput "$repo\training\nn\tie_gate.log" `
  -RedirectStandardError "$repo\training\nn\tie_gate.err.log" `
  -WindowStyle Hidden
Write-Host "launched; log: training/nn/tie_gate.log"
