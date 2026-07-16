# Rescue-mode heuristic fix gate battery (report-log 2026-07-16, PRE-REG A
# + amendment). main.py MUST NOT be edited while this runs.
$repo = "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
Start-Process -FilePath "python" `
  -ArgumentList "training/nn/fix_gate.py --mirror 400 --anchor 200 --scenario-pairs 150" `
  -WorkingDirectory $repo `
  -RedirectStandardOutput "$repo\training\nn\fix_gate.log" `
  -RedirectStandardError "$repo\training\nn\fix_gate.err.log" `
  -WindowStyle Hidden
Write-Host "launched; log: training/nn/fix_gate.log"
