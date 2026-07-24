# ACAN (action-conditioned advantage net) corpus collection — S2, 2026-07-23.
#
# Launches parallel DETACHED workers (the project's proven long-job pattern —
# the CLI harness's own background tracking kills these otherwise) collecting
# search-decision records from the SHIPPED d2/formula search (the 776 agent).
#
# Opponent mix is DIVERSE by design: the net deploys against a diverse ladder
# field, and the -39pp search-RNG contamination perturbs which states are
# REACHED, not the search's assessment AT a reached state -- so labels stay
# valid while coverage gets strictly better than mirror-only.
#
# Each worker gets a unique --tag so merged game ids stay collision-free.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$nn   = Join-Path $repo "training\nn"
$out  = Join-Path $nn "acan_corpus"
New-Item -ItemType Directory -Force $out | Out-Null

# pin the SHIPPED config (sweep closed d3 + phi4 as null; distill what shipped)
$env:TWOPLY_DEPTH = "2"
$env:TWOPLY_LEAF  = "formula"
$env:TWOPLY_NDET  = "3"

$games = 75
$jobs = @(
    @{ tag = "mir1"; opp = "$nn\twoply_agent.py" },
    @{ tag = "mir2"; opp = "$nn\twoply_agent.py" },
    @{ tag = "mir3"; opp = "$nn\twoply_agent.py" },
    @{ tag = "mir4"; opp = "$nn\twoply_agent.py" },
    @{ tag = "luc1"; opp = "$repo\opponents\lucario_agent.py" },
    @{ tag = "luc2"; opp = "$repo\opponents\lucario_agent.py" },
    @{ tag = "luc3"; opp = "$repo\opponents\lucario_agent.py" },
    @{ tag = "luc4"; opp = "$repo\opponents\lucario_agent.py" },
    @{ tag = "abo1"; opp = "$repo\opponents\abomasnow_agent.py" },
    @{ tag = "abo2"; opp = "$repo\opponents\abomasnow_agent.py" },
    @{ tag = "abo3"; opp = "$repo\opponents\abomasnow_agent.py" },
    @{ tag = "gri1"; opp = "$repo\opponents\grimmsnarl_agent.py" },
    @{ tag = "gri2"; opp = "$repo\opponents\grimmsnarl_agent.py" },
    @{ tag = "gri3"; opp = "$repo\opponents\grimmsnarl_agent.py" },
    @{ tag = "sta1"; opp = "$repo\opponents\starmie_agent.py" },
    @{ tag = "sta2"; opp = "$repo\opponents\starmie_agent.py" }
)

foreach ($j in $jobs) {
    $o = Join-Path $out $j.tag
    $args = @("$nn\collect_search_data.py", "--games", "$games",
              "--out", "$o", "--opponent", $j.opp, "--tag", $j.tag)
    Start-Process -FilePath "python" -ArgumentList $args `
        -RedirectStandardOutput "$o.log" -RedirectStandardError "$o.err.log" `
        -WindowStyle Hidden
    Start-Sleep -Milliseconds 300
}

Write-Output "launched $($jobs.Count) workers x $games games = $($jobs.Count * $games) games"
Write-Output "corpus dir: $out"
