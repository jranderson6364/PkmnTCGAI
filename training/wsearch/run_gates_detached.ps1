Set-Location 'C:\Users\jande\Downloads\Projects\PkmnTCGAI'
python training/wsearch/wsearch.py --gate training/wsearch/run1_mean_weights.json --gate-games 600 --anchors lucario,abomasnow,starmie,dragapult --workers 15 *>> 'C:\Users\jande\Downloads\Projects\PkmnTCGAI\training\wsearch\gates_detached.log'
python training/wsearch/wsearch.py --gate training/wsearch/run1_top_elite.json --gate-games 600 --anchors lucario,abomasnow,starmie,dragapult --workers 15 *>> 'C:\Users\jande\Downloads\Projects\PkmnTCGAI\training\wsearch\gates_detached.log'
Add-Content -Path 'C:\Users\jande\Downloads\Projects\PkmnTCGAI\training\wsearch\gates_detached.log' -Value '=== WSEARCH GATES DONE ==='
