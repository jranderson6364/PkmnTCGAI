@echo off
cd /d "C:\Users\jande\Downloads\Projects\PkmnTCGAI"
echo === Gauntlet baseline (v25c) start %date% %time% > training\overnight_logs\gauntlet_v25c.log
python training\gauntlet.py --candidate main.py --name v25c --panel random,lucario,abomasnow,starmie,v21,v22,v23 --games 200 >> training\overnight_logs\gauntlet_v25c.log 2>&1
echo === Gauntlet done %date% %time% >> training\overnight_logs\gauntlet_v25c.log

echo === BC re-collect start %date% %time% > training\overnight_logs\bc_collect.log
python training\bc_collect.py --games 2000 --out bc_data_v25c.pkl >> training\overnight_logs\bc_collect.log 2>&1
echo === BC collect done %date% %time% >> training\overnight_logs\bc_collect.log
