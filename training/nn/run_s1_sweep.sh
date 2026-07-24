cd "C:/Users/jande/Downloads/Projects/PkmnTCGAI"
LOG=training/nn/s1_sweep.log
echo "=== S1 SWEEP (mirror, n=200 each) — depth x value x N_DET ===" > $LOG
echo "reference d2_formula_n3 = twoply_agent.py (shipped 776 config)" >> $LOG

echo "" >> $LOG
echo "[A/B 1] does DEPTH help? d3_formula_n3 vs d2_formula_n3" >> $LOG
python training/ab_test.py training/nn/sweep_d3_formula_n3.py training/nn/twoply_agent.py 200 2>&1 | grep -aE "A win rate" >> $LOG

echo "" >> $LOG
echo "[A/B 2] does VALUE help at depth-3? d3_phi4_n3 vs d3_formula_n3" >> $LOG
python training/ab_test.py training/nn/sweep_d3_phi4_n3.py training/nn/sweep_d3_formula_n3.py 200 2>&1 | grep -aE "A win rate" >> $LOG

echo "" >> $LOG
echo "[A/B 3] value at depth-3 w/ MORE dets (variance control)? d3_phi4_n6 vs d3_formula_n6" >> $LOG
python training/ab_test.py training/nn/sweep_d3_phi4_n6.py training/nn/sweep_d3_formula_n6.py 200 2>&1 | grep -aE "A win rate" >> $LOG

echo "" >> $LOG
echo "=== S1 SWEEP DONE ===" >> $LOG
