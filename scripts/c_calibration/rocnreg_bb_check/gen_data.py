"""Simulate datasets for the ROCnReg check: t(2)/.99 (n=500/500) and the sliver DGP (AUC .80).
Scores are minus the placement values (rank-invariant), labels 0 = negative, 1 = positive.
Writes long CSVs (rep, score, label) and the truth at the evaluation FPRs."""
import sys, numpy as np
sys.path.insert(0, '/home/nathan/Documents/studroc_paper/scripts/c_calibration'); sys.path.insert(0, '/home/nathan/Documents/studroc_paper/src')
import os; os.chdir('/home/nathan/Documents/studroc_paper')
from corner_mechanism import make_sliver, t_shape
n = 500; reps = 200
ks = [1, 3, 5, 10, 25, 50, 100, 250, 450]
p_eval = 1 - np.array(ks) / n
shapes = {"t2_99": t_shape(df=2.0, auc=0.99)[1:3], "sliver80": make_sliver(0.80, n, 0.8, 0.25)[:2], "t30_95": t_shape(df=30.0, auc=0.95)[1:3]}
out = '.'
for name, (R, Rinv) in shapes.items():
    rng = np.random.default_rng(11)
    rows = []
    for r in range(reps):
        u = rng.random(n); w = Rinv(rng.random(n))
        rows.append(np.column_stack([np.full(2 * n, r), -np.concatenate([u, w]), np.r_[np.zeros(n, int), np.ones(n, int)]]))
    np.savetxt(f'{out}/data_{name}.csv', np.vstack(rows), delimiter=',', header='rep,score,label', comments='', fmt=['%d', '%.12g', '%d'])
    np.savetxt(f'{out}/truth_{name}.csv', np.column_stack([ks, p_eval, R(p_eval)]), delimiter=',', header='k,p,truth', comments='', fmt=['%d', '%.6f', '%.10g'])
    print(name, 'truth at p:', np.round(R(p_eval), 5))
