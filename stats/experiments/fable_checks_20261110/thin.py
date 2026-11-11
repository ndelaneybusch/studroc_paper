"""Paired thinned-vs-full trim grid at n0=n1=5000 on one Rust cloud."""
import sys, importlib
import numpy as np
from scipy.stats import norm
fb = importlib.import_module("studroc_paper.methods.fiducial_band")
frs = importlib.import_module("studroc_paper.methods.fiducial_band_rs")
core = frs._require_fiducial_core()

def run(n0, n1, mu, sd, reps, tag, threads):
    rng = np.random.default_rng(3)
    grid = np.arange(n0 + 1) / n0
    truth = norm.sf((norm.isf(grid) - mu) / sd)
    rows = fb.production_trim_rows(len(grid)).astype(np.uint64)
    alphas = (0.5, 0.2, 0.05)
    acc = {a: [] for a in alphas}
    for r in range(reps):
        x = rng.standard_normal(n0); y = sd * rng.standard_normal(n1) + mu
        lab = fb._merged_labels(np.r_[np.zeros(n0, int), np.ones(n1, int)], np.r_[x, y], "random", rng)
        cpos = np.cumsum(lab); khat = np.concatenate([cpos[np.flatnonzero(lab == 0)], [n1]]).astype(np.int64)
        Ms = {a: fb._auto_n_draws(len(grid), a) for a in alphas}
        cloud = core.FiducialCloud(lab.astype(np.uint8), max(Ms.values()), int(rng.integers(0, 2**63)), threads)
        for a in alphas:
            m = Ms[a]
            jt = cloud.trim_depth(m, a, rows); jf = cloud.trim_depth(m, a, None)
            res = [jt, jf]
            js = sorted({jt, jf})
            tubes = dict(zip(js, cloud.tubes(m, js)))
            for j in (jt, jf):
                lo, up = fb._apply_corner_allowances(lower=tubes[j][0], upper=tubes[j][1], khat=khat, n1=n1, trim_depth=j, n_draws=m)
                s = slice(1, -1)
                res += [bool(np.any(truth[s] > up[s] + 1e-6)), bool(np.any(truth[s] < lo[s] - 1e-6)), float(np.mean((up - lo)[s]))]
            acc[a].append(res)
        if (r + 1) % 50 == 0 or r + 1 == reps:
            for a in alphas:
                A = np.array(acc[a], float)
                print(f"{tag} reps={r+1} a={a} M={Ms[a]} j_thin={A[:,0].mean():.2f} j_full={A[:,1].mean():.2f} | thin: cov={1-np.mean((A[:,2]+A[:,3])>0):.3f} up={A[:,2].mean():.3f} lo={A[:,3].mean():.3f} | full: cov={1-np.mean((A[:,5]+A[:,6])>0):.3f} up={A[:,5].mean():.3f} lo={A[:,6].mean():.3f} | width full/thin={A[:,7].mean()/A[:,4].mean():.4f}", flush=True)

if __name__ == "__main__":
    run(5000, 5000, 1.1902, 1.0, int(sys.argv[1]), "bn80", int(sys.argv[2]))
