"""PIT of the truth's min-rank depth within the fiducial cloud, with argmin locations."""
import math, sys, importlib
import numpy as np, torch
from scipy.stats import norm
fb = importlib.import_module("studroc_paper.methods.fiducial_band")

def run(n0, n1, mu, sd, reps, tag, M=3000, lo_cut=0.0, hi_cut=1.0):
    rng = np.random.default_rng(5)
    grid = np.arange(n0 + 1) / n0
    truth = norm.sf((norm.isf(grid) - mu) / sd)
    cols = np.flatnonzero((grid > lo_cut) & (grid < hi_cut) & (grid > 0) & (grid < 1))
    U = []; loc_t = []; loc_c = []; side_t = []
    for _ in range(reps):
        x = rng.standard_normal(n0); y = sd * rng.standard_normal(n1) + mu
        lab = fb._merged_labels(np.r_[np.zeros(n0, int), np.ones(n1, int)], np.r_[x, y], "random", rng)
        cloud = fb._fiducial_cloud(lab, n0, n1, M, grid, rng, torch.device("cpu"), torch.float64).numpy()[:, cols]
        srt = np.sort(cloud, axis=0)
        # inclusive two-sided ranks of each draw, column-wise
        order = np.argsort(cloud, axis=0, kind="stable")
        rk = np.empty_like(order); np.put_along_axis(rk, order, np.arange(1, M + 1)[:, None], axis=0)
        d = np.minimum(rk, M + 1 - rk)
        D = d.min(axis=1); am = d.argmin(axis=1)
        tr = truth[cols]
        below = np.array([np.searchsorted(srt[:, c], tr[c], side="right") for c in range(len(cols))])  # draws <= truth
        above = M - np.array([np.searchsorted(srt[:, c], tr[c], side="left") for c in range(len(cols))])
        dt = np.minimum(below, above) + 1  # depth the truth would have as an extra curve
        Dt = dt.min(); c_star = dt.argmin()
        U.append(np.mean(D < Dt))
        loc_t.append(grid[cols[c_star]]); side_t.append(below[c_star] < above[c_star])  # True: truth low in cloud => upper... truth near bottom
        sel = D <= np.quantile(D, 0.5)
        loc_c.extend(grid[cols[am[sel]]][:50])
    U = np.array(U); loc_t = np.array(loc_t); loc_c = np.array(loc_c); side_t = np.array(side_t)
    print(f"{tag} n={n0}/{n1} cols=({lo_cut},{hi_cut}) raw-tube miss at a=.05/.2/.5: {np.mean(U<.05):.3f} {np.mean(U<.2):.3f} {np.mean(U<.5):.3f}  mean U={U.mean():.3f}")
    bins = np.linspace(0, 1, 11)
    low = U < 0.5
    print("  truth argmin FPR deciles (U<.5):", np.round(np.histogram(loc_t[low], bins)[0] / max(low.sum(), 1), 2))
    print("  cloud argmin FPR deciles (D<=med):", np.round(np.histogram(loc_c, bins)[0] / len(loc_c), 2))
    print(f"  truth at bottom of cloud (truth below lower edge side) share among U<.5: {side_t[low].mean():.2f}", flush=True)

if __name__ == "__main__":
    reps = int(sys.argv[1])
    run(500, 500, 1.1902, 1.0, reps, "bn80")
    run(500, 500, 1.1902, 1.0, reps, "bn80", lo_cut=0.1, hi_cut=0.9)
    run(500, 500, 0.0, 1.0, reps, "diag")
    run(500, 500, 0.0, 1.0, reps, "diag", lo_cut=0.1, hi_cut=0.9)
    run(50, 50, 1.1902, 1.0, reps, "bn80")
