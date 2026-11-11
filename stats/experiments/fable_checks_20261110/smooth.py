"""Second-order mechanism check: the cloud centre is a kernel smooth of the empirical ROC.

Diagnostics: pointwise variance of the truth's standardized deviation from the cloud
centre, its roughness versus a cloud draw's, and a prototype trim calibrated on
kernel-smoothed draws ("smooth-truth trim").
"""
import math, sys, importlib
import numpy as np, torch
from scipy.stats import norm, beta as beta_dist
fb = importlib.import_module("studroc_paper.methods.fiducial_band")

def depths_vs_cloud(srt, P):
    M, K = srt.shape
    d = np.full(P.shape[0], M, dtype=np.int64)
    for k in range(K):
        col = srt[:, k]
        le = np.searchsorted(col, P[:, k], side="right")
        ge = M - np.searchsorted(col, P[:, k], side="left")
        d = np.minimum(d, np.minimum(le, ge))
    return d

def run(n0, n1, mu, sd, reps, tag, M=6000):
    rng = np.random.default_rng(21)
    grid = np.arange(n0 + 1) / n0
    truth = norm.sf((norm.isf(grid) - mu) / sd)
    K = len(grid); inner = slice(1, -1)
    h = np.sqrt(np.maximum(grid * (1 - grid), 1e-12) / (n0 + 2))
    W = np.exp(-0.5 * ((grid[None, :] - grid[:, None]) / np.maximum(h[:, None], 1e-9)) ** 2)
    W /= W.sum(axis=1, keepdims=True)
    alphas = (0.5, 0.2, 0.05)
    probe = [int(q * n0) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    z2 = []; rough_t = {1: [], 4: [], 16: []}; rough_c = {1: [], 4: [], 16: []}
    res = {a: [] for a in alphas}
    mid = n0 // 2
    for _ in range(reps):
        x = rng.standard_normal(n0); y = sd * rng.standard_normal(n1) + mu
        lab = fb._merged_labels(np.r_[np.zeros(n0, int), np.ones(n1, int)], np.r_[x, y], "random", rng)
        cpos = np.cumsum(lab); khat = np.concatenate([cpos[np.flatnonzero(lab == 0)], [n1]]).astype(np.int64)
        C = fb._fiducial_cloud(lab, n0, n1, M, grid, rng, torch.device("cpu"), torch.float64).numpy()
        m = C.mean(axis=0); s = C.std(axis=0) + 1e-12
        zt = (truth - m) / s
        z2.append(zt[probe] ** 2)
        zc = (C[:200] - m) / s
        for lag in rough_t:
            rough_t[lag].append((zt[mid + lag] - zt[mid]) ** 2)
            rough_c[lag].append(np.mean((zc[:, mid + lag] - zc[:, mid]) ** 2))
        srt = np.sort(C, axis=0)
        cols = np.arange(1, K - 1)
        d_raw = depths_vs_cloud(srt[:, cols], C[:, cols])
        P = m + (C - m) @ W.T
        d_sm = depths_vs_cloud(srt[:, cols], P[:, cols])
        for a in alphas:
            out = []
            for d in (d_raw, d_sm):
                j = int(np.sort(d)[int(math.floor(a * M))]); j = max(1, min(j, M // 2))
                lo, up = fb._apply_corner_allowances(lower=srt[j - 1], upper=srt[M - j], khat=khat, n1=n1, trim_depth=j, n_draws=M)
                above = truth > up + 1e-6; below = truth < lo - 1e-6
                core = (grid > 0.1) & (grid < 0.9)
                out += [j, bool(above[inner].any()), bool(below[inner].any()), bool((above | below)[core].any()), float(np.mean((up - lo)[inner]))]
            res[a].append(out)
    print(f"{tag} n={n0}/{n1}: E z_truth^2 at t=.1,.3,.5,.7,.9: {np.round(np.mean(z2, axis=0), 3)}")
    print("   roughness E(dz)^2 at lags 1,4,16 truth:", [round(float(np.mean(rough_t[l])), 4) for l in rough_t], "cloud:", [round(float(np.mean(rough_c[l])), 4) for l in rough_c])
    for a in alphas:
        A = np.array(res[a], float)
        print(f"   a={a}: production-style j={A[:,0].mean():.1f} cov={1-np.mean((A[:,1]+A[:,2])>0):.3f} (up {A[:,1].mean():.3f}, lo {A[:,2].mean():.3f}) core-miss={A[:,3].mean():.3f} | smooth-truth j={A[:,5].mean():.1f} cov={1-np.mean((A[:,6]+A[:,7])>0):.3f} (up {A[:,6].mean():.3f}, lo {A[:,7].mean():.3f}) core-miss={A[:,8].mean():.3f} width ratio={A[:,9].mean()/A[:,4].mean():.4f}", flush=True)

if __name__ == "__main__":
    reps = int(sys.argv[1])
    run(500, 500, 1.1902, 1.0, reps, "bn80")
    run(500, 500, 0.0, 1.0, reps, "diag")
    run(500, 500, 2.3262, 1.0, reps, "bn95")
    run(500, 500, 1.5, 3.0, reps, "het3")
    run(100, 100, 1.1902, 1.0, reps, "bn80")
    run(1500, 1500, 1.1902, 1.0, max(reps // 2, 50), "bn80", M=4000)
