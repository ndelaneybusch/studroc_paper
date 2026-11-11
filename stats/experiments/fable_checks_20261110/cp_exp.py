"""Paired effect of the CP-form upper allowance on the raw C=1 fiducial tube."""
import math, sys
import numpy as np, torch
from scipy.stats import norm, beta as beta_dist
import importlib; fb = importlib.import_module("studroc_paper.methods.fiducial_band")

def one(rng, n0, n1, mu, sd, alphas):
    x = rng.standard_normal(n0); y = sd * rng.standard_normal(n1) + mu
    yt = np.r_[np.zeros(n0, int), np.ones(n1, int)]; ys = np.r_[x, y]
    lab = fb._merged_labels(yt, ys, "random", rng)
    grid = np.arange(n0 + 1) / n0
    cpos = np.cumsum(lab); khat = np.concatenate([cpos[np.flatnonzero(lab == 0)], [n1]]).astype(np.int64)
    truth = norm.sf((norm.isf(grid) - mu) / sd)
    out = {}
    M = max(fb._auto_n_draws(len(grid), a) for a in alphas)
    cloud = fb._fiducial_cloud(lab, n0, n1, M, grid, rng, torch.device("cpu"), torch.float64)
    ds, _ = torch.sort(fb._minp_depths(cloud))
    for a in alphas:
        j = max(1, min(int(ds[int(math.floor(a * M))].item()), M // 2))
        lo_t, up_t = fb._pointwise_order_stats(cloud, j)
        lo = np.clip(lo_t.numpy(), 0, 1); up = np.clip(up_t.numpy(), 0, 1)
        lo[khat == 0] = 0; lo[0] = 0
        ell = j / (M + 1)
        cp = np.ones(len(khat)); it = khat < n1
        cp[it] = beta_dist.ppf(1 - ell, khat[it] + 1, n1 - khat[it])
        # mid-p style / Jeffreys allowance and corner-only allowance
        jf = np.ones(len(khat)); jf[it] = beta_dist.ppf(1 - ell, khat[it] + 0.5, n1 - khat[it] + 0.5)
        ups = {
            "prod": np.maximum.accumulate(np.maximum(up, cp)),
            "corner": np.maximum.accumulate(np.where(khat == n1, 1.0, up)),
            "jeff": np.maximum.accumulate(np.maximum(up, jf)),
            "raw": np.maximum.accumulate(up),
        }
        s = slice(1, -1)
        res = {}
        for k, u in ups.items():
            u = u.copy(); u[-1] = 1
            above = bool(np.any(truth[s] > u[s] + 1e-6)); below = bool(np.any(truth[s] < lo[s] - 1e-6))
            res[k] = (above, below, float(np.mean((u - lo)[s])))
        res["bind"] = float(np.mean((cp > up)[s])); res["ell"] = ell
        out[a] = res
    return out

def run(n0, n1, mu, sd, reps, tag):
    rng = np.random.default_rng(11); alphas = (0.5, 0.2, 0.05)
    acc = {a: [] for a in alphas}
    for _ in range(reps):
        r = one(rng, n0, n1, mu, sd, alphas)
        for a in alphas: acc[a].append(r[a])
    for a in alphas:
        line = [f"{tag} n={n0}/{n1} a={a} ell={np.mean([r['ell'] for r in acc[a]]):.4f} bind={np.mean([r['bind'] for r in acc[a]]):.2f}"]
        for k in ("prod", "jeff", "corner", "raw"):
            ab = np.array([r[k][:2] for r in acc[a]]); w = np.mean([r[k][2] for r in acc[a]])
            line.append(f"{k}: cov={1-np.mean(ab.any(1)):.3f} up={ab[:,0].mean():.3f} lo={ab[:,1].mean():.3f} w={w:.4f}")
        print(" | ".join(line), flush=True)

if __name__ == "__main__":
    reps = int(sys.argv[1])
    run(50, 50, 1.1902, 1.0, reps, "bn80")
    run(500, 500, 1.1902, 1.0, reps, "bn80")
    run(500, 500, 2.3262, 1.0, reps, "bn95")
    run(500, 500, 1.5, 3.0, reps, "het3")
    run(15, 15, 1.1902, 1.0, reps, "bn80")
