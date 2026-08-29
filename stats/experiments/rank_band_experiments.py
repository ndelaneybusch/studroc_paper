"""Minimal falsification experiments for rank-space ROC band candidates.

Arms:
  oracle    - M1 with R0 = true ROC (shared calibration). Validates the ELL
              machinery and measures the irreducible width of a rank-based band.
  plug_lin  - M1 with R0 = empirical ROC polyline (rank-based, per-rep calib).
  plug_step - M1 with R0 = empirical ROC staircase (negative control for
              smoothing; still uses fresh continuous uniform negatives).
  fiducial  - M2 Dirichlet-spacings composition sampler, ELL credible band.
  ks        - repo fixed-width KS band (baseline).
  wh        - repo Working-Hotelling band (baseline; exact representation
              of scores reconstructed per cell).

Everything lives in rank space: negatives ~ U(0,1), positives ~ CDF R_true.
Bands are evaluated on the grid t_k = k/n0 (staircase-upper convention).
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
from scipy.stats import norm
from scipy.stats import t as tdist
from scipy.stats import beta as beta_dist

beta_ppf = beta_dist.ppf

from studroc_paper.methods.ks_band import fixed_width_ks_band
from studroc_paper.methods.working_hotelling import working_hotelling_band

torch.set_num_threads(max(1, torch.get_num_threads()))


# ----------------------------------------------------------------------------
# True curves (rank space): represented as arrays (t, r), r = R_true(t)
# ----------------------------------------------------------------------------

def fine_grid():
    a = np.geomspace(1e-9, 0.05, 4000)
    b = np.linspace(0.05, 0.95, 4000)
    c = 1.0 - np.geomspace(1e-9, 0.05, 4000)[::-1]
    t = np.unique(np.clip(np.concatenate([[0.0], a, b, c, [1.0]]), 0, 1))
    return t


class Curve:
    def __init__(self, t, r):
        r = np.maximum.accumulate(np.clip(r, 0, 1))
        t = np.asarray(t, dtype=np.float64)
        self.t, self.r = t, r
        # generalized inverse: dedupe r keeping first (smallest t)
        ru, idx = np.unique(r, return_index=True)
        self.ri, self.ti = ru, t[idx]

    def eval(self, tq):
        return np.interp(tq, self.t, self.r)

    def inv(self, v):
        return np.interp(v, self.ri, self.ti)

    def auc(self):
        return np.trapezoid(self.r, self.t)


def make_binormal(auc):
    mu = np.sqrt(2.0) * norm.ppf(auc)
    t = fine_grid()
    tt = np.clip(t, 1e-15, 1 - 1e-15)
    r = norm.cdf(mu + norm.ppf(tt))
    r[t == 0] = 0.0
    r[t == 1] = 1.0
    return Curve(t, r)


def _curve_from_cgrid(F, G, c):
    """ROC from class CDFs on threshold grid c (predict + if score > c)."""
    tt = 1.0 - F(c)
    rr = 1.0 - G(c)
    o = np.argsort(tt)
    tt, rr = tt[o], rr[o]
    t = fine_grid()
    r = np.interp(t, tt, rr, left=0.0, right=1.0)
    r[t == 0] = 0.0
    r[t == 1] = 1.0
    return Curve(t, r)


def make_t_shape(auc_target, df=2.0):
    c = np.linspace(-400, 400, 200001)

    def build(delta):
        return _curve_from_cgrid(
            lambda x: tdist.cdf(x, df), lambda x: tdist.cdf(x - delta, df), c
        )

    lo_d, hi_d = 0.0, 60.0
    for _ in range(60):
        mid = 0.5 * (lo_d + hi_d)
        if build(mid).auc() < auc_target:
            lo_d = mid
        else:
            hi_d = mid
    return build(0.5 * (lo_d + hi_d))


def make_bimodal(auc_target, sep=3.0, weight=0.5):
    c = np.linspace(-15, 25, 200001)

    def F(x):
        return weight * norm.cdf(x) + (1 - weight) * norm.cdf(x - sep)

    def build(m):
        return _curve_from_cgrid(F, lambda x: norm.cdf(x - m), c)

    lo_m, hi_m = -5.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo_m + hi_m)
        if build(mid).auc() < auc_target:
            lo_m = mid
        else:
            hi_m = mid
    return build(0.5 * (lo_m + hi_m))


# score representations for WH/KS (exact monotone reconstruction per truth)
def repr_binormal(u_or_w):
    return -norm.ppf(np.clip(u_or_w, 1e-15, 1 - 1e-15))


def make_repr_t(df):
    def f(x):
        return tdist.ppf(np.clip(1.0 - x, 1e-15, 1 - 1e-15), df)

    return f


def make_repr_bimodal(sep=3.0, weight=0.5):
    c = np.linspace(-15, 25, 200001)
    Fc = weight * norm.cdf(c) + (1 - weight) * norm.cdf(c - sep)

    def f(x):
        return np.interp(np.clip(1.0 - x, 1e-15, 1 - 1e-15), Fc, c)

    return f


# ----------------------------------------------------------------------------
# Rank-space simulation and ELL band machinery
# ----------------------------------------------------------------------------

def uniform_order_stats(B, n, rng):
    E = rng.standard_exponential((B, n + 1))
    return np.cumsum(E[:, :n], axis=1) / E.sum(axis=1, keepdims=True)


def rhat_batch(U_sorted, W):
    """Empirical ROC values at t_k = k/n0 (staircase-upper convention).

    R(k/n0) = #{w < u_(k+1)}/n1 for k < n0, R(1) = 1.
    """
    B, n0 = U_sorted.shape
    n1 = W.shape[1]
    Ws = np.sort(W, axis=1)
    cnt = torch.searchsorted(
        torch.from_numpy(np.ascontiguousarray(Ws)),
        torch.from_numpy(np.ascontiguousarray(U_sorted)),
    ).numpy()
    out = np.empty((B, n0 + 1))
    out[:, :n0] = cnt / n1
    out[:, n0] = 1.0
    return out


def ell_bands(dev, alphas):
    """Equal-local-levels trimming. dev: (M, K). Returns per-alpha (lo, hi, j*).

    j* = max depth such that >= (1-alpha) of the M curves lie fully inside
    the pointwise [j-th smallest, j-th largest] tube (tie-inclusive).
    """
    M = dev.shape[0]
    Dt = torch.from_numpy(np.ascontiguousarray(dev.T))  # (K, M)
    csort, _ = torch.sort(Dt, dim=1)
    rank_le = torch.searchsorted(csort, Dt, right=True)
    rank_ge = M - torch.searchsorted(csort, Dt, right=False)
    s = torch.minimum(rank_le, rank_ge).min(dim=0).values.numpy()
    s_sorted = np.sort(s)
    out = {}
    for a in alphas:
        j = int(s_sorted[int(np.floor(a * M))])
        j = max(1, min(j, M // 2))
        lo = csort[:, j - 1].numpy().copy()
        hi = csort[:, M - j].numpy().copy()
        out[a] = (lo, hi, j)
    return out


def polyline_vertices(u, w):
    """Empirical ROC polyline vertices from rank-space data (rank-based)."""
    n0, n1 = len(u), len(w)
    allv = np.concatenate([u, w])
    lab = np.concatenate([np.zeros(n0), np.ones(n1)])
    o = np.argsort(allv, kind="stable")
    lab_s = lab[o]
    xs = np.concatenate([[0.0], np.cumsum(1 - lab_s) / n0])
    ys = np.concatenate([[0.0], np.cumsum(lab_s) / n1])
    return xs, ys, lab_s


def inv_first(ys, xs):
    """Generalized-inverse lookup arrays (dedupe ys keeping smallest xs)."""
    yu, idx = np.unique(ys, return_index=True)
    return yu, xs[idx]


def hazen_polyline(lab_s, n0, n1):
    """Strictly-increasing rank-based ROC smoother with pseudo-count tails.

    Negatives sit at expected uniform positions j/(n0+1); the i-th of r
    positives inside a negative gap sits at (j + i/(r+1))/(n0+1). Mirrored
    for the y-axis. Endpoints (0,0) and (1,1) give non-degenerate tails.
    """
    N = len(lab_s)
    is_pos = lab_s.astype(bool)
    ncnt = np.cumsum(~is_pos)  # negatives seen through element i
    pcnt = np.cumsum(is_pos)
    xs = np.empty(N)
    ys = np.empty(N)
    # x-axis: negatives at ncnt/(n0+1); positives spread inside current gap
    gap_x = ncnt.copy()
    gap_x[is_pos] = ncnt[is_pos]  # gap index for positives
    # index within run of consecutive positives (1-based) and run length
    sub = np.zeros(N)
    runlen = np.zeros(N)
    i = 0
    while i < N:
        if is_pos[i]:
            j = i
            while j < N and is_pos[j]:
                j += 1
            r = j - i
            sub[i:j] = np.arange(1, r + 1)
            runlen[i:j] = r
            i = j
        else:
            i += 1
    xs[~is_pos] = ncnt[~is_pos] / (n0 + 1)
    xs[is_pos] = (ncnt[is_pos] + sub[is_pos] / (runlen[is_pos] + 1)) / (n0 + 1)
    # y-axis mirrored
    sub2 = np.zeros(N)
    runlen2 = np.zeros(N)
    i = 0
    while i < N:
        if not is_pos[i]:
            j = i
            while j < N and not is_pos[j]:
                j += 1
            r = j - i
            sub2[i:j] = np.arange(1, r + 1)
            runlen2[i:j] = r
            i = j
        else:
            i += 1
    ys[is_pos] = pcnt[is_pos] / (n1 + 1)
    ys[~is_pos] = (pcnt[~is_pos] + sub2[~is_pos] / (runlen2[~is_pos] + 1)) / (n1 + 1)
    xs = np.concatenate([[0.0], xs, [1.0]])
    ys = np.concatenate([[0.0], ys, [1.0]])
    return xs, ys


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

TOL = 1e-12
LANDMARKS = (0.01, 0.05, 0.10, 0.50)


def band_metrics(L, U, rtrue_k, tk):
    """L, U in TPR space (already clipped)."""
    below = rtrue_k < L - TOL  # truth escapes under the band (optimistic band)
    above = rtrue_k > U + TOL
    viol = np.maximum(np.maximum(L - rtrue_k, rtrue_k - U), 0.0)
    depth = float(viol.max())
    m = {
        "cov": bool(not below.any() and not above.any()),
        "viol_low": bool(below.any()),   # truth below lower bound
        "viol_high": bool(above.any()),  # truth above upper bound
        "depth": depth,
        "worst_fpr": float(tk[int(viol.argmax())]) if depth > TOL else np.nan,
        "area": float(np.mean(U - L)),
    }
    for lm in LANDMARKS:
        k = int(round(lm * (len(tk) - 1)))
        m[f"w{lm}"] = float(U[k] - L[k])
    return m


def reflect_band(rhat, lo, hi):
    L = np.clip(rhat - hi, 0, 1)
    U = np.clip(rhat - lo, 0, 1)
    return L, U


# ----------------------------------------------------------------------------
# Fiducial (M2) arm
# ----------------------------------------------------------------------------

def _run_ids(mask_seq):
    """For elements where mask is True (in sequence order), return run ids of
    consecutive-True runs, aligned to the True elements only."""
    m = np.asarray(mask_seq, dtype=bool)
    starts = m & ~np.concatenate([[False], m[:-1]])
    return np.cumsum(starts)[m] - 1


def fiducial_curves(lab_s, n0, n1, M, tk, rng):
    """M fiducial ROC draws given merged label sequence (ascending rank order).

    Main gaps of each class CDF are Dirichlet(1,...,1); the other class's
    elements inside a gap are spread at uniform-order-statistic fractions of
    that gap (the natural interpolation of the fiducial CDF between order
    statistics).
    """
    is_pos = lab_s.astype(bool)
    N = len(lab_s)
    ncnt = np.cumsum(~is_pos).astype(int)  # negatives seen through element i
    pcnt = np.cumsum(is_pos).astype(int)
    p = rng.standard_exponential((M, n0 + 1))
    q = rng.standard_exponential((M, n1 + 1))
    Pc = np.cumsum(p, axis=1) / p.sum(axis=1, keepdims=True)  # S_k = first k gaps
    Qc = np.cumsum(q, axis=1) / q.sum(axis=1, keepdims=True)

    L = N + 2
    xv = np.zeros((M, L))
    yv = np.zeros((M, L))

    def axis_coords(count_own, is_other, Cc, n_own, n_other):
        """Coordinates on the axis owned by 'own' class for all N elements."""
        out = np.empty((M, N))
        own = ~is_other
        j_own = count_own[own]  # >= 1 at own elements
        out[:, own] = Cc[:, j_own - 1]
        # other-class elements: base + sorted-uniform fraction of current gap
        j = count_own[is_other]  # gap index j -> gap (j+1) with mass S_{j+1}-S_j
        base = np.where(j > 0, Cc[:, np.maximum(j - 1, 0)], 0.0)
        mass = Cc[:, j] - base
        rid = _run_ids(is_other)  # run ids for consecutive other-elements
        U = rng.random((M, is_other.sum()))
        key = np.sort(rid[None, :] + U, axis=1)
        frac = key - rid[None, :]
        out[:, is_other] = base + frac * mass
        return out

    xv[:, 1:-1] = axis_coords(ncnt, is_pos, Pc, n0, n1)
    yv[:, 1:-1] = axis_coords(pcnt, ~is_pos, Qc, n1, n0)
    xv[:, -1] = 1.0
    yv[:, -1] = 1.0
    # batched linear interpolation of (xv, yv) at tk
    xt = torch.from_numpy(xv)
    yt = torch.from_numpy(yv)
    tq = torch.from_numpy(np.broadcast_to(tk, (M, len(tk))).copy())
    idx = torch.searchsorted(xt.contiguous(), tq, right=True)
    idx = torch.clamp(idx, 1, L - 1)
    x1 = torch.gather(xt, 1, idx - 1)
    x2 = torch.gather(xt, 1, idx)
    y1 = torch.gather(yt, 1, idx - 1)
    y2 = torch.gather(yt, 1, idx)
    frac = (tq - x1) / torch.clamp(x2 - x1, min=1e-15)
    R = y1 + torch.clamp(frac, 0, 1) * (y2 - y1)
    return R.numpy()


# ----------------------------------------------------------------------------
# Cell runner
# ----------------------------------------------------------------------------

def run_cell(name, curve, score_repr, n0, n1, reps, M, M_or, alphas, seed,
             arms):
    rng = np.random.default_rng(seed)
    tk = np.arange(n0 + 1) / n0
    rtrue_k = curve.eval(tk)

    # data reps (batched)
    U_data = uniform_order_stats(reps, n0, rng)
    W_data = curve.inv(rng.random((reps, n1)))
    Rhat_data = rhat_batch(U_data, W_data)

    acc = {arm: {a: [] for a in alphas} for arm in arms}
    jstars = {arm: {a: [] for a in alphas} for arm in arms}

    # --- oracle: shared calibration ---
    if "oracle" in arms:
        Uo = uniform_order_stats(M_or, n0, rng)
        Wo = curve.inv(rng.random((M_or, n1)))
        Do = rhat_batch(Uo, Wo) - rtrue_k
        bands_o = ell_bands(Do, alphas)
        for r in range(reps):
            for a in alphas:
                lo, hi, j = bands_o[a]
                L, U = reflect_band(Rhat_data[r], lo, hi)
                acc["oracle"][a].append(band_metrics(L, U, rtrue_k, tk))
                jstars["oracle"][a].append(j)

    t0 = time.time()
    for r in range(reps):
        u, w = U_data[r], W_data[r]
        xs, ys, lab_s = polyline_vertices(u, w)
        r0_k = np.interp(tk, xs, ys)  # upper staircase at dup xs (np.interp: last)
        yi, xi = inv_first(ys, xs)

        for arm in arms:
            if arm in ("oracle", "ks", "wh"):
                continue
            if arm == "plug_hz":
                xs_h, ys_h = hazen_polyline(lab_s, n0, n1)
                r0h_k = np.interp(tk, xs_h, ys_h)
                Us = uniform_order_stats(M, n0, rng)
                V = np.interp(rng.random((M, n1)), ys_h, xs_h)
                dev = rhat_batch(Us, V) - r0h_k
            elif arm == "plug_lin":
                Us = uniform_order_stats(M, n0, rng)
                V = np.interp(rng.random((M, n1)), yi, xi)
                dev = rhat_batch(Us, V) - r0_k
            elif arm == "plug_step":
                Us = uniform_order_stats(M, n0, rng)
                idx = np.searchsorted(yi, rng.random((M, n1)), side="left")
                V = xi[np.minimum(idx, len(xi) - 1)]
                dev = rhat_batch(Us, V) - r0_k
            elif arm == "fiducial":
                Rt = fiducial_curves(lab_s, n0, n1, M, tk, rng)
                bands = ell_bands(Rt, alphas)
                khat = np.rint(Rhat_data[r] * n1).astype(int)
                for a in alphas:
                    lo, hi, j = bands[a]
                    L, U = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
                    acc[arm][a].append(band_metrics(L, U, rtrue_k, tk))
                    jstars[arm][a].append(j)
                    if "fid_cp" in arms:
                        # union upper edge with exact Clopper-Pearson upper
                        # bound at the band's own implied local level j/(M+1)
                        a_loc = j / (M + 1)
                        cp_up = np.ones(len(khat))
                        kk = khat < n1
                        cp_up[kk] = beta_ppf(1 - a_loc, khat[kk] + 1,
                                             n1 - khat[kk])
                        U2 = np.maximum.accumulate(np.maximum(U, cp_up))
                        acc["fid_cp"][a].append(
                            band_metrics(L, np.clip(U2, 0, 1), rtrue_k, tk))
                        jstars["fid_cp"][a].append(j)
                continue
            elif arm == "fid_cp":
                continue  # produced alongside "fiducial"
            bands = ell_bands(dev, alphas)
            for a in alphas:
                lo, hi, j = bands[a]
                L, U = reflect_band(Rhat_data[r], lo, hi)
                acc[arm][a].append(band_metrics(L, U, rtrue_k, tk))
                jstars[arm][a].append(j)

        # baselines on reconstructed scores
        if "ks" in arms or "wh" in arms:
            y_true = np.concatenate([np.zeros(n0), np.ones(n1)])
            y_score = np.concatenate([score_repr(u), score_repr(w)])
            for arm, fn in (("ks", fixed_width_ks_band),
                            ("wh", working_hotelling_band)):
                if arm not in arms:
                    continue
                try:
                    _, L, U = fn(y_true, y_score, k=n0 + 1, alpha=min(alphas))
                except Exception:
                    L = np.zeros(n0 + 1)
                    U = np.ones(n0 + 1)
                acc[arm][min(alphas)].append(band_metrics(L, U, rtrue_k, tk))
        if r % 50 == 49:
            print(f"  [{name}] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
                  flush=True)

    # summarize
    out = {}
    for arm in arms:
        out[arm] = {}
        for a in alphas:
            rows = acc[arm][a]
            if not rows:
                continue
            n = len(rows)
            depths = np.array([x["depth"] for x in rows])
            missers = depths > TOL
            out[arm][str(a)] = {
                "n": n,
                "coverage": float(np.mean([x["cov"] for x in rows])),
                "viol_low": float(np.mean([x["viol_low"] for x in rows])),
                "viol_high": float(np.mean([x["viol_high"] for x in rows])),
                "mean_depth_missers": float(depths[missers].mean()) if missers.any() else 0.0,
                "p95_depth": float(np.quantile(depths, 0.95)),
                "max_depth": float(depths.max()),
                "med_worst_fpr": float(np.nanmedian([x["worst_fpr"] for x in rows]))
                if missers.any() else np.nan,
                "area": float(np.mean([x["area"] for x in rows])),
                **{f"w{lm}": float(np.mean([x[f"w{lm}"] for x in rows]))
                   for lm in LANDMARKS},
                "mean_jstar": float(np.mean(jstars[arm][a])) if jstars[arm][a] else np.nan,
            }
    return out


CELLS = {
    "C1": dict(truth=("binormal", 0.75), n0=500, n1=500),
    "C2": dict(truth=("binormal", 0.95), n0=500, n1=500),
    "C3": dict(truth=("binormal", 0.95), n0=150, n1=150),
    "C4": dict(truth=("bimodal", 0.90), n0=500, n1=500),
    "C5": dict(truth=("t2", 0.95), n0=500, n1=500),
    "C6": dict(truth=("binormal", 0.95), n0=5000, n1=5000),
    "C7": dict(truth=("binormal", 0.90), n0=25, n1=25),
}


def build_truth(spec):
    kind, auc = spec
    if kind == "binormal":
        return make_binormal(auc), repr_binormal
    if kind == "t2":
        return make_t_shape(auc, df=2.0), make_repr_t(2.0)
    if kind == "bimodal":
        return make_bimodal(auc), make_repr_bimodal()
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["C1", "C2"])
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--M", type=int, default=800)
    ap.add_argument("--Mor", type=int, default=4000)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.2, 0.05])
    ap.add_argument("--arms", nargs="+",
                    default=["oracle", "plug_hz", "fiducial", "ks", "wh"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for cname in args.cells:
        spec = CELLS[cname]
        curve, srepr = build_truth(spec["truth"])
        print(f"== {cname}: {spec['truth']} n0={spec['n0']} n1={spec['n1']} "
              f"(true AUC={curve.auc():.4f}) ==", flush=True)
        t0 = time.time()
        res = run_cell(cname, curve, srepr, spec["n0"], spec["n1"],
                       args.reps, args.M, args.Mor, args.alphas,
                       args.seed + sum(ord(ch) for ch in cname), args.arms)
        res["_meta"] = {**spec, "true_auc": curve.auc(), "reps": args.reps,
                        "M": args.M, "Mor": args.Mor,
                        "runtime_s": time.time() - t0}
        results[cname] = res
        print(json.dumps({cname: res}, indent=1, default=str), flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=str)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
