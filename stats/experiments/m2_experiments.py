"""M2 follow-up experiments (P1-P5) for the rank-space fiducial ROC band.

Reuses the machinery in ``rank_band_experiments.py`` (same directory).

Sub-experiments (``--exp``):
  p1diag  Trim-level recalibration diagnostics: for every rep record the full
          profile of the fiducial band over a ladder of trim depths j, plus the
          depth statistic of the TRUE curve among the fiducial draws.  This
          yields, essentially for free, (a) coverage/area as a function of j,
          (b) coverage/area as a function of the "effective alpha" used in the
          fiducial trim rule, and hence the exact recalibration map
          alpha_eff(alpha) per cell.
  p1cal   Per-rep frequentist calibration of the trim depth through a plug-in
          (Hazen) curve.  Expensive (nested Monte Carlo), reduced reps.
  p2      New vulnerability slices (imbalance, AUC .99, AUC .55, kinked truth)
          with optional lower-edge (Beta/CP) allowance.
  p3      Ties / discreteness red team (quantised scores).
  p4      M-vs-K saturation characterisation + thinned-grid trimming.

Key identity used everywhere
----------------------------
Let R_1..R_M be the fiducial draws on the grid t_k = k/n0 and let
  a_k = #{m : R_m(t_k) <= c(t_k)},   b_k = #{m : R_m(t_k) >= c(t_k)}
for a reference curve c.  The pointwise [j-th smallest, j-th largest] tube
contains c iff j <= min_k min(a_k, b_k) =: S(c).  So the depth statistic of the
*truth* directly gives the coverage of every trim depth at once, and the
frequentist-optimal trim depth is the alpha-quantile of S(truth).  The
fiducial rule instead uses the alpha-quantile of S(draw) over the draws.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.stats import beta as beta_dist

import rank_band_experiments as rbe

beta_ppf = beta_dist.ppf
TOL = rbe.TOL
LANDMARKS = rbe.LANDMARKS


# ---------------------------------------------------------------------------
# extra truths
# ---------------------------------------------------------------------------

def make_kink(t_kink=0.004, tpr_kink=0.6):
    """Adversarial piecewise-linear truth: near-vertical to `tpr_kink` by
    FPR `t_kink`, then a shallow straight line to (1, 1)."""
    t = rbe.fine_grid()
    r = np.where(t <= t_kink,
                 tpr_kink * t / t_kink,
                 tpr_kink + (1 - tpr_kink) * (t - t_kink) / (1 - t_kink))
    r[t == 0] = 0.0
    r[t == 1] = 1.0
    return rbe.Curve(t, r)


def build_truth(spec):
    kind = spec[0]
    if kind == "kink":
        return make_kink(spec[1], spec[2]), rbe.repr_binormal
    return rbe.build_truth(spec)


CELLS = dict(rbe.CELLS)
CELLS.update({
    # P2: vulnerability slices
    "P2a": dict(truth=("binormal", 0.90), n0=900, n1=100),
    "P2b": dict(truth=("binormal", 0.90), n0=100, n1=900),
    "P2c": dict(truth=("binormal", 0.99), n0=500, n1=500),
    "P2d": dict(truth=("binormal", 0.99), n0=150, n1=150),
    "P2e": dict(truth=("binormal", 0.55), n0=500, n1=500),
    "P2f": dict(truth=("kink", 0.004, 0.6), n0=500, n1=500),
    # P4: M-vs-K
    "P4a": dict(truth=("binormal", 0.95), n0=500, n1=500),
    "P4b": dict(truth=("binormal", 0.95), n0=2000, n1=2000),
    "P4c": dict(truth=("binormal", 0.95), n0=5000, n1=5000),
})


# ---------------------------------------------------------------------------
# fiducial draws (chunked over M so big cells fit in RAM)
# ---------------------------------------------------------------------------

def fid_draws(lab_s, n0, n1, M, tk, rng, chunk=1500):
    if M <= chunk:
        return rbe.fiducial_curves(lab_s, n0, n1, M, tk, rng)
    out = np.empty((M, len(tk)))
    done = 0
    while done < M:
        m = min(chunk, M - done)
        out[done:done + m] = rbe.fiducial_curves(lab_s, n0, n1, m, tk, rng)
        done += m
    return out


# ---------------------------------------------------------------------------
# trim-depth ladder + exact binomial allowance tables
# ---------------------------------------------------------------------------

def make_ladder(M):
    lo = np.arange(1, min(41, M // 2 + 1))
    hi = np.unique(np.rint(np.geomspace(max(lo[-1], 1), max(M // 2, 1), 70)))
    lad = np.unique(np.concatenate([lo, hi])).astype(int)
    return lad[(lad >= 1) & (lad <= max(M // 2, 1))]


def cp_tables(ladder, M, n1, lo_mode="none"):
    """cp_up[j, khat] = exact CP upper bound for a Binomial(n1, p) count khat at
    one-sided level j/(M+1); cp_lo the mirrored lower bound.

    lo_mode: 'none'  -> no lower allowance
             'full'  -> the full mirrored CP lower bound at every khat
             'deg'   -> only the degenerate part (khat == 0 => bound 0), the
                        exact mirror of "CP upper == 1 when khat == n1"
    """
    aloc = ladder / (M + 1.0)
    kh = np.arange(n1 + 1)
    A, K = np.meshgrid(aloc, kh, indexing="ij")
    up = np.ones_like(A)
    m = K < n1
    up[m] = beta_ppf(1 - A[m], K[m] + 1, n1 - K[m])
    lo = None
    if lo_mode == "full":
        lo = np.zeros_like(A)
        m2 = K > 0
        lo[m2] = beta_ppf(A[m2], K[m2], n1 - K[m2] + 1)
    elif lo_mode == "deg":
        lo = np.ones_like(A)
        lo[K == 0] = 0.0
    return up, lo


# ---------------------------------------------------------------------------
# core per-rep analysis
# ---------------------------------------------------------------------------

def _kchunk_for(K, M):
    # keep each (kc, M) working block under ~2e7 doubles
    return max(1, min(K, int(2e7 // max(M, 1))))


def rep_profile(R, rtrue_k, khat, ladder, cp_up, cp_lo=None, trim_rows=None,
                lm_idx=None):
    """Full ladder profile of the fiducial band for one replicate.

    R          (M, K) fiducial draws
    rtrue_k    (K,)   evaluation truth on the grid
    khat       (K,)   empirical TPR counts (for the exact binomial allowances)
    trim_rows  optional index array: grid rows used for the min-p trim
               (band is still built and evaluated on all K rows)
    """
    M, K = R.shape
    J = len(ladder)
    Dt = torch.from_numpy(np.ascontiguousarray(R.T))     # (K, M)
    jl = torch.from_numpy((ladder - 1).astype(np.int64))
    ju = torch.from_numpy((M - ladder).astype(np.int64))
    LO = np.empty((K, J))
    HI = np.empty((K, J))
    s = torch.full((M,), M, dtype=torch.int64)
    rt = torch.from_numpy(np.ascontiguousarray(rtrue_k))
    Slow = M
    Shigh = M
    use_rows = None
    if trim_rows is not None:
        use_rows = np.zeros(K, dtype=bool)
        use_rows[trim_rows] = True
    kc = _kchunk_for(K, M)
    for k0 in range(0, K, kc):
        k1 = min(K, k0 + kc)
        Dc = Dt[k0:k1]
        cs, _ = torch.sort(Dc, dim=1)
        if use_rows is None or use_rows[k0:k1].any():
            src = cs
            tgt = Dc
            if use_rows is not None:
                sel = torch.from_numpy(np.nonzero(use_rows[k0:k1])[0])
                src = cs[sel]
                tgt = Dc[sel]
            rle = torch.searchsorted(src, tgt, right=True)
            rge = M - torch.searchsorted(src, tgt, right=False)
            s = torch.minimum(s, torch.minimum(rle, rge).min(dim=0).values)
        LO[k0:k1] = cs[:, jl].numpy()
        HI[k0:k1] = cs[:, ju].numpy()
        q = rt[k0:k1, None].contiguous()
        a = torch.searchsorted(cs, q, right=True).squeeze(1)
        b = M - torch.searchsorted(cs, q, right=False).squeeze(1)
        Slow = min(Slow, int(a.min()))
        Shigh = min(Shigh, int(b.min()))

    L = np.clip(LO.T, 0.0, 1.0)                      # (J, K)
    U = np.clip(HI.T, 0.0, 1.0)
    Uraw = U.copy()
    U = np.maximum.accumulate(np.maximum(U, cp_up[:, khat]), axis=1)
    if cp_lo is not None:
        L = np.maximum.accumulate(np.minimum(L, cp_lo[:, khat]), axis=1)

    d_lo = L - rtrue_k[None, :]
    d_hi = rtrue_k[None, :] - U
    viol = np.maximum(np.maximum(d_lo, d_hi), 0.0)
    depth = viol.max(axis=1)
    wf = np.where(depth > TOL, viol.argmax(axis=1), -1)
    out = {
        "Slow": Slow, "Shigh": Shigh,
        "vlow": (d_lo > TOL).any(axis=1),
        "vhigh": (d_hi > TOL).any(axis=1),
        "depth": depth,
        "worst_k": wf,
        "area": (U - L).mean(axis=1),
        "area_raw": (Uraw - np.clip(LO.T, 0, 1)).mean(axis=1),
        "s_sorted": np.sort(s.numpy()),
    }
    out["cov"] = ~(out["vlow"] | out["vhigh"])
    if lm_idx is not None:
        out["w"] = (U - L)[:, lm_idx]
    return out


def summarize_sel(prof, sel, tk, n0):
    """Aggregate ladder-indexed per-rep profiles at a per-rep chosen index."""
    reps = len(prof)
    idx = np.asarray(sel)
    cov = np.array([prof[r]["cov"][idx[r]] for r in range(reps)])
    vl = np.array([prof[r]["vlow"][idx[r]] for r in range(reps)])
    vh = np.array([prof[r]["vhigh"][idx[r]] for r in range(reps)])
    dp = np.array([prof[r]["depth"][idx[r]] for r in range(reps)])
    ar = np.array([prof[r]["area"][idx[r]] for r in range(reps)])
    wk = np.array([prof[r]["worst_k"][idx[r]] for r in range(reps)])
    W = np.array([prof[r]["w"][idx[r]] for r in range(reps)])
    miss = dp > TOL
    wfpr = tk[wk[miss]] if miss.any() else np.array([])
    return {
        "n": int(reps),
        "coverage": float(cov.mean()),
        "viol_low": float(vl.mean()),
        "viol_high": float(vh.mean()),
        "mean_depth_missers": float(dp[miss].mean()) if miss.any() else 0.0,
        "p95_depth": float(np.quantile(dp, 0.95)),
        "max_depth": float(dp.max()),
        "med_worst_fpr": float(np.median(wfpr)) if miss.any() else float("nan"),
        "frac_miss_corner": float(np.mean(wfpr <= 10.0 / n0)) if miss.any() else float("nan"),
        "area": float(ar.mean()),
        **{f"w{lm}": float(W[:, i].mean()) for i, lm in enumerate(LANDMARKS)},
    }


# ---------------------------------------------------------------------------
# data generation (continuous and quantised)
# ---------------------------------------------------------------------------

def gen_rep_data(curve, n0, n1, reps, rng):
    U = rbe.uniform_order_stats(reps, n0, rng)
    W = curve.inv(rng.random((reps, n1)))
    return U, W


def quantized_truth(curve, Q, mode="trapezoid"):
    """ROC of the Q-level quantised score under a tie-handling convention.

    'trapezoid' -- random tie break (== jittering): the score becomes genuinely
    continuous and its ROC is the linear interpolation through the bin corners.
    'staircase' -- all tied negatives ranked above all tied positives (the
    pessimistic convention): the ROC is the lower staircase.
    """
    e = np.arange(Q + 1) / Q
    if mode == "trapezoid":
        return rbe.Curve(e, curve.eval(e))
    t = np.unique(np.concatenate([np.linspace(0, 1, 40001), e,
                                  np.clip(e - 1e-9, 0, 1)]))
    b = np.ceil(t * Q - 1e-9)
    on_edge = np.abs(t * Q - np.round(t * Q)) < 1e-9
    r = np.where(on_edge, curve.eval(np.round(t * Q) / Q),
                 curve.eval(np.maximum(b - 1, 0) / Q))
    r[t == 0] = 0.0
    r[t == 1] = 1.0
    return rbe.Curve(t, r)


def tie_break(u, w, Q, mode, rng):
    """Return (u', w') pseudo-continuous rank-space values from Q-level scores.

    Bin index = floor(x*Q) in rank space (equal-probability bins for the
    negative class; larger x == lower score).
    mode 'jitter'  : uniform random position inside the bin (random tie break;
                     exactly valid for the trapezoidal ROC).
    mode 'even'    : each class spread evenly inside the bin at (i-.5)/count
                     (deterministic mid-rank-style interleaving).
    mode 'neg1st'  : all negatives placed before all positives inside a bin
                     (pessimistic / staircase-lower convention).
    """
    bu = np.minimum((u * Q).astype(int), Q - 1)
    bw = np.minimum((w * Q).astype(int), Q - 1)
    if mode == "jitter":
        return (bu + rng.random(len(bu))) / Q, (bw + rng.random(len(bw))) / Q

    def even_pos(b):
        o = np.argsort(b, kind="stable")
        bs = b[o]
        # rank within bin
        first = np.concatenate([[True], bs[1:] != bs[:-1]])
        start = np.maximum.accumulate(np.where(first, np.arange(len(bs)), 0))
        within = np.arange(len(bs)) - start
        cnt = np.bincount(bs, minlength=Q)[bs]
        pos = np.empty(len(bs))
        pos[o] = (within + 0.5) / np.maximum(cnt, 1)
        return pos

    if mode == "even":
        return (bu + even_pos(bu)) / Q, (bw + even_pos(bw)) / Q
    if mode == "neg1st":
        # negatives occupy [0, .5) of the bin, positives [.5, 1)
        return ((bu + 0.25 * even_pos(bu)) / Q,
                (bw + 0.5 + 0.25 * even_pos(bw)) / Q)
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# main per-cell driver used by p1diag / p2 / p3 / p4
# ---------------------------------------------------------------------------

def run_profile_cell(name, curve, n0, n1, reps, M, seed, *,
                     eval_curve=None, quant=None, tie_mode="jitter",
                     cp_lo_mode="none", thin=None, sub_Ms=(), verbose=True):
    """Run `reps` replicates, storing the full trim-depth ladder profile.

    Returns (profiles, ladder, meta) plus (for sub_Ms) profiles at reduced M.
    """
    rng = np.random.default_rng(seed)
    tk = np.arange(n0 + 1) / n0
    ev = eval_curve if eval_curve is not None else curve
    rtrue_k = ev.eval(tk)
    lm_idx = np.array([int(round(lm * n0)) for lm in LANDMARKS])

    Ms = sorted(set([M] + list(sub_Ms)))
    ladders = {m: make_ladder(m) for m in Ms}
    tabs = {m: cp_tables(ladders[m], m, n1, cp_lo_mode) for m in Ms}

    trim_rows = None
    if thin:
        keep = set(range(min(25, n0 + 1))) | set(range(max(0, n0 - 24), n0 + 1))
        keep |= set(range(0, n0 + 1, thin))
        trim_rows = np.array(sorted(keep))

    keys = [(m, None) for m in Ms]
    if trim_rows is not None:
        keys += [(m, trim_rows) for m in Ms]

    U_data, W_data = gen_rep_data(curve, n0, n1, reps, rng)
    prof = {("main" if m == M else f"M{m}") + ("" if tr is None else "_thin"):
            [] for m, tr in keys}
    kladder = {("main" if m == M else f"M{m}") + ("" if tr is None else "_thin"):
               ladders[m] for m, tr in keys}
    t0 = time.time()
    for r in range(reps):
        u, w = U_data[r], W_data[r]
        if quant is not None:
            u, w = tie_break(u, w, quant, tie_mode, rng)
            u = np.sort(u)
        _, _, lab_s = rbe.polyline_vertices(u, w)
        # empirical TPR counts at t_k (staircase-upper convention)
        khat = np.rint(rbe.rhat_batch(u[None, :], w[None, :])[0] * n1).astype(int)
        R = fid_draws(lab_s, n0, n1, M, tk, rng)
        for m, tr in keys:
            key = ("main" if m == M else f"M{m}") + ("" if tr is None else "_thin")
            Rm = R if m == M else R[:m]
            cu, cl = tabs[m]
            prof[key].append(rep_profile(Rm, rtrue_k, khat, ladders[m], cu, cl,
                                         trim_rows=tr, lm_idx=lm_idx))
        if verbose and (r % 25 == 24):
            print(f"  [{name}] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
                  flush=True)
    meta = dict(n0=n0, n1=n1, reps=reps, M=M, sub_Ms=list(sub_Ms),
                true_auc=float(ev.auc()), quant=quant, tie_mode=tie_mode,
                cp_lo_mode=cp_lo_mode, thin=thin, n_trim_rows=(None if trim_rows is None
                                        else int(len(trim_rows))),
                runtime_s=time.time() - t0)
    return prof, kladder, meta, tk


AE_GRID = np.round(np.arange(0.005, 0.9951, 0.005), 4)

# universal trim-level recalibration exponent: alpha_eff = 1 - (1-alpha)**C
C_RECAL = 2.2


def aggregate(prof, ladder, tk, n0, alphas, ae_grid=AE_GRID):
    """Aggregate a list of per-rep profiles into by-j and by-alpha_eff tables."""
    reps = len(prof)
    M = None
    J = len(ladder)
    # ---- by fixed j ----
    cov = np.array([p["cov"] for p in prof])
    area = np.array([p["area"] for p in prof])
    covraw = np.array([[min(p["Slow"], p["Shigh"]) >= j for j in ladder]
                       for p in prof])
    by_j = {
        "j": ladder.tolist(),
        "cov": cov.mean(axis=0).tolist(),
        "cov_raw": covraw.mean(axis=0).tolist(),
        "area": area.mean(axis=0).tolist(),
    }
    # ---- by alpha_eff (the fiducial trim rule at effective level ae) ----
    Msz = len(prof[0]["s_sorted"])
    jsel = np.empty((reps, len(ae_grid)), dtype=int)
    jval = np.empty((reps, len(ae_grid)), dtype=int)
    for r, p in enumerate(prof):
        ss = p["s_sorted"]
        jj = ss[np.floor(ae_grid * Msz).astype(int)]
        jj = np.clip(jj, 1, max(Msz // 2, 1))
        jval[r] = jj
        jsel[r] = np.clip(np.searchsorted(ladder, jj, side="right") - 1, 0, J - 1)
    by_ae = {"ae": ae_grid.tolist()}
    covs, areas, jm, vls, vhs = [], [], [], [], []
    for i in range(len(ae_grid)):
        idx = jsel[:, i]
        covs.append(float(np.mean([prof[r]["cov"][idx[r]] for r in range(reps)])))
        areas.append(float(np.mean([prof[r]["area"][idx[r]] for r in range(reps)])))
        vls.append(float(np.mean([prof[r]["vlow"][idx[r]] for r in range(reps)])))
        vhs.append(float(np.mean([prof[r]["vhigh"][idx[r]] for r in range(reps)])))
        jm.append(float(jval[:, i].mean()))
    by_ae.update(cov=covs, area=areas, mean_j=jm, vlow=vls, vhigh=vhs)
    # ---- headline: the plain fiducial rule at each nominal alpha ----
    fid = {}
    for a in alphas:
        i = int(np.argmin(np.abs(ae_grid - a)))
        d = summarize_sel(prof, jsel[:, i], tk, n0)
        d["mean_jstar"] = float(jval[:, i].mean())
        d["ae"] = float(ae_grid[i])
        fid[str(a)] = d
    # ---- recalibrated: largest ae whose coverage >= 1 - alpha ----
    recal = {}
    cova = np.array(covs)
    for a in alphas:
        ok = np.nonzero(cova >= 1 - a - 1e-12)[0]
        if len(ok) == 0:
            recal[str(a)] = {"ae_star": None}
            continue
        i = int(ok.max())
        d = summarize_sel(prof, jsel[:, i], tk, n0)
        d["mean_jstar"] = float(jval[:, i].mean())
        d["ae_star"] = float(ae_grid[i])
        recal[str(a)] = d
    # ---- fixed universal recalibration map ae = 1 - (1-alpha)^C ----
    fixed = {}
    for a in alphas:
        ae = 1.0 - (1.0 - a) ** C_RECAL
        i = int(np.argmin(np.abs(ae_grid - ae)))
        d = summarize_sel(prof, jsel[:, i], tk, n0)
        d["mean_jstar"] = float(jval[:, i].mean())
        d["ae_star"] = float(ae_grid[i])
        fixed[str(a)] = d
    # ---- S(truth) distribution (drives the exact recalibration) ----
    Strue = np.array([min(p["Slow"], p["Shigh"]) for p in prof])
    sdist = {
        "S_true_q": {str(q): float(np.quantile(Strue, q))
                     for q in (0.05, 0.1, 0.2, 0.5)},
        "S_draw_q": {str(q): float(np.mean([np.quantile(p["s_sorted"], q)
                                            for p in prof]))
                     for q in (0.05, 0.1, 0.2, 0.5)},
        "mean_S_true": float(Strue.mean()),
    }
    return {"by_j": by_j, "by_ae": by_ae, "fid_cp": fid, "recal": recal,
            "fid_rc": fixed, "depth_stats": sdist}


# ---------------------------------------------------------------------------
# P1cal: per-rep frequentist calibration of the trim depth via a plug-in curve
# ---------------------------------------------------------------------------

def calibrate_j(lab_s, n0, n1, tk, ladder_in, cp_in, ncal, m_in, rng, alphas):
    """Plug-in (Hazen) frequentist calibration of the trim depth.

    Simulate `ncal` rank-space datasets from the plug-in curve R0; for each,
    draw `m_in` fiducial curves, build the *whole* band ladder (CP allowance
    included, exactly as in production) and record whether it covers R0.  The
    calibrated depth is the largest ladder value whose simulated coverage is
    at least 1-alpha.  Returned on the m_in scale.
    """
    xs_h, ys_h = rbe.hazen_polyline(lab_s, n0, n1)
    r0_k = np.interp(tk, xs_h, ys_h)
    cov = np.zeros(len(ladder_in))
    for _ in range(ncal):
        uu = np.sort(rng.random(n0))
        ww = np.interp(rng.random(n1), ys_h, xs_h)
        _, _, ls = rbe.polyline_vertices(uu, ww)
        kh = np.rint(rbe.rhat_batch(uu[None, :], ww[None, :])[0]
                     * n1).astype(int)
        Rc = rbe.fiducial_curves(ls, n0, n1, m_in, tk, rng)
        p = rep_profile(Rc, r0_k, kh, ladder_in, cp_in)
        cov += p["cov"]
    cov /= ncal
    out = {}
    for al in alphas:
        ok = np.nonzero(cov >= 1 - al - 1e-12)[0]
        out[al] = int(ladder_in[ok.max()]) if len(ok) else 1
    return out, cov


def run_p1cal(cells, reps, M, ncal, m_in, alphas, seed, out_path):
    res = {}
    for cname in cells:
        spec = CELLS[cname]
        curve, _ = build_truth(spec["truth"])
        n0, n1 = spec["n0"], spec["n1"]
        tk = np.arange(n0 + 1) / n0
        rtrue_k = curve.eval(tk)
        lm_idx = np.array([int(round(lm * n0)) for lm in LANDMARKS])
        ladder = make_ladder(M)
        cu, _ = cp_tables(ladder, M, n1)
        ladder_in = make_ladder(m_in)
        cu_in, _ = cp_tables(ladder_in, m_in, n1)
        rng = np.random.default_rng(seed + sum(ord(c) for c in cname))
        U_data, W_data = gen_rep_data(curve, n0, n1, reps, rng)
        prof, jsel_cal = [], []
        t0 = time.time()
        print(f"== p1cal {cname} n0={n0} reps={reps} M={M} ncal={ncal} "
              f"m_in={m_in} ==", flush=True)
        for r in range(reps):
            u, w = U_data[r], W_data[r]
            _, _, lab_s = rbe.polyline_vertices(u, w)
            khat = np.rint(rbe.rhat_batch(u[None, :], w[None, :])[0]
                           * n1).astype(int)
            jc, _ = calibrate_j(lab_s, n0, n1, tk, ladder_in, cu_in, ncal,
                                m_in, rng, alphas)
            R = fid_draws(lab_s, n0, n1, M, tk, rng)
            prof.append(rep_profile(R, rtrue_k, khat, ladder, cu,
                                    lm_idx=lm_idx))
            # inner depth is on the m_in scale -> rescale to the outer M scale
            row = []
            for al in alphas:
                j_out = max(1, int(round(jc[al] * M / m_in)))
                j_out = min(j_out, M // 2)
                row.append(int(np.clip(np.searchsorted(ladder, j_out,
                                                       side="right") - 1,
                                       0, len(ladder) - 1)))
            jsel_cal.append(row)
            if r % 10 == 9:
                print(f"  [{cname}] rep {r+1}/{reps} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        jsel_cal = np.array(jsel_cal)
        cd = aggregate(prof, ladder, tk, n0, alphas)
        cd["fid_cal"] = {}
        for i, al in enumerate(alphas):
            d = summarize_sel(prof, jsel_cal[:, i], tk, n0)
            d["mean_jstar"] = float(ladder[jsel_cal[:, i]].mean())
            cd["fid_cal"][str(al)] = d
        cd["_meta"] = dict(spec, reps=reps, M=M, ncal=ncal, m_in=m_in,
                           runtime_s=time.time() - t0,
                           true_auc=float(curve.auc()))
        res[cname] = cd
        print(json.dumps({cname: {"fid_cp": cd["fid_cp"],
                                  "fid_cal": cd["fid_cal"],
                                  "recal": cd["recal"]}}, indent=1,
                         default=str), flush=True)
        _save(res, out_path)
    return res


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def _save(res, path):
    if path:
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"saved -> {path}", flush=True)


def run_profile_experiment(cells, reps, M, alphas, seed, out_path, *,
                           quant=None, tie_mode="jitter", cp_lo_mode="none",
                           thin=None, sub_Ms=(), eval_mode="trapezoid"):
    res = {}
    for cname in cells:
        spec = CELLS[cname]
        curve, _ = build_truth(spec["truth"])
        n0, n1 = spec["n0"], spec["n1"]
        ev = quantized_truth(curve, quant, eval_mode) if quant else None
        print(f"== {cname} {spec['truth']} n0={n0} n1={n1} reps={reps} M={M}"
              f"{f' Q={quant}/{tie_mode}/{eval_mode}' if quant else ''}"
              f"{f' thin={thin}' if thin else ''} ==", flush=True)
        prof, ladders, meta, tk = run_profile_cell(
            cname, curve, n0, n1, reps, M,
            seed + sum(ord(c) for c in cname), eval_curve=ev, quant=quant,
            tie_mode=tie_mode, cp_lo_mode=cp_lo_mode, thin=thin,
            sub_Ms=sub_Ms)
        cd = {}
        for key, pl in prof.items():
            cd[key] = aggregate(pl, ladders[key], tk, n0, alphas)
            cd[key]["_M"] = len(pl[0]["s_sorted"])
        cd["_meta"] = dict(spec, eval_mode=eval_mode, **meta)
        res[cname] = cd
        hl = {k: v["fid_cp"] for k, v in cd.items() if k != "_meta"}
        print(json.dumps({cname: hl}, indent=1, default=str), flush=True)
        _save(res, out_path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["p1diag", "p1cal", "p2", "p3", "p4"])
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--M", type=int, default=3000)
    ap.add_argument("--subM", nargs="*", type=int, default=[])
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.5, 0.2, 0.1, 0.05])
    ap.add_argument("--quant", type=int, default=None)
    ap.add_argument("--tie", default="jitter")
    ap.add_argument("--evalmode", default="trapezoid")
    ap.add_argument("--cplo", default="none",
                    choices=["none", "full", "deg"])
    ap.add_argument("--thin", type=int, default=None)
    ap.add_argument("--ncal", type=int, default=120)
    ap.add_argument("--min", dest="m_in", type=int, default=400)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.exp == "p1cal":
        run_p1cal(args.cells, args.reps, args.M, args.ncal, args.m_in,
                  args.alphas, args.seed, args.out)
    else:
        run_profile_experiment(
            args.cells, args.reps, args.M, args.alphas, args.seed, args.out,
            quant=args.quant, tie_mode=args.tie, cp_lo_mode=args.cplo,
            thin=args.thin, sub_Ms=tuple(args.subM), eval_mode=args.evalmode)


if __name__ == "__main__":
    main()
