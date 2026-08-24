"""M3 experiments: composed exact one-sample equal-local-levels (ELL) bands.

M3 is the provable finite-sample distribution-free guarantee layer for the
rank-space ROC problem.  Rank-space convention (identical to
``rank_band_experiments.py``): negatives ``u ~ U(0,1)`` iid, positives ``w``
with CDF ``R_true``; a point is classified positive when its value is below the
threshold, so ``FPR(c) = F0(c)``, ``TPR(c) = F1(c)`` and
``R(t) = F1(F0^{-1}(t))`` with both class CDFs increasing.  In rank space
``F0`` is the identity and ``F1 = R_true``, but the *method* only ever reads
the merged label sequence, exactly like the fiducial arm.

Construction
------------
For one sample ``Z_(1) < ... < Z_(n)`` from a continuous CDF ``H``,
``H(Z_(i)) ~ Beta(i, n+1-i)`` exactly, and the whole vector is distributed as
uniform order statistics.  A one-sided equal-local-levels band uses the same
local level ``gamma`` at every ``i``:

    lower:  H(Z_(i)) >= BetaInv(gamma; i, n+1-i)  for all i
    upper:  H(Z_(i)) <= BetaInv(1-gamma; i, n+1-i) for all i

The simultaneous coverage of the lower band is ``P(min_i BetaCDF(U_(i); i,
n+1-i) >= gamma)``, so ``gamma`` is the ``level``-quantile of that statistic;
the upper statistic has the same law by the ``u -> 1-u`` symmetry of uniform
order statistics.  For the two-sided band the corresponding statistic is
``min_i min(BetaCDF, 1-BetaCDF)``.  Both are distribution-free: they depend on
``(n, level)`` only, so calibration happens once per sample size, never per
replicate.

Between order statistics each one-sided band is extended monotonically
(``F^lo`` carries the bound of the last order statistic at or below ``c``;
``F^hi`` the bound of the next order statistic at or above ``c``; 0 / 1 beyond
the extremes).

Composition.  ``F0 >= F0^lo`` gives ``F0^{-1}(t) <= (F0^lo)^{-1}(t)``, hence
``R(t) <= F1^hi((F0^lo)^{-1}(t))``: the ROC upper edge composes the F1-upper
band with the F0-lower band.  Mirrored, the lower edge composes F1-lower with
F0-upper.  Written in terms of the merged sequence, with ``p_i`` the number of
positives ranked below the ``i``-th negative:

    U(t) = b1_hi[ p_{iup(t)} + 1 ],  iup(t) = min{ i : b0_lo[i] >= t }
    L(t) = b1_lo[ p_{ilo(t)-1} ],    ilo(t) = min{ i : b0_hi[i] >= t }

with ``b1_hi[n1+1] = 1``, ``b1_lo[0] = 0`` and ``U = 1`` when no ``iup``
exists.  Both index maps depend only on ``(n0, gamma0)``, so they are
precomputed once per (cell, level).

Alpha split.  Coverage of the composed band is at least
``P(E1 n E2) * P(E3 n E4)`` (the two class samples are independent), where
E1/E2 are the F0 lower/upper events and E3/E4 the F1 ones.  Two splits are
measured:

* ``sidak`` (primary): a two-sided ELL band per class at level
  ``alpha_class = 1 - sqrt(1-alpha)``, so the product is exactly ``1-alpha``.
* ``bonf`` (the literal four-component split): four one-sided bands at
  ``alpha/4`` each.

Endpoints ``R(0)=0`` and ``R(1)=1`` hold for every continuous DGP, so the band
is pinned there (``U(0)=L(0)=0``, ``L(1)=U(1)=1``); this is free validity and
makes areas comparable with the fiducial band, which pins them implicitly.

Experiments (``--exp``)
----------------------
``m3grid``  M3 alone over a ladder of nominal levels: realised coverage (a
            theorem says >= nominal), area, miss depths, and the *effective*
            level at which M3's coverage equals a target -- the M3 analogue of
            the ``recal`` ceiling, i.e. a direct measure of conservatism.
``joint``   M3 ladder plus the fiducial band in the same replicate: the miss
            cap (fiducial n M3(alpha/10)) and the two-directional containment
            probe against the fiducial band.
"""

import argparse
import json
import time

import numpy as np
import torch
from scipy.special import betainc
from scipy.stats import beta as beta_dist

import m2_experiments as m2
import rank_band_experiments as rbe

beta_ppf = beta_dist.ppf
TOL = rbe.TOL
LANDMARKS = rbe.LANDMARKS

# Nominal M3 levels evaluated in every replicate (nested: wider at small alpha).
LEVEL_GRID = np.array([
    0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15,
    0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
])


# ---------------------------------------------------------------------------
# ELL calibration: distribution-free, once per (n, B)
# ---------------------------------------------------------------------------

_CAL = {}


def ell_stats(n, B, seed=987654321):
    """Sorted Monte Carlo samples of the one- and two-sided ELL statistics.

    Returns ``(t1_sorted, t2_sorted)`` where
    ``t1 = min_i BetaCDF(U_(i); i, n+1-i)`` (the one-sided lower statistic) and
    ``t2 = min_i min(BetaCDF, 1-BetaCDF)`` (the two-sided one).
    """
    key = (n, B, seed)
    if key in _CAL:
        return _CAL[key]
    rng = np.random.default_rng(seed + n)
    i = np.arange(1, n + 1, dtype=float)
    a, b = i, n + 1.0 - i
    chunk = max(1, int(4e6 // max(n, 1)))
    t1 = np.empty(B)
    t2 = np.empty(B)
    done = 0
    while done < B:
        m = min(chunk, B - done)
        U = rbe.uniform_order_stats(m, n, rng)
        P = betainc(a, b, U)
        t1[done:done + m] = P.min(axis=1)
        t2[done:done + m] = np.minimum(P, 1.0 - P).min(axis=1)
        done += m
    out = (np.sort(t1), np.sort(t2))
    _CAL[key] = out
    return out


SAFETY_SE = 2.0


def gamma_at(stat_sorted, level, safety=SAFETY_SE):
    """Conservative empirical quantile of the ELL statistic.

    The plain choice ``gamma = T_(floor(level*B)+1)`` makes the *estimated*
    ``P(T < gamma)`` equal ``level``, which leaves the guarantee exposed to the
    calibration's own Monte Carlo error.  The index is therefore shaded down by
    ``safety`` binomial standard errors, so realised coverage stays on the
    conservative side of nominal by construction.  ``k = 0`` means the level is
    below the Monte Carlo resolution altogether; the most conservative
    tabulated gamma is returned (still the safe direction)."""
    B = len(stat_sorted)
    k = int(np.floor(level * B)
            - np.ceil(safety * np.sqrt(level * (1.0 - level) * B)))
    k = max(k, 0)
    return float(stat_sorted[min(k, B - 1)]), k


def ell_bounds(n, gamma):
    """Per-order-statistic lower/upper ELL bounds at local level ``gamma``."""
    i = np.arange(1, n + 1)
    return beta_ppf(gamma, i, n + 1 - i), beta_ppf(1.0 - gamma, i, n + 1 - i)


class M3Plan:
    """Level-dependent, data-independent part of one M3 band.

    ``iup`` / ``ilo`` are the negative-order-statistic index maps of the
    composition; ``b1hi_ext`` / ``b1lo_ext`` the positive-class ELL bounds with
    the exact degenerate entries (``b1hi[n1+1] = 1``, ``b1lo[0] = 0``).
    """

    def __init__(self, n0, n1, g0_lo, g0_hi, g1_lo, g1_hi):
        tk = np.arange(n0 + 1) / n0
        b0lo, _ = ell_bounds(n0, g0_lo)
        _, b0hi = ell_bounds(n0, g0_hi)
        b1lo, _ = ell_bounds(n1, g1_lo)
        _, b1hi = ell_bounds(n1, g1_hi)
        iup = np.searchsorted(b0lo, tk, side="left") + 1      # 1..n0+1
        self.sent = iup > n0                                  # no such i -> U=1
        self.iup = np.clip(iup, 1, n0)
        self.ilo = np.minimum(np.searchsorted(b0hi, tk, side="left") + 1,
                              n0 + 1)
        self.b1hi_ext = np.concatenate([[0.0], b1hi, [1.0]])   # index 1..n1+1
        self.b1lo_ext = np.concatenate([[0.0], b1lo])          # index 0..n1
        self.n0, self.n1 = n0, n1

    def band(self, pcnt):
        """``pcnt[i]`` = positives ranked below the i-th negative, ``pcnt[0]=0``."""
        U = self.b1hi_ext[pcnt[self.iup] + 1]
        U = np.where(self.sent, 1.0, U)
        L = self.b1lo_ext[pcnt[self.ilo - 1]]
        U[0] = 0.0
        L[0] = 0.0
        L[-1] = 1.0
        U[-1] = 1.0
        return L, U


def make_plans(n0, n1, levels, B0, B1, split="sidak"):
    """One :class:`M3Plan` per nominal level, plus the calibration bookkeeping."""
    t1_0, t2_0 = ell_stats(n0, B0)
    t1_1, t2_1 = ell_stats(n1, B1)
    plans, info = {}, {}
    for a in levels:
        if split == "sidak":
            ac = 1.0 - np.sqrt(1.0 - a)
            g0, k0 = gamma_at(t2_0, ac)
            g1, k1 = gamma_at(t2_1, ac)
            plans[a] = M3Plan(n0, n1, g0, g0, g1, g1)
            info[a] = dict(alpha_class=ac, gamma0=g0, gamma1=g1,
                           mc_index=[k0, k1])
        elif split == "bonf":
            ac = a / 4.0
            g0, k0 = gamma_at(t1_0, ac)
            g1, k1 = gamma_at(t1_1, ac)
            plans[a] = M3Plan(n0, n1, g0, g0, g1, g1)
            info[a] = dict(alpha_comp=ac, gamma0=g0, gamma1=g1,
                           mc_index=[k0, k1])
        else:
            raise ValueError(split)
    return plans, info


# ---------------------------------------------------------------------------
# per-replicate metrics
# ---------------------------------------------------------------------------

def band_row(L, U, rtrue, lm_idx):
    viol = np.maximum(np.maximum(L - rtrue, rtrue - U), 0.0)
    d = float(viol.max())
    return dict(vlow=bool((L - rtrue > TOL).any()),
                vhigh=bool((rtrue - U > TOL).any()),
                depth=d,
                worst_k=int(viol.argmax()) if d > TOL else -1,
                area=float(np.mean(U - L)),
                w=(U - L)[lm_idx],
                empty=bool((L - U > TOL).any()))


def summarize(rows, tk, n0):
    d = np.array([r["depth"] for r in rows])
    miss = d > TOL
    wk = np.array([r["worst_k"] for r in rows])
    wfpr = tk[wk[miss]] if miss.any() else np.array([])
    W = np.array([r["w"] for r in rows])
    return {
        "n": len(rows),
        "coverage": float(np.mean([not (r["vlow"] or r["vhigh"]) for r in rows])),
        "viol_low": float(np.mean([r["vlow"] for r in rows])),
        "viol_high": float(np.mean([r["vhigh"] for r in rows])),
        "mean_depth_missers": float(d[miss].mean()) if miss.any() else 0.0,
        "p95_depth": float(np.quantile(d, 0.95)),
        "max_depth": float(d.max()),
        "med_worst_fpr": float(np.median(wfpr)) if miss.any() else float("nan"),
        "frac_miss_corner": (float(np.mean(wfpr <= 10.0 / n0))
                             if miss.any() else float("nan")),
        "area": float(np.mean([r["area"] for r in rows])),
        "frac_empty": float(np.mean([r["empty"] for r in rows])),
        **{f"w{lm}": float(W[:, i].mean()) for i, lm in enumerate(LANDMARKS)},
    }


def pcnt_from_khat(khat, n0):
    """positives-below-the-i-th-negative array (length n0+1, entry 0 = 0)."""
    p = np.empty(n0 + 1, dtype=np.int64)
    p[0] = 0
    p[1:] = khat[:n0]
    return p


# ---------------------------------------------------------------------------
# fiducial band at chosen trim depths (production recipe: CP upper allowance)
# ---------------------------------------------------------------------------

def fid_sorted_and_depths(R):
    """Column-sorted draws plus each draw's min-p depth."""
    M, K = R.shape
    Dt = torch.from_numpy(np.ascontiguousarray(R.T))
    cs, _ = torch.sort(Dt, dim=1)
    s = torch.full((M,), M, dtype=torch.int64)
    kc = max(1, min(K, int(2e7 // max(M, 1))))
    for k0 in range(0, K, kc):
        k1 = min(K, k0 + kc)
        rle = torch.searchsorted(cs[k0:k1], Dt[k0:k1], right=True)
        rge = M - torch.searchsorted(cs[k0:k1], Dt[k0:k1], right=False)
        s = torch.minimum(s, torch.minimum(rle, rge).min(dim=0).values)
    return cs, np.sort(s.numpy())


def fid_band_at(cs, M, j, khat, n1):
    """Fiducial band at trim depth ``j`` with the exact CP upper allowance."""
    L = np.clip(cs[:, j - 1].numpy(), 0.0, 1.0)
    U = np.clip(cs[:, M - j].numpy(), 0.0, 1.0)
    aloc = j / (M + 1.0)
    cp = np.ones(n1 + 1)
    kk = np.arange(n1)
    cp[:n1] = beta_ppf(1.0 - aloc, kk + 1, n1 - kk)
    U = np.maximum.accumulate(np.maximum(U, cp[khat]))
    return L, np.clip(U, 0.0, 1.0)


def trim_depth(s_sorted, ae, M):
    j = int(s_sorted[min(int(np.floor(ae * M)), M - 1)])
    return max(1, min(j, M // 2))


# ---------------------------------------------------------------------------
# experiment: M3 alone over the level ladder
# ---------------------------------------------------------------------------

def run_m3grid(cells, reps, seed, out_path, B, levels, splits, diag_reps=0):
    res = {}
    for cname in cells:
        spec = m2.CELLS[cname]
        curve, _ = m2.build_truth(spec["truth"])
        n0, n1 = spec["n0"], spec["n1"]
        tk = np.arange(n0 + 1) / n0
        rtrue = curve.eval(tk)
        lm_idx = np.array([int(round(lm * n0)) for lm in LANDMARKS])
        rng = np.random.default_rng(seed + sum(ord(c) for c in cname))
        t0 = time.time()
        B0 = B if n0 <= 700 else max(20000, B // 4)
        B1 = B if n1 <= 700 else max(20000, B // 4)
        plans = {}
        info = {}
        for sp in splits:
            plans[sp], info[sp] = make_plans(n0, n1, levels, B0, B1, sp)
        tcal = time.time() - t0
        print(f"== m3grid {cname} {spec['truth']} n0={n0} n1={n1} reps={reps} "
              f"B=({B0},{B1}) calib {tcal:.0f}s ==", flush=True)

        U_data, W_data = m2.gen_rep_data(curve, n0, n1, reps, rng)
        rows = {sp: {a: [] for a in levels} for sp in splits}
        # component-wise realised coverage at alpha=.05, a direct sign-bug
        # detector: each of the four one-sided events must hold at its own rate
        cref = {sp: (ell_bounds(n0, info[sp][0.05]["gamma0"])
                     + ell_bounds(n1, info[sp][0.05]["gamma1"]))
                for sp in splits}
        comp = {sp: np.zeros(4) for sp in splits}
        t0 = time.time()
        for r in range(reps):
            u, w = U_data[r], W_data[r]
            khat = np.rint(rbe.rhat_batch(u[None, :], w[None, :])[0]
                           * n1).astype(np.int64)
            pcnt = pcnt_from_khat(khat, n0)
            us = np.sort(u)
            ws = curve.eval(np.sort(w))   # F1 at the positives' order stats
            for sp in splits:
                for a in levels:
                    L, U = plans[sp][a].band(pcnt)
                    rows[sp][a].append(band_row(L, U, rtrue, lm_idx))
                b0lo, b0hi, b1lo, b1hi = cref[sp]
                comp[sp] += np.array([
                    float((us >= b0lo - 1e-15).all()),
                    float((us <= b0hi + 1e-15).all()),
                    float((ws >= b1lo - 1e-15).all()),
                    float((ws <= b1hi + 1e-15).all()),
                ])
            if r % 100 == 99:
                print(f"  [{cname}] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
                      flush=True)
        cd = {"_meta": dict(spec, reps=reps, B0=B0, B1=B1,
                            true_auc=float(curve.auc()),
                            runtime_s=time.time() - t0)}
        for sp in splits:
            blk = {"levels": levels.tolist(),
                   "calib": {str(a): info[sp][a] for a in levels},
                   "component_cov": (comp[sp] / reps).tolist(),
                   "by_level": {str(a): summarize(rows[sp][a], tk, n0)
                                for a in levels}}
            cd[sp] = blk
        res[cname] = cd
        print(json.dumps({cname: {sp: {a: {k: v for k, v in
                                          cd[sp]["by_level"][a].items()
                                          if k in ("coverage", "area",
                                                   "viol_low", "viol_high",
                                                   "max_depth")}
                                      for a in ("0.5", "0.05")}
                                  for sp in splits}}, indent=1, default=str),
              flush=True)
        _save(res, out_path)
    return res


# ---------------------------------------------------------------------------
# experiment: M3 + fiducial in the same replicate (miss cap + containment)
# ---------------------------------------------------------------------------

CFID = 2.0   # production trim-level exponent (alpha_eff = 1-(1-alpha)^C)


def run_joint(cells, reps, M, seed, out_path, B, levels, alphas, cap_frac):
    res = {}
    for cname in cells:
        spec = m2.CELLS[cname]
        curve, _ = m2.build_truth(spec["truth"])
        n0, n1 = spec["n0"], spec["n1"]
        tk = np.arange(n0 + 1) / n0
        rtrue = curve.eval(tk)
        lm_idx = np.array([int(round(lm * n0)) for lm in LANDMARKS])
        rng = np.random.default_rng(seed + sum(ord(c) for c in cname))
        B0 = B if n0 <= 700 else max(20000, B // 4)
        B1 = B if n1 <= 700 else max(20000, B // 4)
        t0 = time.time()
        cap_levels = sorted({round(a * cap_frac, 6) for a in alphas})
        all_levels = np.array(sorted(set(levels.tolist()) | set(cap_levels),
                                     reverse=True))
        plans, info = make_plans(n0, n1, all_levels, B0, B1, "sidak")
        print(f"== joint {cname} {spec['truth']} n0={n0} n1={n1} reps={reps} "
              f"M={M} B=({B0},{B1}) calib {time.time()-t0:.0f}s ==", flush=True)

        U_data, W_data = m2.gen_rep_data(curve, n0, n1, reps, rng)
        arms = {}
        for a in alphas:
            for nm in ("fid_cp", "fid_rc", "fp_cp", "fp_rc",
                       "cap_cp", "cap_rc"):
                arms[(nm, a)] = []
        # interior masks: containment can fail only on a handful of grid points
        # near the corners, so the probe is run on nested interiors as well as
        # on the whole grid (k0 = number of grid points trimmed at each end).
        k0s = [0, 2, 5, 10, 25, max(1, int(round(0.05 * n0)))]
        masks = {}
        for k0 in k0s:
            mm = np.zeros(n0 + 1, dtype=bool)
            mm[k0:n0 + 1 - k0] = True
            masks[k0] = mm
        # containment fractions and the signed overhang of M3 outside the
        # fiducial band (the latter is the certified miss-depth cap)
        cont = {(a, ref, k0): {lv: [] for lv in all_levels}
                for a in alphas for ref in ("cp", "rc") for k0 in k0s}
        contr = {(a, ref, k0): {lv: [] for lv in all_levels}
                 for a in alphas for ref in ("cp", "rc") for k0 in k0s}
        poke = {(a, ref, k0): {lv: [] for lv in all_levels}
                for a in alphas for ref in ("cp", "rc") for k0 in k0s}
        bind = {(a, ref): [] for a in alphas for ref in ("cp", "rc")}
        jrec = {a: [] for a in alphas}
        t0 = time.time()
        for r in range(reps):
            u, w = U_data[r], W_data[r]
            _, _, lab_s = rbe.polyline_vertices(u, w)
            khat = np.rint(rbe.rhat_batch(u[None, :], w[None, :])[0]
                           * n1).astype(np.int64)
            pcnt = pcnt_from_khat(khat, n0)
            m3 = {lv: plans[lv].band(pcnt) for lv in all_levels}
            R = m2.fid_draws(lab_s, n0, n1, M, tk, rng)
            cs, s_sorted = fid_sorted_and_depths(R)
            del R
            for a in alphas:
                jc = trim_depth(s_sorted, a, M)
                jr = trim_depth(s_sorted, 1.0 - (1.0 - a) ** CFID, M)
                jrec[a].append((jc, jr))
                for nm, j in (("cp", jc), ("rc", jr)):
                    Lf, Uf = fid_band_at(cs, M, j, khat, n1)
                    arms[(f"fid_{nm}", a)].append(
                        band_row(Lf, Uf, rtrue, lm_idx))
                    # The published M2 recipe applies the CP upper allowance at
                    # t=0 as well, where the staircase-upper empirical count
                    # khat[0] can be large while R(0) = 0 exactly.  Pinning
                    # U(0)=0 is free validity, and M3 is pinned there, so the
                    # cap and containment comparisons use the pinned band --
                    # otherwise every statistic is dominated by that one point.
                    Uf = Uf.copy()
                    Uf[0] = 0.0
                    arms[(f"fp_{nm}", a)].append(
                        band_row(Lf, Uf, rtrue, lm_idx))
                    Lm, Um = m3[round(a * cap_frac, 6)]
                    Lx, Ux = np.maximum(Lf, Lm), np.minimum(Uf, Um)
                    arms[(f"cap_{nm}", a)].append(
                        band_row(Lx, Ux, rtrue, lm_idx))
                    bind[(a, nm)].append(
                        float(np.mean((Lx > Lf + TOL) | (Ux < Uf - TOL))))
                    for lv in all_levels:
                        Ll, Uu = m3[lv]
                        dlo = Lf - Ll      # >0 where M3 sits below fid's floor
                        dhi = Uu - Uf      # >0 where M3 sits above fid's roof
                        for k0, mm in masks.items():
                            # M3(lv) inside fid(alpha): the domination direction
                            cont[(a, nm, k0)][lv].append(
                                bool((dlo[mm] <= TOL).all()
                                     and (dhi[mm] <= TOL).all()))
                            # M3(lv) outside fid(alpha): the miss-cap direction
                            contr[(a, nm, k0)][lv].append(
                                bool((dlo[mm] >= -TOL).all()
                                     and (dhi[mm] >= -TOL).all()))
                            # sup overhang = the depth a miss of the capped
                            # band can still reach when the truth is in M3(lv)
                            poke[(a, nm, k0)][lv].append(
                                float(max(dlo[mm].max(), dhi[mm].max(), 0.0)))
            del cs
            if r % 25 == 24:
                print(f"  [{cname}] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
                      flush=True)

        cd = {"_meta": dict(spec, reps=reps, M=M, B0=B0, B1=B1, C=CFID,
                            cap_frac=cap_frac, true_auc=float(curve.auc()),
                            k0s=k0s, runtime_s=time.time() - t0),
              "levels": all_levels.tolist(),
              "calib": {str(a): info[a] for a in all_levels},
              "arms": {f"{nm}|{a}": summarize(v, tk, n0)
                       for (nm, a), v in arms.items()},
              "mean_j": {str(a): [float(np.mean([x[0] for x in jrec[a]])),
                                  float(np.mean([x[1] for x in jrec[a]]))]
                         for a in alphas},
              "bind_frac": {f"{a}|{ref}": float(np.mean(v))
                            for (a, ref), v in bind.items()},
              "poke": {f"{a}|{ref}|{k0}": {str(lv): [float(np.mean(v)),
                                                    float(np.quantile(v, 0.95)),
                                                    float(np.max(v))]
                                           for lv, v in dd.items()}
                       for (a, ref, k0), dd in poke.items()},
              "contain_in": {f"{a}|{ref}|{k0}": {str(lv): float(np.mean(v))
                                                 for lv, v in dd.items()}
                             for (a, ref, k0), dd in cont.items()},
              "contain_out": {f"{a}|{ref}|{k0}": {str(lv): float(np.mean(v))
                                                  for lv, v in dd.items()}
                              for (a, ref, k0), dd in contr.items()}}
        res[cname] = cd
        print(json.dumps({cname: {k: cd["arms"][k] for k in sorted(cd["arms"])
                                  if k.endswith("|0.05")}}, indent=1,
                         default=str), flush=True)
        _save(res, out_path)
    return res


def _save(res, path):
    if path:
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"saved -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["m3grid", "joint"])
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--M", type=int, default=3000)
    ap.add_argument("--B", type=int, default=100000)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.5])
    ap.add_argument("--splits", nargs="+", default=["sidak", "bonf"])
    ap.add_argument("--capfrac", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.exp == "m3grid":
        run_m3grid(args.cells, args.reps, args.seed, args.out, args.B,
                   LEVEL_GRID, args.splits)
    else:
        run_joint(args.cells, args.reps, args.M, args.seed, args.out, args.B,
                  LEVEL_GRID, args.alphas, args.capfrac)


if __name__ == "__main__":
    main()
