"""M4b experiments: bracketed (worst-case) calibration of the fiducial trim.

Exact rank-test inversion (M4) is intractable, so the testable relaxation is:
use M3 at overall level 50% as a cheap confidence set of curves, pick a few
extreme members, calibrate the fiducial trim level frequentistically against
each, and take the most conservative answer.  Unlike per-replicate *plug-in*
calibration (m2_report.md 1c), which is biased conservative by 1.3-1.7x in the
trim depth because one dataset's plug-in curve is rougher than the truth, a
worst-case bracket over a set is at least aiming at a set that contains the
truth with known probability.

What is measured
----------------
``bracket``  Per replicate: the M3(50%) band, three raw members of the set
             (lower edge, midline, upper edge) and their monotone-smoothed
             counterparts, a frequentist calibration of the trim depth against
             each member, the worst-case (smallest) depth over each trio, and
             the resulting band's coverage / area against ``fid_cp`` (C=1) and
             ``fid_rc`` (C=2).  Two calibration read-outs per member:
             ``thresh`` -- the largest ladder depth whose simulated coverage of
             the member reaches 1-alpha, i.e. exactly the ``calibrate_j``
             convention of m2 -- and ``quant`` -- the alpha-quantile of the
             member's own min-p depth ``S`` across the inner simulations, which
             is the same target computed as an order statistic instead of a
             threshold crossing (cheaper in Monte Carlo, mildly conservative
             because it ignores the CP upper allowance).
             The oracle calibration (against the *true* curve) is run once per
             cell at two inner budgets, which separates the bracket's own bias
             from the bias induced by a small ``ncal``.
``family``   The premise check: the exact per-curve calibration ceiling
             ``ae*`` for an ordered family of truths, to see whether the
             calibration functional is monotone in early slope.

Candidate members are piecewise-linear on the FPR grid, hence continuous CDFs;
they are sampled with an explicit generalized inverse (``sample_curve``)
because ``rbe.Curve.inv`` is only correct for strictly increasing curves and
the M3 edges have long flat stretches.
"""

import argparse
import json
import time

import numpy as np
import torch

import m2_experiments as m2
import m3_experiments as m3
import rank_band_experiments as rbe

TOL = rbe.TOL
LANDMARKS = rbe.LANDMARKS


# ---------------------------------------------------------------------------
# candidate curves on the FPR grid
# ---------------------------------------------------------------------------

def sample_curve(tk, y, size, rng):
    """Draw ``size`` values with CDF the piecewise-linear curve ``(tk, y)``.

    Uses the exact generalized inverse of the interpolant, which stays correct
    across flat stretches (where the density is zero) and steep ones.
    """
    v = rng.random(size)
    j = np.clip(np.searchsorted(y, v, side="right"), 1, len(y) - 1)
    y0, y1 = y[j - 1], y[j]
    frac = np.where(y1 > y0, (v - y0) / np.maximum(y1 - y0, 1e-300), 0.0)
    return tk[j - 1] + frac * (tk[j] - tk[j - 1])


def smooth_curve(y, w):
    """Monotone moving-average smoothing of a grid curve, endpoints pinned."""
    k = np.ones(w) / w
    z = np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k,
                                                     mode="same")
    z = np.maximum.accumulate(np.clip(z, 0.0, 1.0))
    z[0] = 0.0
    z[-1] = 1.0
    return np.maximum.accumulate(z)


def members(L, U, n0):
    """The bracket's candidate members: raw M3 edges/midline, then smoothed."""
    mid = 0.5 * (L + U)
    w = max(3, int(round(np.sqrt(n0))) | 1)
    raw = [("lo", L.copy()), ("mid", mid), ("hi", U.copy())]
    for c in raw:
        c[1][0] = 0.0
        c[1][-1] = 1.0
    sm = [("s_" + nm, smooth_curve(y, w)) for nm, y in raw]
    return raw + sm


# ---------------------------------------------------------------------------
# frequentist calibration of the trim depth against a hypothesized curve
# ---------------------------------------------------------------------------

def calibrate(y0, n0, n1, tk, ladder, cp_up, ncal, m_in, rng, alphas):
    """Calibrate the trim depth against the curve ``y0`` (values on ``tk``).

    Returns ``{alpha: (j_thresh, j_quant)}`` on the ``m_in`` depth scale.
    """
    cov = np.zeros(len(ladder))
    S = np.empty(ncal, dtype=int)
    for c in range(ncal):
        uu = np.sort(rng.random(n0))
        ww = sample_curve(tk, y0, n1, rng)
        _, _, ls = rbe.polyline_vertices(uu, ww)
        kh = np.rint(rbe.rhat_batch(uu[None, :], ww[None, :])[0]
                     * n1).astype(int)
        Rc = rbe.fiducial_curves(ls, n0, n1, m_in, tk, rng)
        p = m2.rep_profile(Rc, y0, kh, ladder, cp_up)
        cov += p["cov"]
        S[c] = min(p["Slow"], p["Shigh"])
    cov /= ncal
    Ss = np.sort(S)
    out = {}
    for a in alphas:
        ok = np.nonzero(cov >= 1 - a - 1e-12)[0]
        jt = int(ladder[ok.max()]) if len(ok) else 1
        jq = int(max(1, Ss[min(int(np.floor(a * ncal)), ncal - 1)]))
        out[a] = (jt, jq)
    return out, cov.tolist()


def rescale(j_in, m_in, M):
    return int(np.clip(round(j_in * (M + 1) / (m_in + 1)), 1, max(M // 2, 1)))


# ---------------------------------------------------------------------------
# bracket experiment
# ---------------------------------------------------------------------------

def run_bracket(cells, reps, M, ncal, m_in, ncal_or, alphas, seed, out_path, B,
                set_level):
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
        plans, _ = m3.make_plans(n0, n1, np.array([set_level]), B0, B0,
                                 "sidak")
        plan = plans[set_level]
        ladder = m2.make_ladder(m_in)
        cp_in, _ = m2.cp_tables(ladder, m_in, n1)
        print(f"== bracket {cname} {spec['truth']} n0={n0} n1={n1} reps={reps} "
              f"M={M} ncal={ncal} m_in={m_in} set_level={set_level} ==",
              flush=True)

        t0 = time.time()
        orc = {}
        for nc in sorted({ncal, ncal_or}):
            o, _ = calibrate(rtrue, n0, n1, tk, ladder, cp_in, nc, m_in, rng,
                             alphas)
            orc[nc] = {str(a): [rescale(o[a][0], m_in, M),
                                rescale(o[a][1], m_in, M)] for a in alphas}
            print(f"   oracle calibration ncal={nc}: "
                  f"{orc[nc]} ({time.time()-t0:.0f}s)", flush=True)

        U_data, W_data = m2.gen_rep_data(curve, n0, n1, reps, rng)
        names = [nm for nm, _ in members(np.zeros(n0 + 1), np.ones(n0 + 1), n0)]
        arms = {}
        for a in alphas:
            for nm in ["fid_cp", "fid_rc", "brk_raw", "brk_sm", "brk_all",
                       "brk_raw_q", "brk_sm_q", "orc_j"]:
                arms[(nm, a)] = []
        jlog = {a: {nm: [] for nm in names + [n + "_q" for n in names]
                    + ["brk_raw", "brk_sm", "brk_all", "brk_raw_q",
                       "brk_sm_q", "cp", "rc"]} for a in alphas}
        slope = {nm: [] for nm in names}
        t0 = time.time()
        for r in range(reps):
            u, w = U_data[r], W_data[r]
            _, _, lab_s = rbe.polyline_vertices(u, w)
            khat = np.rint(rbe.rhat_batch(u[None, :], w[None, :])[0]
                           * n1).astype(np.int64)
            pcnt = m3.pcnt_from_khat(khat, n0)
            L50, U50 = plan.band(pcnt)
            cands = members(L50, U50, n0)
            jm = {}
            for nm, y0 in cands:
                cal, _ = calibrate(y0, n0, n1, tk, ladder, cp_in, ncal, m_in,
                                   rng, alphas)
                jm[nm] = cal
                slope[nm].append(float(y0[lm_idx[1]]))
            R = m2.fid_draws(lab_s, n0, n1, M, tk, rng)
            cs, s_sorted = m3.fid_sorted_and_depths(R)
            del R
            for a in alphas:
                jc = m3.trim_depth(s_sorted, a, M)
                jr = m3.trim_depth(s_sorted, 1.0 - (1.0 - a) ** m3.CFID, M)
                jraw = min(rescale(jm[nm][a][0], m_in, M)
                           for nm in ("lo", "mid", "hi"))
                jsm = min(rescale(jm[nm][a][0], m_in, M)
                          for nm in ("s_lo", "s_mid", "s_hi"))
                jrq = min(rescale(jm[nm][a][1], m_in, M)
                          for nm in ("lo", "mid", "hi"))
                jsq = min(rescale(jm[nm][a][1], m_in, M)
                          for nm in ("s_lo", "s_mid", "s_hi"))
                jall = min(jraw, jsm)
                jo = orc[ncal_or][str(a)][0]
                for nm, jj in (("fid_cp", jc), ("fid_rc", jr),
                               ("brk_raw", jraw), ("brk_sm", jsm),
                               ("brk_all", jall), ("brk_raw_q", jrq),
                               ("brk_sm_q", jsq), ("orc_j", jo)):
                    Lf, Uf = m3.fid_band_at(cs, M, jj, khat, n1)
                    arms[(nm, a)].append(
                        m3.band_row(Lf, Uf, rtrue, lm_idx))
                for nm in names:
                    jlog[a][nm].append(rescale(jm[nm][a][0], m_in, M))
                    jlog[a][nm + "_q"].append(rescale(jm[nm][a][1], m_in, M))
                jlog[a]["brk_raw"].append(jraw)
                jlog[a]["brk_sm"].append(jsm)
                jlog[a]["brk_all"].append(jall)
                jlog[a]["brk_raw_q"].append(jrq)
                jlog[a]["brk_sm_q"].append(jsq)
                jlog[a]["cp"].append(jc)
                jlog[a]["rc"].append(jr)
            del cs
            if r % 5 == 4:
                print(f"  [{cname}] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
                      flush=True)

        cd = {"_meta": dict(spec, reps=reps, M=M, ncal=ncal, m_in=m_in,
                            ncal_oracle=ncal_or, set_level=set_level, B0=B0,
                            C=m3.CFID, true_auc=float(curve.auc()),
                            runtime_s=time.time() - t0),
              "oracle_j": {str(k): v for k, v in orc.items()},
              "arms": {f"{nm}|{a}": m3.summarize(v, tk, n0)
                       for (nm, a), v in arms.items()},
              "mean_j": {str(a): {nm: float(np.mean(v))
                                  for nm, v in jlog[a].items()}
                         for a in alphas},
              "j_raw": {str(a): {nm: v for nm, v in jlog[a].items()}
                        for a in alphas},
              "member_slope": {nm: v for nm, v in slope.items()}}
        res[cname] = cd
        print(json.dumps({cname: {"mean_j": cd["mean_j"],
                                  "arms": {k: {kk: vv for kk, vv in
                                               cd["arms"][k].items()
                                               if kk in ("coverage", "area")}
                                           for k in sorted(cd["arms"])}}},
                         indent=1, default=str), flush=True)
        _save(res, out_path)
    return res


# ---------------------------------------------------------------------------
# ordered-family premise check
# ---------------------------------------------------------------------------

def cap_draw_chunk(max_elems=8_000_000):
    """Shrink the fiducial-draw chunk so very large grids still fit in RAM.

    ``m2.fid_draws`` chunks at 1500 draws, which at n0 = 20,000 would allocate
    several GB of interpolation temporaries.  The chunk is capped by total
    (draws x grid) elements instead, leaving the harness file untouched.
    """
    orig = m2.fid_draws

    def wrapped(lab_s, n0, n1, M, tk, rng, chunk=1500):
        return orig(lab_s, n0, n1, M, tk, rng,
                    max(25, min(chunk, int(max_elems // max(len(tk), 1)))))

    m2.fid_draws = wrapped


def run_family(specs, n0, n1, reps, M, alphas, seed, out_path):
    """Per-curve calibration ceiling ae* for an ordered family of truths."""
    if n0 > 6000:
        cap_draw_chunk()
    res = {}
    for sp in specs:
        kind, val = sp.split(":")
        spec = (kind, float(val)) if kind != "kink" else ("kink", 0.004, 0.6)
        curve, _ = m2.build_truth(spec)
        name = f"{kind}{val}"
        tk = np.arange(n0 + 1) / n0
        print(f"== family {name} n0={n0} n1={n1} reps={reps} M={M} "
              f"AUC={curve.auc():.4f} ==", flush=True)
        prof, ladders, meta, tkk = m2.run_profile_cell(
            name, curve, n0, n1, reps, M,
            seed + n0 + sum(ord(c) for c in name))
        agg = m2.aggregate(prof["main"], ladders["main"], tkk, n0, alphas)
        y = curve.eval(tk)
        agg["_meta"] = dict(meta, truth=spec, name=name,
                            early_slope_05=float(y[int(round(0.05 * n0))]),
                            early_slope_01=float(y[int(round(0.01 * n0))]),
                            auc=float(curve.auc()))
        res[name] = agg
        print(json.dumps({name: {"recal": {a: {"ae_star": v.get("ae_star"),
                                               "mean_jstar": v.get("mean_jstar")}
                                           for a, v in agg["recal"].items()}}},
                         indent=1, default=str), flush=True)
        _save(res, out_path)
    return res


def _save(res, path):
    if path:
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"saved -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["bracket", "family"])
    ap.add_argument("--cells", nargs="+", default=["C2"])
    ap.add_argument("--specs", nargs="+",
                    default=["binormal:0.70", "binormal:0.80", "binormal:0.90",
                             "binormal:0.95", "binormal:0.99", "t2:0.95"])
    ap.add_argument("--n0", type=int, default=500)
    ap.add_argument("--n1", type=int, default=500)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--M", type=int, default=3000)
    ap.add_argument("--ncal", type=int, default=40)
    ap.add_argument("--min", dest="m_in", type=int, default=600)
    ap.add_argument("--ncalor", type=int, default=300)
    ap.add_argument("--B", type=int, default=100000)
    ap.add_argument("--setlevel", type=float, default=0.5)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.2])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.exp == "bracket":
        run_bracket(args.cells, args.reps, args.M, args.ncal, args.m_in,
                    args.ncalor, args.alphas, args.seed, args.out, args.B,
                    args.setlevel)
    else:
        run_family(args.specs, args.n0, args.n1, args.reps, args.M,
                   args.alphas, args.seed, args.out)


if __name__ == "__main__":
    main()
