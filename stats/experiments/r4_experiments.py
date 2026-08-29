"""Round-4 experiments for the rank-space fiducial ROC band.

What is left of the M4 programme after round 3 falsified its main relaxation
(`m3m4_report.md` §5), plus the round-3 loose ends that are not owned by the
separate C-calibration study (`stats/c_calibration_spec.md`).

Sub-experiments (``--exp``)
--------------------------
``fpcal``   **Fiducial-predictive trim calibration** (P1).  Instead of
            calibrating the trim depth against one plug-in curve (measured
            1.3-1.7x conservative, `m2_report.md` §1c) or against a worst case
            over a data-derived set (falsified in round 3), calibrate against
            draws from the fiducial cloud itself: per replicate draw ``ncal``
            candidate curves from the outer cloud, simulate one rank-space
            dataset from each, build an inner cloud plus the full production
            band ladder, and record whether it covers that candidate.  The
            calibrated depth is the largest depth whose coverage *averaged over
            candidates* reaches ``1-alpha`` -- i.e. the calibration is
            integrated over the fiducial predictive law rather than plugged in
            at a point.  A smoothed-candidate variant is run alongside (it
            carries a tuning constant, the window, and is flagged as such).
            Diagnostics: the depth contrast between a candidate's own min-p
            depth and its inner cloud's depth distribution, which is the
            mechanism that decides the outcome either way.

``rough``   **Rank-computable roughness functionals** (P2, theory doc §12.3 /
            `next_method_ideas.md` §8.4).  Round 3 showed the calibration
            ceiling ``ae*`` is flat along early slope and moves along a
            roughness-like axis.  This computes a battery of candidate
            functionals of the merged label sequence (run statistics, windowed
            local-slope variation, near-corner detectors, cloud rank-path
            crossing counts, plug-in depth contrast) on fresh replicates of 14
            cells, stores them per replicate together with the full
            coverage-vs-trim-level table, and lets `r4_analyze.py` fit a
            level rule on a subset of shapes and score it out-of-sample.

``exact``   **Exact Monte Carlo test at a named curve** (P3, the tractable
            fragment of M4).  ``H0: R = R0`` is simple in rank space, so the
            null law of any statistic is exactly simulable.  Statistic: the
            min-p depth of the observed empirical ROC in a *fixed independent*
            cloud of empirical ROCs simulated from ``R0``; its null
            distribution comes from a second independent null sample, which
            makes the Monte Carlo p-value exactly valid (conservative with
            ties, exact with the tie randomisation, both reported).

``repair``  **Steep-corner pointwise repair probe** (P5, round-3 still-open
            #5, which conjectured that M3's edges are *narrower* than the
            fiducial band's at the first interior grid points).  This
            intersects the fiducial band with M3(alpha2) restricted to grid
            points ``1 <= k <= kc`` (never ``k = 0``: pinning ``U(0) = 0`` is
            forbidden distribution-free, `fiducial_band_theory.md`
            Cor. 9.3), with ``alpha2`` spent by union bound, and measures the
            coverage/area/landmark-width change against the plain band --
            including a matched-realised-coverage comparison read off the
            trim-depth ladder, which is the only fair width comparison.

``corner``  The diagnostic behind ``repair``: how tight M3 actually is,
            relative to the fiducial band, on the first ``kc`` interior grid
            points, as a function of M3's nominal level.  It answers *why* the
            repair does or does not bite, and prices the union bound that would
            be needed for it to bite at all.

``m3grid``  **M3's nominal->actual level map** (P4) on the shapes and sizes
            round 3 did not cover, over a level ladder refined where the "which
            nominal level realises 95% coverage" question lives.  This is a
            thin wrapper around the unmodified ``m3_experiments.run_m3grid``.

Everything imports the published harnesses and does not modify them.
"""

import argparse
import json
import time

import numpy as np
import torch

import m2_experiments as m2
import m3_experiments as m3
import m4_experiments as m4
import rank_band_experiments as rbe

TOL = rbe.TOL
LANDMARKS = rbe.LANDMARKS
AE_GRID = m2.AE_GRID


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _save(res, path):
    if path:
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"saved -> {path}", flush=True)


def khat_of(u, w, n1):
    return np.rint(rbe.rhat_batch(u[None, :], w[None, :])[0] * n1).astype(np.int64)


def smooth_window(n0):
    """The moving-average window used for every smoothed variant here."""
    return max(3, int(round(np.sqrt(n0))) | 1)


# ===========================================================================
# P1 -- fiducial-predictive trim calibration
# ===========================================================================

def fp_calibrate(cands, n0, n1, tk, ladder, cp_up, m_in, rng, alphas):
    """Calibrate the trim depth against draws from the fiducial predictive law.

    ``cands`` (ncal, K) are candidate curves on the grid.  For each candidate
    one rank-space dataset is simulated from it, an inner cloud of ``m_in``
    fiducial draws is built, and the whole production band ladder is scored for
    coverage *of that candidate*.  Averaging the coverage indicator over
    candidates integrates the calibration over the predictive law.

    Returns ``({alpha: (j_thresh, j_quant)}, S_candidate, inner_depth_quantiles,
    coverage_ladder)`` with depths on the ``m_in`` scale.
    """
    ncal = cands.shape[0]
    cov = np.zeros(len(ladder))
    S = np.empty(ncal, dtype=np.int64)
    QD = np.empty((ncal, 3))
    for c in range(ncal):
        y0 = np.ascontiguousarray(cands[c])
        uu = np.sort(rng.random(n0))
        ww = m4.sample_curve(tk, y0, n1, rng)
        _, _, ls = rbe.polyline_vertices(uu, ww)
        kh = khat_of(uu, ww, n1)
        Rc = rbe.fiducial_curves(ls, n0, n1, m_in, tk, rng)
        p = m2.rep_profile(Rc, y0, kh, ladder, cp_up)
        cov += p["cov"]
        S[c] = min(p["Slow"], p["Shigh"])
        QD[c] = np.quantile(p["s_sorted"], [0.05, 0.2, 0.5])
    cov /= ncal
    Ss = np.sort(S)
    out = {}
    for a in alphas:
        ok = np.nonzero(cov >= 1 - a - 1e-12)[0]
        jt = int(ladder[ok.max()]) if len(ok) else 1
        jq = int(max(1, Ss[min(int(np.floor(a * ncal)), ncal - 1)]))
        out[a] = (jt, jq)
    return out, S, QD, cov


def run_fpcal(cells, reps, M, ncal, m_in, alphas, seed, out_path, arms):
    res = {}
    for cname in cells:
        spec = m2.CELLS[cname]
        curve, _ = m2.build_truth(spec["truth"])
        n0, n1 = spec["n0"], spec["n1"]
        tk = np.arange(n0 + 1) / n0
        rtrue_k = curve.eval(tk)
        lm_idx = np.array([int(round(lm * n0)) for lm in LANDMARKS])
        ladder = m2.make_ladder(M)
        cu, _ = m2.cp_tables(ladder, M, n1)
        ladder_in = m2.make_ladder(m_in)
        cu_in, _ = m2.cp_tables(ladder_in, m_in, n1)
        wsm = smooth_window(n0)
        rng = np.random.default_rng(seed + sum(ord(c) for c in cname))
        U_data, W_data = m2.gen_rep_data(curve, n0, n1, reps, rng)

        print(f"== fpcal {cname} {spec['truth']} n0={n0} n1={n1} reps={reps} "
              f"M={M} ncal={ncal} m_in={m_in} arms={arms} ==", flush=True)

        prof = []
        jsel = {(arm, kind): [] for arm in arms for kind in ("t", "q")}
        jval = {(arm, kind): [] for arm in arms for kind in ("t", "q")}
        Slog = {arm: [] for arm in arms}
        QDlog = {arm: [] for arm in arms}
        t0 = time.time()
        for r in range(reps):
            u, w = U_data[r], W_data[r]
            _, _, lab_s = rbe.polyline_vertices(u, w)
            khat = khat_of(u, w, n1)
            cands = rbe.fiducial_curves(lab_s, n0, n1, ncal, tk, rng)
            for arm in arms:
                if arm == "raw":
                    C = cands
                else:
                    C = np.stack([m4.smooth_curve(cands[i], wsm)
                                  for i in range(ncal)])
                cal, S, QD, _ = fp_calibrate(C, n0, n1, tk, ladder_in, cu_in,
                                             m_in, rng, alphas)
                Slog[arm].append(S.tolist())
                QDlog[arm].append(QD.mean(axis=0).tolist())
                for kind, pos in (("t", 0), ("q", 1)):
                    row_j, row_i = [], []
                    for a in alphas:
                        j_out = m4.rescale(cal[a][pos], m_in, M)
                        row_j.append(j_out)
                        row_i.append(int(np.clip(
                            np.searchsorted(ladder, j_out, side="right") - 1,
                            0, len(ladder) - 1)))
                    jval[(arm, kind)].append(row_j)
                    jsel[(arm, kind)].append(row_i)
            R = m2.fid_draws(lab_s, n0, n1, M, tk, rng)
            prof.append(m2.rep_profile(R, rtrue_k, khat, ladder, cu,
                                       lm_idx=lm_idx))
            del R
            if r % 10 == 9:
                print(f"  [{cname}] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
                      flush=True)

        cd = m2.aggregate(prof, ladder, tk, n0, alphas)
        for arm in arms:
            for kind in ("t", "q"):
                sel = np.array(jsel[(arm, kind)])
                jv = np.array(jval[(arm, kind)])
                key = f"fid_pred_{arm}" + ("" if kind == "t" else "_q")
                cd[key] = {}
                for i, a in enumerate(alphas):
                    d = m2.summarize_sel(prof, sel[:, i], tk, n0)
                    d["mean_jstar"] = float(jv[:, i].mean())
                    d["med_jstar"] = float(np.median(jv[:, i]))
                    cd[key][str(a)] = d
        # candidate-vs-inner-cloud depth contrast: what the calibration sees
        cd["cand_depth"] = {}
        for arm in arms:
            S = np.concatenate([np.asarray(x) for x in Slog[arm]])
            QD = np.array(QDlog[arm])
            cd["cand_depth"][arm] = {
                "S_cand_q": {str(q): float(np.quantile(S, q))
                             for q in (0.05, 0.2, 0.5)},
                "S_inner_draw_q": {"0.05": float(QD[:, 0].mean()),
                                   "0.2": float(QD[:, 1].mean()),
                                   "0.5": float(QD[:, 2].mean())},
                "mean_S_cand": float(S.mean()),
            }
        cd["_meta"] = dict(spec, reps=reps, M=M, ncal=ncal, m_in=m_in,
                           arms=list(arms), smooth_window=wsm,
                           true_auc=float(curve.auc()),
                           runtime_s=time.time() - t0)
        res[cname] = cd
        keys = ["fid_cp", "fid_rc", "recal"] + [
            f"fid_pred_{arm}{sfx}" for arm in arms for sfx in ("", "_q")]
        print(json.dumps({cname: {k: {a: {kk: round(vv, 4)
                                         for kk, vv in cd[k][a].items()
                                         if kk in ("coverage", "area",
                                                   "mean_jstar", "ae_star")}
                                     for a in cd[k]} for k in keys}},
                         indent=1, default=str), flush=True)
        _save(res, out_path)
    return res


# ===========================================================================
# P2 -- rank-computable roughness functionals
# ===========================================================================

def run_stats(lab_s, n0, n1):
    """Label-sequence run statistics (rank-computable, O(N))."""
    lab = lab_s.astype(np.int64)
    ch = np.nonzero(np.diff(lab))[0]
    runs = len(ch) + 1
    N = n0 + n1
    mu = 1.0 + 2.0 * n0 * n1 / N
    var = 2.0 * n0 * n1 * (2.0 * n0 * n1 - N) / (N * N * (N - 1.0))
    z = (runs - mu) / np.sqrt(var)
    bnd = np.concatenate([[-1], ch, [N - 1]])
    lens = np.diff(bnd)
    p = np.bincount(lens)
    p = p[p > 0] / runs
    ent = float(-(p * np.log(p)).sum())
    return dict(runs_z=float(z), run_ent=ent,
                run_maxlen=float(lens.max() / N))


def slope_stats(y, n0, wpts):
    """Windowed local-slope statistics of a grid curve (window ~ n0^{-1/2}).

    ``slope_up*`` measure the *upward* part of the slope variation.  A binormal
    ROC has strictly decreasing slope, so for that whole family these are pure
    sampling noise; a truth with a mid-curve inflection (bimodal) or a
    near-corner followed by a steeper stretch (t(2)) has genuine up-turns.
    That is the concavity-defect axis, which is what separates the two
    off-family shapes from the binormal ladder in the round-3 ceiling table.
    """
    nb = max(2, n0 // wpts)
    idx = np.minimum(np.arange(nb + 1) * wpts, n0)
    dy = np.diff(y[idx])
    dt = np.diff(idx) / n0
    s = dy / dt
    sm = max(s.mean(), 1e-12)
    ds = np.diff(s)
    up = np.maximum(ds, 0.0).sum()
    return dict(slope_sd=float(s.std() / sm),
                slope_logsd=float(np.std(np.log(s + 0.05))),
                slope_max=float(s.max() / sm),
                slope_curv=float(np.abs(np.diff(s)).mean() / sm),
                slope_up=float(up / max(np.abs(ds).sum(), 1e-12)),
                slope_upmag=float(up / sm),
                slope_nup=float((ds > 0).mean()))


def lcm(y, t):
    """Least concave majorant of the grid curve ``(t, y)`` (upper convex hull).

    The ROC of any DGP whose likelihood ratio is monotone is concave, and the
    whole binormal family is; the L1 gap between the empirical curve and its
    own LCM is therefore a rank-computable *concavity defect*, zero up to
    sampling noise on a concave truth and systematically positive otherwise.
    """
    hull = [0]
    for i in range(1, len(y)):
        while len(hull) >= 2:
            i0, i1 = hull[-2], hull[-1]
            # drop i1 if it lies below the chord (i0, i)
            if ((y[i1] - y[i0]) * (t[i] - t[i0])
                    <= (y[i] - y[i0]) * (t[i1] - t[i0]) + 1e-15):
                hull.pop()
            else:
                break
        hull.append(i)
    h = np.array(hull)
    return np.interp(t, t[h], y[h])


def lcm_stats(y, t, tag):
    g = lcm(y, t) - y
    return {f"lcm_gap_{tag}": float(g.mean()), f"lcm_max_{tag}": float(g.max())}


def path_stats(cs, M, y):
    """Rank path of a curve through the cloud, and its crossing count.

    Under the erosion law (`fiducial_band_theory.md` §7.1) the coverage
    exponent is the ratio of *effective independent looks* of a draw to those
    of the truth.  The number of times a curve's local-rank path through the
    cloud crosses the cloud median is a direct geometric proxy for that count,
    and it is computable from ranks alone (with a plug-in stand-in for the
    truth).
    """
    q = torch.from_numpy(np.ascontiguousarray(y))[:, None]
    a = torch.searchsorted(cs, q, right=True).squeeze(1)
    b = M - torch.searchsorted(cs, q, right=False).squeeze(1)
    S = int(torch.minimum(a, b).min())
    u = a.numpy() / M - 0.5
    sg = np.sign(u)
    sg = sg[sg != 0]
    x = int((sg[1:] != sg[:-1]).sum()) if len(sg) > 1 else 0
    return S, x


def draw_path_stats(cs, M, R, rows):
    """Crossing counts for a subsample of the cloud's own draws."""
    q = torch.from_numpy(np.ascontiguousarray(R[rows].T))
    a = torch.searchsorted(cs, q, right=True).numpy() / M - 0.5
    sg = np.sign(a)
    out = []
    for j in range(a.shape[1]):
        s = sg[:, j]
        s = s[s != 0]
        out.append(int((s[1:] != s[:-1]).sum()) if len(s) > 1 else 0)
    return float(np.mean(out))


def rough_profile(R, rtrue_k, khat, ladder, cp_up, lm_idx, y_plug, y_lcm,
                  n_path, rng):
    """One sort of the cloud; the band ladder, the depths, and the functionals."""
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
    s_sorted = np.sort(s.numpy())

    jl = torch.from_numpy((ladder - 1).astype(np.int64))
    ju = torch.from_numpy((M - ladder).astype(np.int64))
    L = np.clip(cs[:, jl].numpy().T, 0.0, 1.0)      # (J, K)
    U = np.clip(cs[:, ju].numpy().T, 0.0, 1.0)
    U = np.maximum.accumulate(np.maximum(U, cp_up[:, khat]), axis=1)
    d_lo = L - rtrue_k[None, :]
    d_hi = rtrue_k[None, :] - U
    viol = np.maximum(np.maximum(d_lo, d_hi), 0.0)
    depth = viol.max(axis=1)
    prof = {
        "vlow": (d_lo > TOL).any(axis=1),
        "vhigh": (d_hi > TOL).any(axis=1),
        "depth": depth,
        "worst_k": np.where(depth > TOL, viol.argmax(axis=1), -1),
        "area": (U - L).mean(axis=1),
        "w": (U - L)[:, lm_idx],
        "s_sorted": s_sorted,
    }
    prof["cov"] = ~(prof["vlow"] | prof["vhigh"])
    S_true, x_true = path_stats(cs, M, rtrue_k)
    prof["Slow"] = S_true          # min(a, b) already; kept for aggregate()
    prof["Shigh"] = S_true
    S_pl, x_pl = path_stats(cs, M, y_plug)
    S_lc, x_lc = path_stats(cs, M, y_lcm)
    rows = rng.choice(M, size=min(n_path, M), replace=False)
    x_dr = draw_path_stats(cs, M, R, rows)
    q = np.quantile(s_sorted, [0.05, 0.2, 0.5])
    func = dict(
        xing_plug=float(x_pl), xing_lcm=float(x_lc), xing_draw=float(x_dr),
        xing_ratio_plug=float(x_dr / max(x_pl, 1.0)),
        xing_ratio_lcm=float(x_dr / max(x_lc, 1.0)),
        dc05_plug=float(S_pl / max(q[0], 1.0)),
        dc50_plug=float(S_pl / max(q[2], 1.0)),
        dc05_lcm=float(S_lc / max(q[0], 1.0)),
        dc50_lcm=float(S_lc / max(q[2], 1.0)),
        S_plug=float(S_pl), S_lcm=float(S_lc),
    )
    diag = dict(S_true=float(S_true), xing_true=float(x_true),
                q05_draw=float(q[0]), q50_draw=float(q[2]))
    return prof, func, diag


R4_CELLS = {
    # fitting set: the 7 shapes of res_m4_family.json at n = 500
    "F_b70": dict(truth=("binormal", 0.70), n0=500, n1=500),
    "F_b80": dict(truth=("binormal", 0.80), n0=500, n1=500),
    "F_b90": dict(truth=("binormal", 0.90), n0=500, n1=500),
    "F_b95": dict(truth=("binormal", 0.95), n0=500, n1=500),
    "F_b99": dict(truth=("binormal", 0.99), n0=500, n1=500),
    "F_t295": dict(truth=("t2", 0.95), n0=500, n1=500),
    "F_bim90": dict(truth=("bimodal", 0.90), n0=500, n1=500),
    # held-out shapes at n = 500 (published ae* from m2_report.md §1a)
    "H_b55": dict(truth=("binormal", 0.55), n0=500, n1=500),      # P2e
    "H_kink": dict(truth=("kink", 0.004, 0.6), n0=500, n1=500),   # P2f
    "H_imb91": dict(truth=("binormal", 0.90), n0=900, n1=100),    # P2a
    "H_imb19": dict(truth=("binormal", 0.90), n0=100, n1=900),    # P2b
    # n axis at (nearly) fixed shape
    "N_b95_150": dict(truth=("binormal", 0.95), n0=150, n1=150),   # C3
    "N_b95_2000": dict(truth=("binormal", 0.95), n0=2000, n1=2000),  # P4b
    "N_b90_25": dict(truth=("binormal", 0.90), n0=25, n1=25),      # C7
}


def _bits(x):
    return "".join("1" if v else "0" for v in x)


def _floats(x, nd=5):
    return ",".join(f"{v:.{nd}f}" for v in x)


def run_rough(cells, reps, M, alphas, seed, out_path, n_path):
    res = {}
    for cname in cells:
        spec = R4_CELLS[cname]
        curve, _ = m2.build_truth(spec["truth"])
        n0, n1 = spec["n0"], spec["n1"]
        tk = np.arange(n0 + 1) / n0
        rtrue_k = curve.eval(tk)
        lm_idx = np.array([int(round(lm * n0)) for lm in LANDMARKS])
        ladder = m2.make_ladder(M)
        cu, _ = m2.cp_tables(ladder, M, n1)
        wsm = smooth_window(n0)
        wpts = max(2, int(round(np.sqrt(n0))))
        rng = np.random.default_rng(seed + sum(ord(c) for c in cname))
        U_data, W_data = m2.gen_rep_data(curve, n0, n1, reps, rng)
        print(f"== rough {cname} {spec['truth']} n0={n0} n1={n1} reps={reps} "
              f"M={M} wsm={wsm} wpts={wpts} ==", flush=True)

        prof, funcs, diags = [], [], []
        t0 = time.time()
        for r in range(reps):
            u, w = U_data[r], W_data[r]
            xs, ys, lab_s = rbe.polyline_vertices(u, w)
            khat = khat_of(u, w, n1)
            r0_k = np.interp(tk, xs, ys)                 # staircase-upper
            xs_h, ys_h = rbe.hazen_polyline(lab_s, n0, n1)
            y_plug = np.interp(tk, xs_h, ys_h)
            y_lcm = lcm(y_plug, tk)
            R = m2.fid_draws(lab_s, n0, n1, M, tk, rng)
            p, f, d = rough_profile(R, rtrue_k, khat, ladder, cu, lm_idx,
                                    y_plug, y_lcm, n_path, rng)
            del R
            f.update(run_stats(lab_s, n0, n1))
            for pref, yy in (("emp", r0_k), ("plug", y_plug)):
                for k, v in slope_stats(yy, n0, wpts).items():
                    f[f"{k}_{pref}"] = v
                f.update(lcm_stats(yy, tk, pref))
            prof.append(p)
            funcs.append(f)
            diags.append(d)
            if r % 50 == 49:
                print(f"  [{cname}] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
                      flush=True)

        cd = m2.aggregate(prof, ladder, tk, n0, alphas)
        # per-rep coverage / area over the alpha_eff grid, so that a level rule
        # fitted afterwards can be scored per replicate without re-simulating.
        Msz = len(prof[0]["s_sorted"])
        J = len(ladder)
        cov_ae, area_ae, j_ae = [], [], []
        for p in prof:
            jj = np.clip(p["s_sorted"][np.floor(AE_GRID * Msz).astype(int)],
                         1, max(Msz // 2, 1))
            idx = np.clip(np.searchsorted(ladder, jj, side="right") - 1, 0, J - 1)
            cov_ae.append(_bits(p["cov"][idx]))
            area_ae.append(_floats(p["area"][idx]))
            j_ae.append(_floats(jj, 0))
        cd["per_rep"] = {"ae_grid": AE_GRID.tolist(), "cov_ae": cov_ae,
                         "area_ae": area_ae, "j_ae": j_ae}
        cd["funcs"] = {k: [float(f[k]) for f in funcs] for k in funcs[0]}
        cd["diags"] = {k: [float(d[k]) for d in diags] for k in diags[0]}
        # the same shape functionals evaluated on the TRUE curve: an oracle
        # reference that says whether the feature the rank functional is trying
        # to see is actually present in the truth.
        tf = slope_stats(rtrue_k, n0, wpts)
        tf.update(lcm_stats(rtrue_k, tk, "true"))
        cd["_meta"] = dict(spec, reps=reps, M=M, smooth_window=wsm,
                           slope_window_pts=wpts, n_path=n_path,
                           true_auc=float(curve.auc()), truth_funcs=tf,
                           runtime_s=time.time() - t0)
        res[cname] = cd
        print(json.dumps({cname: {
            "recal": {a: cd["recal"][a].get("ae_star") for a in cd["recal"]},
            "func_mean": {k: round(float(np.mean(v)), 4)
                          for k, v in cd["funcs"].items()},
            "diag_mean": {k: round(float(np.mean(v)), 3)
                          for k, v in cd["diags"].items()}}},
            indent=1, default=str), flush=True)
        _save(res, out_path)
    return res


# ===========================================================================
# P3 -- exact Monte Carlo test at a named curve
# ===========================================================================

def emp_roc_batch(curve, n0, n1, B, rng):
    U = rbe.uniform_order_stats(B, n0, rng)
    W = curve.inv(rng.random((B, n1)))
    return rbe.rhat_batch(U, W)


def depth_vs_cloud(cs, B, Rq):
    """min-p depth of each row of ``Rq`` in the cloud whose sorted columns are
    ``cs`` (K, B).  The cloud is independent of ``Rq``, so no self-inclusion
    correction is needed and the statistic is a fixed functional of the curve."""
    q = torch.from_numpy(np.ascontiguousarray(Rq.T))
    a = torch.searchsorted(cs, q, right=True)
    b = B - torch.searchsorted(cs, q, right=False)
    return torch.minimum(a, b).min(dim=0).values.numpy()


def mc_pvalues(T_null_sorted, T_obs, rng):
    """Monte Carlo p-values against an independent null sample of the statistic.

    Small ``T`` is evidence against H0.  ``p_cons`` is the standard
    (conservative under ties) MC p-value; ``p_rand`` randomises the atom, which
    makes the test exactly level-alpha.
    """
    BN = len(T_null_sorted)
    lt = np.searchsorted(T_null_sorted, T_obs, side="left")
    le = np.searchsorted(T_null_sorted, T_obs, side="right")
    eq = le - lt
    p_cons = (1.0 + le) / (BN + 1.0)
    p_rand = (lt + rng.random(len(T_obs)) * (1.0 + eq)) / (BN + 1.0)
    return p_cons, p_rand


def local_perturb(curve, t1, kappa):
    """Alternative confined to the early-FPR region ``t <= t1``.

    ``R_alt(t) = R(t1) * (R(t)/R(t1))**kappa`` for ``t <= t1``, identity above.
    Monotone by construction, matches the null exactly at ``t1`` and above, so
    the deviation is genuinely local; ``kappa > 1`` pushes the corner down.
    """
    t = curve.t.copy()
    r = curve.r.copy()
    r1 = float(curve.eval(t1))
    m = t <= t1
    r[m] = r1 * np.clip(r[m] / max(r1, 1e-300), 0.0, 1.0) ** kappa
    return rbe.Curve(t, r)


EXACT_NULLS = {
    "b95_n150": (("binormal", 0.95), 150, 150),
    "b95_n500": (("binormal", 0.95), 500, 500),
    "t295_n500": (("t2", 0.95), 500, 500),
    "bim90_n500": (("bimodal", 0.90), 500, 500),
    "b99_n500": (("binormal", 0.99), 500, 500),
}


def run_exact(nulls, reps, BA, BN, alphas, seed, out_path, aucs, kappas, t1):
    res = {}
    for name in nulls:
        spec, n0, n1 = EXACT_NULLS[name]
        curve0, _ = m2.build_truth(spec)
        rng = np.random.default_rng(seed + sum(ord(c) for c in name))
        t0 = time.time()
        RA = emp_roc_batch(curve0, n0, n1, BA, rng)
        cs, _ = torch.sort(torch.from_numpy(np.ascontiguousarray(RA.T)), dim=1)
        del RA
        RN = emp_roc_batch(curve0, n0, n1, BN, rng)
        T_null = np.sort(depth_vs_cloud(cs, BA, RN))
        del RN
        print(f"== exact {name} {spec} n0={n0} n1={n1} BA={BA} BN={BN} "
              f"reps={reps} (T_null q: "
              f"{[int(np.quantile(T_null, q)) for q in (0.05, 0.2, 0.5)]}) ==",
              flush=True)

        alts = {"null": curve0}
        if spec[0] == "binormal":
            for a in aucs:
                if abs(a - spec[1]) > 1e-9:
                    alts[f"auc{a}"] = rbe.make_binormal(a)
            for k in kappas:
                alts[f"kappa{k}"] = local_perturb(curve0, t1, k)
        rows = {}
        for aname, cv in alts.items():
            Rt = emp_roc_batch(cv, n0, n1, reps, rng)
            T = depth_vs_cloud(cs, BA, Rt)
            pc, pr = mc_pvalues(T_null, T, rng)
            tkf = np.arange(n0 + 1) / n0
            dev = float(np.abs(cv.eval(tkf) - curve0.eval(tkf)).max())
            rows[aname] = {
                "auc": float(cv.auc()), "sup_dev": dev,
                "mean_T": float(T.mean()),
                **{f"rej_cons@{a}": float((pc <= a).mean()) for a in alphas},
                **{f"rej_rand@{a}": float((pr <= a).mean()) for a in alphas},
            }
            print(f"  {aname:10s} auc={rows[aname]['auc']:.4f} "
                  f"dev={dev:.4f} " +
                  " ".join(f"{a}:{rows[aname][f'rej_rand@{a}']:.3f}"
                           for a in alphas), flush=True)
        res[name] = {"_meta": dict(truth=spec, n0=n0, n1=n1, reps=reps, BA=BA,
                                  BN=BN, t1=t1,
                                  null_auc=float(curve0.auc()),
                                  runtime_s=time.time() - t0),
                     "T_null_q": {str(q): float(np.quantile(T_null, q))
                                  for q in (0.01, 0.05, 0.1, 0.2, 0.5)},
                     "rows": rows}
        _save(res, out_path)
    return res


# ===========================================================================
# P5 -- steep-corner pointwise repair probe
# ===========================================================================

def band_ladder(cs, M, ladder, khat, cp_up):
    jl = torch.from_numpy((ladder - 1).astype(np.int64))
    ju = torch.from_numpy((M - ladder).astype(np.int64))
    L = np.clip(cs[:, jl].numpy().T, 0.0, 1.0)
    U = np.clip(cs[:, ju].numpy().T, 0.0, 1.0)
    U = np.maximum.accumulate(np.maximum(U, cp_up[:, khat]), axis=1)
    return L, U


def score_ladder(L, U, rtrue_k, lm_idx):
    d_lo = L - rtrue_k[None, :]
    d_hi = rtrue_k[None, :] - U
    viol = np.maximum(np.maximum(d_lo, d_hi), 0.0)
    depth = viol.max(axis=1)
    return {"cov": ~((d_lo > TOL).any(axis=1) | (d_hi > TOL).any(axis=1)),
            "depth": depth, "area": (U - L).mean(axis=1),
            "w": (U - L)[:, lm_idx]}


def run_repair(cells, reps, M, alphas, a2_fracs, kcs, seed, out_path, B):
    res = {}
    for cname in cells:
        spec = m2.CELLS[cname]
        curve, _ = m2.build_truth(spec["truth"])
        n0, n1 = spec["n0"], spec["n1"]
        tk = np.arange(n0 + 1) / n0
        rtrue_k = curve.eval(tk)
        lm_idx = np.array([int(round(lm * n0)) for lm in LANDMARKS])
        ladder = m2.make_ladder(M)
        cu, _ = m2.cp_tables(ladder, M, n1)
        kc_list = sorted({int(k) if k >= 1 else max(1, int(round(k * n0)))
                          for k in kcs})
        a2_list = sorted({round(a * f, 6) for a in alphas for f in a2_fracs})
        B0 = B if n0 <= 700 else max(20000, B // 4)
        B1 = B if n1 <= 700 else max(20000, B // 4)
        t0 = time.time()
        plans, info = m3.make_plans(n0, n1, np.array(a2_list), B0, B1, "sidak")
        rng = np.random.default_rng(seed + sum(ord(c) for c in cname))
        U_data, W_data = m2.gen_rep_data(curve, n0, n1, reps, rng)
        print(f"== repair {cname} {spec['truth']} n0={n0} n1={n1} reps={reps} "
              f"M={M} a2={a2_list} kc={kc_list} calib {time.time()-t0:.0f}s ==",
              flush=True)

        # "mono" additionally applies the free monotone tightening that a
        # monotone estimand permits (U(t) := min_{s>=t} U(s), mirrored below),
        # which lets the corner cap propagate; "plain" keeps the repair strictly
        # local so the effect at the first interior grid points is isolated.
        configs = ["base"] + [f"{a2}|{kc}|{mo}" for a2 in a2_list
                              for kc in kc_list for mo in ("plain", "mono")]
        acc = {c: {"cov": [], "area": [], "w": [], "depth": []}
               for c in configs}
        sdep = []
        t0 = time.time()
        for r in range(reps):
            u, w = U_data[r], W_data[r]
            _, _, lab_s = rbe.polyline_vertices(u, w)
            khat = khat_of(u, w, n1)
            pcnt = m3.pcnt_from_khat(khat, n0)
            m3b = {a2: plans[a2].band(pcnt) for a2 in a2_list}
            R = m2.fid_draws(lab_s, n0, n1, M, tk, rng)
            cs, s_sorted = m3.fid_sorted_and_depths(R)
            del R
            sdep.append(s_sorted)
            L, U = band_ladder(cs, M, ladder, khat, cu)
            del cs
            for c in configs:
                if c == "base":
                    sc = score_ladder(L, U, rtrue_k, lm_idx)
                else:
                    a2s, kcs_, mo = c.split("|")
                    Lm, Um = m3b[float(a2s)]
                    kc = int(kcs_)
                    L2, U2 = L.copy(), U.copy()
                    sl = slice(1, kc + 1)
                    L2[:, sl] = np.maximum(L[:, sl], Lm[None, sl])
                    U2[:, sl] = np.minimum(U[:, sl], Um[None, sl])
                    if mo == "mono":
                        U2 = np.minimum.accumulate(U2[:, ::-1], axis=1)[:, ::-1]
                        L2 = np.maximum.accumulate(L2, axis=1)
                    sc = score_ladder(L2, U2, rtrue_k, lm_idx)
                for k in ("cov", "area", "w", "depth"):
                    acc[c][k].append(sc[k])
            if r % 50 == 49:
                print(f"  [{cname}] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
                      flush=True)

        Msz = M
        jsel = np.empty((reps, len(AE_GRID)), dtype=int)
        jval = np.empty((reps, len(AE_GRID)), dtype=int)
        J = len(ladder)
        for r in range(reps):
            jj = np.clip(sdep[r][np.floor(AE_GRID * Msz).astype(int)], 1,
                         max(Msz // 2, 1))
            jval[r] = jj
            jsel[r] = np.clip(np.searchsorted(ladder, jj, side="right") - 1,
                              0, J - 1)
        cd = {"_meta": dict(spec, reps=reps, M=M, B0=B0, B1=B1,
                            a2_list=a2_list, kc_list=kc_list,
                            true_auc=float(curve.auc()),
                            runtime_s=time.time() - t0),
              "ladder": ladder.tolist(), "ae_grid": AE_GRID.tolist(),
              "calib": {str(a): info[a] for a in a2_list}, "by": {}}
        for c in configs:
            cov = np.array(acc[c]["cov"])          # (reps, J)
            area = np.array(acc[c]["area"])
            dep = np.array(acc[c]["depth"])
            W = np.array(acc[c]["w"])              # (reps, J, nLM)
            blk = {"by_j": {"cov": cov.mean(axis=0).tolist(),
                            "area": area.mean(axis=0).tolist(),
                            "max_depth": dep.max(axis=0).tolist()}}
            bya = {"cov": [], "area": [], "max_depth": [], "mean_j": []}
            for i in range(len(AE_GRID)):
                idx = jsel[:, i]
                rr = np.arange(reps)
                bya["cov"].append(float(cov[rr, idx].mean()))
                bya["area"].append(float(area[rr, idx].mean()))
                bya["max_depth"].append(float(dep[rr, idx].max()))
                bya["mean_j"].append(float(jval[:, i].mean()))
            blk["by_ae"] = bya
            arms = {}
            for a in alphas:
                for tag, ae in (("C1", a), ("C2", 1.0 - (1.0 - a) ** 2)):
                    i = int(np.argmin(np.abs(AE_GRID - ae)))
                    idx = jsel[:, i]
                    rr = np.arange(reps)
                    arms[f"{a}|{tag}"] = {
                        "ae": float(AE_GRID[i]),
                        "coverage": float(cov[rr, idx].mean()),
                        "area": float(area[rr, idx].mean()),
                        "max_depth": float(dep[rr, idx].max()),
                        "mean_j": float(jval[:, i].mean()),
                        **{f"w{lm}": float(W[rr, idx, k].mean())
                           for k, lm in enumerate(LANDMARKS)}}
            blk["arms"] = arms
            cd["by"][c] = blk
        res[cname] = cd
        print(json.dumps({cname: {c: {k: round(v["coverage"], 4)
                                      for k, v in cd["by"][c]["arms"].items()}
                                  for c in configs}}, indent=1, default=str),
              flush=True)
        print(json.dumps({cname: {c: {k: round(v["area"], 5)
                                      for k, v in cd["by"][c]["arms"].items()}
                                  for c in configs}}, indent=1, default=str),
              flush=True)
        _save(res, out_path)
    return res


# ---------------------------------------------------------------------------

def run_corner(cells, reps, M, alphas, kc, seed, out_path, B):
    """Why the P5 repair does or does not bite: how tight M3 is, relative to
    the fiducial band, on the first ``kc`` interior grid points.

    Reports the width ratio in that window and the *nominal M3 level at which
    M3 first becomes the tighter band there* -- the level the union bound would
    have to pay for the repair to change anything.
    """
    levels = np.array([0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
                       0.2, 0.1, 0.05, 0.02, 0.0125, 0.005])
    res = {}
    for cname in cells:
        spec = m2.CELLS[cname]
        curve, _ = m2.build_truth(spec["truth"])
        n0, n1 = spec["n0"], spec["n1"]
        tk = np.arange(n0 + 1) / n0
        rng = np.random.default_rng(seed + sum(ord(c) for c in cname))
        B0 = B if n0 <= 700 else max(20000, B // 4)
        B1 = B if n1 <= 700 else max(20000, B // 4)
        plans, _ = m3.make_plans(n0, n1, levels, B0, B1, "sidak")
        kcw = min(int(kc), n0 - 1)
        sl = slice(1, kcw + 1)
        U_data, W_data = m2.gen_rep_data(curve, n0, n1, reps, rng)
        print(f"== corner {cname} {spec['truth']} n0={n0} reps={reps} "
              f"window k=1..{kcw} ==", flush=True)
        acc = {}
        for r in range(reps):
            u, w = U_data[r], W_data[r]
            _, _, lab_s = rbe.polyline_vertices(u, w)
            khat = khat_of(u, w, n1)
            pcnt = m3.pcnt_from_khat(khat, n0)
            R = m2.fid_draws(lab_s, n0, n1, M, tk, rng)
            cs, s_sorted = m3.fid_sorted_and_depths(R)
            del R
            for a in alphas:
                for tag, ae in (("C1", a), ("C2", 1.0 - (1.0 - a) ** 2)):
                    j = m3.trim_depth(s_sorted, ae, M)
                    Lf, Uf = m3.fid_band_at(cs, M, j, khat, n1)
                    wf = (Uf - Lf)[sl]
                    for lv in levels:
                        Lm, Um = plans[lv].band(pcnt)
                        wm = (Um - Lm)[sl]
                        k = f"{a}|{tag}|{lv}"
                        d = acc.setdefault(k, {"wratio": [], "tight_any": [],
                                               "tight_frac": [],
                                               "up_tight": [], "lo_tight": [],
                                               "minratio": []})
                        d["wratio"].append(float(np.mean(wm / np.maximum(wf, 1e-12))))
                        d["minratio"].append(float(np.min(wm / np.maximum(wf, 1e-12))))
                        t = (Um[sl] < Uf[sl] - TOL) | (Lm[sl] > Lf[sl] + TOL)
                        d["tight_any"].append(float(t.any()))
                        d["tight_frac"].append(float(t.mean()))
                        d["up_tight"].append(float((Um[sl] < Uf[sl] - TOL).mean()))
                        d["lo_tight"].append(float((Lm[sl] > Lf[sl] + TOL).mean()))
            del cs
            if r % 25 == 24:
                print(f"  [{cname}] rep {r+1}/{reps}", flush=True)
        res[cname] = {"_meta": dict(spec, reps=reps, M=M, kc=kcw,
                                    levels=levels.tolist(), B0=B0, B1=B1),
                      "stats": {k: {kk: float(np.mean(vv))
                                    for kk, vv in v.items()}
                                for k, v in acc.items()}}
        for a in alphas:
            for tag in ("C1", "C2"):
                print(f"  alpha={a} {tag}: "
                      + " ".join(
                          f"{lv:g}:{res[cname]['stats'][f'{a}|{tag}|{lv}']['wratio']:.2f}"
                          f"/{res[cname]['stats'][f'{a}|{tag}|{lv}']['tight_any']:.2f}"
                          for lv in levels), flush=True)
        _save(res, out_path)
    return res


# ===========================================================================
# P4 -- M3 nominal->actual map on the shapes and sizes round 3 did not cover
# ===========================================================================

# round 3's ladder (`m3_experiments.LEVEL_GRID`) refined between 0.3 and 1.0,
# where the "which nominal level realises 95% coverage" question lives.
LEVEL_GRID_R4 = np.array(sorted(
    set(m3.LEVEL_GRID.tolist())
    | {0.975, 0.925, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35}, reverse=True))


def run_m3grid(cells, reps, seed, out_path, B):
    return m3.run_m3grid(cells, reps, seed, out_path, B, LEVEL_GRID_R4,
                         ["sidak"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["fpcal", "rough", "exact", "repair", "m3grid",
                             "corner"])
    ap.add_argument("--cells", nargs="+", default=["C2"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--M", type=int, default=3000)
    ap.add_argument("--ncal", type=int, default=60)
    ap.add_argument("--min", dest="m_in", type=int, default=1000)
    ap.add_argument("--arms", nargs="+", default=["raw", "sm"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.2, 0.05])
    ap.add_argument("--npath", type=int, default=64)
    ap.add_argument("--BA", type=int, default=4000)
    ap.add_argument("--BN", type=int, default=4000)
    ap.add_argument("--aucs", nargs="+", type=float,
                    default=[0.93, 0.94, 0.96, 0.97])
    ap.add_argument("--kappas", nargs="+", type=float,
                    default=[0.6, 0.8, 1.25, 1.6])
    ap.add_argument("--t1", type=float, default=0.05)
    ap.add_argument("--a2frac", nargs="+", type=float, default=[0.1, 0.25])
    ap.add_argument("--kc", nargs="+", type=float, default=[10, 25, 0.02])
    ap.add_argument("--B", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.exp == "fpcal":
        run_fpcal(args.cells, args.reps, args.M, args.ncal, args.m_in,
                  args.alphas, args.seed, args.out, tuple(args.arms))
    elif args.exp == "rough":
        run_rough(args.cells, args.reps, args.M, args.alphas, args.seed,
                  args.out, args.npath)
    elif args.exp == "corner":
        run_corner(args.cells, args.reps, args.M, args.alphas, args.kc[0],
                   args.seed, args.out, args.B)
    elif args.exp == "m3grid":
        run_m3grid(args.cells, args.reps, args.seed, args.out, args.B)
    elif args.exp == "exact":
        run_exact(args.cells, args.reps, args.BA, args.BN, args.alphas,
                  args.seed, args.out, args.aucs, args.kappas, args.t1)
    else:
        run_repair(args.cells, args.reps, args.M, args.alphas, args.a2frac,
                   args.kc, args.seed, args.out, args.B)


if __name__ == "__main__":
    main()
