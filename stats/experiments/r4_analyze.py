"""Tables for the round-4 experiments (`r4_experiments.py`, `r4_report.md`).

Usage (from stats/experiments):
    U="uv run --project /home/nathan/Documents/studroc_paper python"
    $U r4_analyze.py fpcal  res_r4_fpcal_C2.json res_r4_fpcal_C5.json
    $U r4_analyze.py rough  res_r4_rough.json
    $U r4_analyze.py m3grid res_r4_m3grid.json res_m3_p1grid.json
    $U r4_analyze.py repair res_r4_repair.json

Published comparison columns are read from the round-2/3 result JSONs, never
retyped from the reports.
"""

import json
import sys

import numpy as np

# published fiducial-band results, per cell: (file, key)
PUB = {
    "C1": ("res_p1diag_a.json", "C1"), "C2": ("res_p1diag_a.json", "C2"),
    "C4": ("res_p1diag_b.json", "C4"), "C5": ("res_p1diag_b.json", "C5"),
    "C3": ("res_p2_a.json", "C3"), "C7": ("res_p2_a.json", "C7"),
    "P2b": ("res_p2_a.json", "P2b"), "P2d": ("res_p2_a.json", "P2d"),
    "P2e": ("res_p2_a.json", "P2e"), "P2a": ("res_p2_b.json", "P2a"),
    "P2c": ("res_p2_b.json", "P2c"), "P2f": ("res_p2_b.json", "P2f"),
    "P4a": ("res_p4_ab.json", "P4a"), "P4b": ("res_p4_ab.json", "P4b"),
    "P4c": ("res_p4_c.json", "P4c"),
}
FAMILY = "res_m4_family.json"

# round-4 `rough` cells -> the published cell whose ae* is the target
ROUGH_TARGET = {
    "F_b70": (FAMILY, "binormal0.70"), "F_b80": (FAMILY, "binormal0.80"),
    "F_b90": (FAMILY, "binormal0.90"), "F_b95": (FAMILY, "binormal0.95"),
    "F_b99": (FAMILY, "binormal0.99"), "F_t295": (FAMILY, "t20.95"),
    "F_bim90": (FAMILY, "bimodal0.90"),
    "H_b55": PUB["P2e"], "H_kink": PUB["P2f"],
    "H_imb91": PUB["P2a"], "H_imb19": PUB["P2b"],
    "N_b95_150": PUB["C3"], "N_b95_2000": PUB["P4b"], "N_b90_25": PUB["C7"],
}
FIT_CELLS = ["F_b70", "F_b80", "F_b90", "F_b95", "F_b99", "F_t295", "F_bim90"]
HELD_CELLS = ["H_b55", "H_kink", "H_imb91", "H_imb19"]
N_CELLS = ["N_b95_150", "N_b95_2000", "N_b90_25"]

_cache = {}


def load(f):
    if f not in _cache:
        with open(f) as fh:
            _cache[f] = json.load(fh)
    return _cache[f]


def blk(f, k):
    d = load(f)[k]
    return d.get("main", d)


def at_ae(b, ae, field="area"):
    """Value of a by_ae field at the grid point nearest ``ae``."""
    g = np.array(b["by_ae"]["ae"])
    i = int(np.argmin(np.abs(g - ae)))
    return b["by_ae"][field][i]


def pub_fid(cell, alpha):
    """(area, cov) of fid_cp (C=1) and fid_rc (C=2) at ``alpha``."""
    f, k = PUB[cell]
    b = blk(f, k)
    ae2 = 1.0 - (1.0 - alpha) ** 2
    return (at_ae(b, alpha, "area"), at_ae(b, alpha, "cov"),
            at_ae(b, ae2, "area"), at_ae(b, ae2, "cov"))


def pub_aestar(f, k, alpha):
    b = blk(f, k)
    r = b["recal"].get(str(alpha))
    return None if r is None else r.get("ae_star")


def cstar(ae, alpha):
    return None if ae is None else np.log1p(-ae) / np.log1p(-alpha)


# ---------------------------------------------------------------------------

def fpcal(files):
    for f in files:
        d = load(f)
        for cell, cd in d.items():
            m = cd["_meta"]
            print(f"\n=== fpcal {cell}  {m['truth']} n0={m['n0']} n1={m['n1']} "
                  f"reps={m['reps']} M={m['M']} ncal={m['ncal']} "
                  f"m_in={m['m_in']} ({m['runtime_s']:.0f}s) ===")
            arms = ["fid_cp", "fid_rc", "fid_pred_raw", "fid_pred_raw_q",
                    "fid_pred_sm", "fid_pred_sm_q", "recal"]
            hdr = f"{'arm':<15}" + "".join(
                f"{'cov@' + a:>10}{'area':>9}{'j':>8}" for a in cd["fid_cp"])
            print(hdr)
            base = {}
            for arm in arms:
                row = f"{arm:<15}"
                for a in cd["fid_cp"]:
                    v = cd[arm][a]
                    row += (f"{v['coverage']:>10.3f}{v['area']:>9.4f}"
                            f"{v['mean_jstar']:>8.1f}")
                    base.setdefault(a, {})[arm] = v
                print(row)
            print(f"{'ae*':<15}" + "".join(
                f"{cd['recal'][a].get('ae_star', float('nan')):>27.3f}"
                for a in cd["fid_cp"]))
            print("conservatism in j vs the recal ceiling "
                  "(ceiling / calibrated):")
            for a in cd["fid_cp"]:
                jc = base[a]["recal"]["mean_jstar"]
                s = "  ".join(
                    f"{arm.replace('fid_pred_', '')}: "
                    f"{jc / max(base[a][arm]['mean_jstar'], 1e-9):.2f}x"
                    for arm in ("fid_pred_raw", "fid_pred_sm", "fid_cp",
                                "fid_rc"))
                print(f"  alpha={a}: {s}")
            print("depth contrast:")
            ds = cd["depth_stats"]
            print(f"  outer cloud   S(truth) q05/q20/q50 = "
                  f"{ds['S_true_q']['0.05']:.1f}/{ds['S_true_q']['0.2']:.1f}/"
                  f"{ds['S_true_q']['0.5']:.1f}   vs S(draw) "
                  f"{ds['S_draw_q']['0.05']:.1f}/{ds['S_draw_q']['0.2']:.1f}/"
                  f"{ds['S_draw_q']['0.5']:.1f}")
            for arm, cdp in cd["cand_depth"].items():
                sc, sd = cdp["S_cand_q"], cdp["S_inner_draw_q"]
                print(f"  inner ({arm:>3s})   S(cand)  q05/q20/q50 = "
                      f"{sc['0.05']:.1f}/{sc['0.2']:.1f}/{sc['0.5']:.1f}"
                      f"   vs S(draw) {sd['0.05']:.1f}/{sd['0.2']:.1f}/"
                      f"{sd['0.5']:.1f}"
                      f"   ratio q05 {sc['0.05'] / max(sd['0.05'], 1e-9):.2f}"
                      f" q50 {sc['0.5'] / max(sd['0.5'], 1e-9):.2f}")


# ---------------------------------------------------------------------------

def _cov_at(cd, ae, rep=None):
    g = np.array(cd["per_rep"]["ae_grid"])
    i = int(np.argmin(np.abs(g - ae)))
    if rep is None:
        return np.mean([int(s[i]) for s in cd["per_rep"]["cov_ae"]])
    return int(cd["per_rep"]["cov_ae"][rep][i])


def _area_at(cd, ae, rep=None):
    g = np.array(cd["per_rep"]["ae_grid"])
    i = int(np.argmin(np.abs(g - ae)))
    A = cd["per_rep"]["area_ae"]
    if rep is None:
        return float(np.mean([float(s.split(",")[i]) for s in A]))
    return float(A[rep].split(",")[i])


def rough(files):
    d = load(files[0])
    cells = [c for c in list(ROUGH_TARGET) if c in d]
    keys = sorted(d[cells[0]]["funcs"])
    alphas = [0.5, 0.2, 0.05]

    print("=== cell inventory, targets and own ae* ===")
    print(f"{'cell':<12}{'n0':>6}{'n1':>6}{'reps':>6}  "
          + "".join(f"{'ae*pub@' + str(a):>12}{'ae*own':>9}" for a in alphas))
    tgt = {}
    for c in cells:
        m = d[c]["_meta"]
        row = f"{c:<12}{m['n0']:>6}{m['n1']:>6}{m['reps']:>6}  "
        for a in alphas:
            p = pub_aestar(*ROUGH_TARGET[c], a)
            o = d[c]["recal"][str(a)].get("ae_star")
            tgt.setdefault(a, {})[c] = p
            row += (f"{(p if p is not None else float('nan')):>12.3f}"
                    f"{(o if o is not None else float('nan')):>9.3f}")
        print(row)

    print("\n=== candidate functionals: cell means, and correlation with "
          "C*(published) over the 7 fitting cells ===")
    F = {k: np.array([np.mean(d[c]["funcs"][k]) for c in cells])
         for k in keys}
    fit_i = [cells.index(c) for c in FIT_CELLS if c in cells]
    scored = []
    for k in keys:
        row = []
        for a in alphas:
            y = np.array([cstar(tgt[a][cells[i]], a) for i in fit_i])
            x = F[k][fit_i]
            if np.std(x) < 1e-12 or np.any(np.isnan(y)):
                row.append(0.0)
                continue
            row.append(float(np.corrcoef(x, y)[0, 1]))
        # leave-one-out RMSE of the 1-predictor fit at alpha=.5 and .2
        loo = []
        for a in (0.5, 0.2):
            y = np.array([cstar(tgt[a][cells[i]], a) for i in fit_i])
            x = F[k][fit_i]
            err = []
            for j in range(len(x)):
                m_ = np.ones(len(x), bool)
                m_[j] = False
                A = np.c_[np.ones(m_.sum()), x[m_]]
                try:
                    b = np.linalg.lstsq(A, y[m_], rcond=None)[0]
                except np.linalg.LinAlgError:
                    err.append(np.nan)
                    continue
                err.append(y[j] - (b[0] + b[1] * x[j]))
            loo.append(float(np.sqrt(np.nanmean(np.square(err)))))
        scored.append((np.mean(loo), row, loo, k))
    scored.sort()
    print(f"{'functional':<20}{'r@.5':>7}{'r@.2':>7}{'r@.05':>7}"
          f"{'LOO@.5':>8}{'LOO@.2':>8}   cell means (fit | held | n-axis)")
    for sc, row, loo, k in scored:
        means = " ".join(f"{F[k][cells.index(c)]:.3g}"
                         for c in cells)
        print(f"{k:<20}{row[0]:>7.2f}{row[1]:>7.2f}{row[2]:>7.2f}"
              f"{loo[0]:>8.3f}{loo[1]:>8.3f}   {means}")
    # null model: predict the fitting-set mean C*
    for a in (0.5, 0.2):
        y = np.array([cstar(tgt[a][cells[i]], a) for i in fit_i])
        print(f"null model (constant) LOO RMSE at alpha={a}: "
              f"{np.sqrt(np.mean((y - y.mean()) ** 2)) * np.sqrt(len(y) / (len(y) - 1)):.3f}"
              f"   (C* range {y.min():.2f}-{y.max():.2f})")

    print("\n=== two-predictor fits (the 'corner steepness + concavity "
          "defect' hypothesis), leave-one-cell-out RMSE on the 7 fitting "
          "cells ===")
    pairs = []
    ks = [k for _, _, _, k in scored[:14]]
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            loo = []
            for a in (0.5, 0.2):
                y = np.array([cstar(tgt[a][cells[q]], a) for q in fit_i])
                X = np.c_[F[ks[i]][fit_i], F[ks[j]][fit_i]]
                err = []
                for q in range(len(y)):
                    m_ = np.ones(len(y), bool)
                    m_[q] = False
                    A = np.c_[np.ones(m_.sum()), X[m_]]
                    b = np.linalg.lstsq(A, y[m_], rcond=None)[0]
                    err.append(y[q] - (b[0] + b[1] * X[q, 0] + b[2] * X[q, 1]))
                loo.append(float(np.sqrt(np.mean(np.square(err)))))
            pairs.append((float(np.mean(loo)), loo, ks[i], ks[j]))
    pairs.sort()
    for sc, loo, k1, k2 in pairs[:6]:
        print(f"  {k1:<20}+ {k2:<20} LOO@.5={loo[0]:.3f} LOO@.2={loo[1]:.3f}")

    print("\n=== out-of-sample scoring of a functional-driven level rule ===")
    best = [k for _, _, _, k in scored[:4]]
    for k in best:
        print(f"\n-- rule: C* = a + b * {k}  (fitted on the 7 fitting cells) --")
        for a in alphas:
            y = np.array([cstar(tgt[a][cells[i]], a) for i in fit_i])
            x = F[k][fit_i]
            A = np.c_[np.ones(len(x)), x]
            b = np.linalg.lstsq(A, y, rcond=None)[0]
            print(f"  alpha={a}: C* = {b[0]:.3f} + {b[1]:.4g} * f")
            print(f"    {'cell':<12}{'set':>7}{'Chat':>7}{'ae_hat':>8}"
                  f"{'cov(rule,cellmean)':>20}{'cov(rule,per-rep)':>19}"
                  f"{'cov(C=2)':>10}{'cov(C=1)':>10}"
                  f"{'ar(rule)':>10}{'ar(C=2)':>10}")
            for c in cells:
                cd = d[c]
                fv = np.array(cd["funcs"][k])
                ch = float(np.clip(b[0] + b[1] * fv.mean(), 1.0, 5.0))
                ae = 1.0 - (1.0 - a) ** ch
                covc = _cov_at(cd, ae)
                arc = _area_at(cd, ae)
                cr, ar = [], []
                for r in range(len(fv)):
                    chr_ = float(np.clip(b[0] + b[1] * fv[r], 1.0, 5.0))
                    aer = 1.0 - (1.0 - a) ** chr_
                    cr.append(_cov_at(cd, aer, r))
                    ar.append(_area_at(cd, aer, r))
                grp = ("fit" if c in FIT_CELLS else
                       "held" if c in HELD_CELLS else "n")
                print(f"    {c:<12}{grp:>7}{ch:>7.2f}{ae:>8.3f}"
                      f"{covc:>20.3f}{np.mean(cr):>19.3f}"
                      f"{_cov_at(cd, 1 - (1 - a) ** 2):>10.3f}"
                      f"{_cov_at(cd, a):>10.3f}"
                      f"{arc:>10.4f}"
                      f"{_area_at(cd, 1 - (1 - a) ** 2):>10.4f}")
            for tag, fn in (("rule-cellmean", lambda c, ae: _cov_at(d[c], ae)),):
                pass
            # spreads
            def spread(sel, mode):
                v = []
                for c in sel:
                    cd = d[c]
                    fv = np.array(cd["funcs"][k])
                    if mode == "fix":
                        v.append(_cov_at(cd, 1 - (1 - a) ** 2))
                    elif mode == "cell":
                        ch = float(np.clip(b[0] + b[1] * fv.mean(), 1.0, 5.0))
                        v.append(_cov_at(cd, 1 - (1 - a) ** ch))
                    else:
                        cr = []
                        for r in range(len(fv)):
                            chr_ = float(np.clip(b[0] + b[1] * fv[r], 1.0, 5.0))
                            cr.append(_cov_at(cd, 1 - (1 - a) ** chr_, r))
                        v.append(float(np.mean(cr)))
                return min(v), max(v), max(v) - min(v)
            for name, sel in (("held-out", [c for c in HELD_CELLS if c in cells]),
                              ("all", cells)):
                s1, s2, s3 = spread(sel, "fix")
                t1, t2, t3 = spread(sel, "cell")
                u1, u2, u3 = spread(sel, "rep")
                print(f"    spread over {name:<9} C=2: {s1:.3f}-{s2:.3f} "
                      f"({s3 * 100:.1f}pp)   rule/cellmean: {t1:.3f}-{t2:.3f} "
                      f"({t3 * 100:.1f}pp)   rule/per-rep: {u1:.3f}-{u2:.3f} "
                      f"({u3 * 100:.1f}pp)")

    print("\n=== out-of-sample scoring of the best two-predictor rules ===")
    for _, _, k1, k2 in pairs[:2]:
        print(f"\n-- rule: C* = a + b*{k1} + c*{k2} --")
        for a in alphas:
            y = np.array([cstar(tgt[a][cells[q]], a) for q in fit_i])
            X = np.c_[F[k1][fit_i], F[k2][fit_i]]
            b = np.linalg.lstsq(np.c_[np.ones(len(y)), X], y, rcond=None)[0]
            covs = {"fix": [], "rule": [], "rep": []}
            names = []
            for c in cells:
                cd = d[c]
                f1 = np.array(cd["funcs"][k1])
                f2 = np.array(cd["funcs"][k2])
                ch = float(np.clip(b[0] + b[1] * f1.mean() + b[2] * f2.mean(),
                                   1.0, 5.0))
                cr = [_cov_at(cd, 1 - (1 - a) ** float(np.clip(
                    b[0] + b[1] * f1[r] + b[2] * f2[r], 1.0, 5.0)), r)
                    for r in range(len(f1))]
                names.append((c, ch, _cov_at(cd, 1 - (1 - a) ** ch),
                              float(np.mean(cr)),
                              _cov_at(cd, 1 - (1 - a) ** 2)))
            print(f"  alpha={a}: " + "  ".join(
                f"{c}:{ch:.2f}/{cc:.3f}" for c, ch, cc, _, _ in names))
            for lab, sel in (("held-out", HELD_CELLS), ("all", cells)):
                v = [(x[2], x[3], x[4]) for x in names if x[0] in sel]
                for i, tag in ((2, "C=2"), (0, "rule/cellmean"),
                               (1, "rule/per-rep")):
                    w = [q[i] for q in v]
                    print(f"    {lab:<9} {tag:<14} {min(w):.3f}-{max(w):.3f} "
                          f"({(max(w) - min(w)) * 100:.1f}pp)")

    print("\n=== co-movement check: within-cell Spearman(functional, realised "
          "truth depth S_true) and (functional, per-rep miss) ===")
    print(f"{'functional':<20}" + "".join(f"{c[:9]:>10}" for c in cells))
    for _, _, _, k in scored[:8]:
        row = f"{k:<20}"
        for c in cells:
            x = np.array(d[c]["funcs"][k])
            s = np.array(d[c]["diags"]["S_true"])
            rx = np.argsort(np.argsort(x))
            rs = np.argsort(np.argsort(s))
            row += f"{np.corrcoef(rx, rs)[0, 1]:>10.2f}"
        print(row)


# ---------------------------------------------------------------------------

def m3grid(files):
    cells, src = [], {}
    for f in files:
        for c in load(f):
            if c not in src:
                src[c] = f
                cells.append(c)
    order = ["C7", "P2d", "C3", "C1", "C2", "C4", "C5", "P2e", "P2f", "P2c",
             "P2a", "P2b", "P4b", "P4c"]
    cells = [c for c in order if c in cells] + [c for c in cells
                                                if c not in order]
    print("=== M3 nominal -> actual coverage map (sidak split) ===")
    lv = None
    rows = {}
    for c in cells:
        d = load(src[c])[c]
        m = d["_meta"]
        bl = d["sidak"]["by_level"]
        lv = [float(x) for x in d["sidak"]["levels"]]
        rows[c] = dict(meta=m, cov={float(a): bl[a]["coverage"] for a in bl},
                       area={float(a): bl[a]["area"] for a in bl},
                       maxd={float(a): bl[a]["max_depth"] for a in bl})
    print(f"{'cell':<6}{'truth':<16}{'n0/n1':>10}{'reps':>6}  "
          + "".join(f"{a:>7g}" for a in sorted(lv, reverse=True)))
    for c in cells:
        r = rows[c]
        m = r["meta"]
        print(f"{c:<6}{str(m['truth']):<16}"
              f"{str(m['n0']) + '/' + str(m['n1']):>10}{m['reps']:>6}  "
              + "".join(f"{r['cov'][a]:>7.3f}" for a in sorted(lv, reverse=True)))

    print("\n=== largest nominal alpha' attaining a coverage target ===")
    print(f"{'cell':<6}{'n':>7}{'a95':>8}{'a80':>8}{'a50':>8}")
    a95, a80 = {}, {}
    for c in cells:
        r = rows[c]
        n = min(r["meta"]["n0"], r["meta"]["n1"])
        got = {}
        for tgt, lab in ((0.95, "a95"), (0.80, "a80"), (0.50, "a50")):
            ok = [a for a in lv if r["cov"][a] >= tgt]
            got[lab] = max(ok) if ok else float("nan")
        a95[c], a80[c] = got["a95"], got["a80"]
        print(f"{c:<6}{n:>7}{got['a95']:>8.3f}{got['a80']:>8.3f}"
              f"{got['a50']:>8.3f}")

    print("\n=== infimum over shapes, by sample size ===")
    bysz = {}
    for c in cells:
        n = min(rows[c]["meta"]["n0"], rows[c]["meta"]["n1"])
        bysz.setdefault(n, []).append(c)
    for n in sorted(bysz):
        cc = bysz[n]
        print(f"  n={n:<6} cells={','.join(cc):<28} "
              f"inf a95={min(a95[c] for c in cc):.3f}  "
              f"inf a80={min(a80[c] for c in cc):.3f}")
    gi95 = min(a95.values())
    gi80 = min(a80.values())
    print(f"  inf over ALL measured cells: a95={gi95:.3f}  a80={gi80:.3f}")

    print("\n=== a fixed nominal alpha' applied to every cell: worst case "
          "over the measured library ===")
    print(f"{chr(945)}'   min cov   cells below .95   max area/fid_rc   "
          f"max area/fid_cp   mean area/fid_rc")
    for a in [x for x in sorted(lv, reverse=True) if 0.2 <= x <= 0.9]:
        cov = [rows[c]["cov"][a] for c in cells]
        rr, rc = [], []
        for c in cells:
            if c not in PUB:
                continue
            acp, _, arc, _ = pub_fid(c, 0.05)
            rr.append(rows[c]["area"][a] / arc)
            rc.append(rows[c]["area"][a] / acp)
        bad = [c for c in cells if rows[c]["cov"][a] < 0.95]
        print(f"{a:<5g}{min(cov):>8.3f}   {','.join(bad) if bad else '-':<17}"
              f"{max(rr):>17.2f}{max(rc):>18.2f}{np.mean(rr):>19.2f}")

    print(f"\n=== M3 area at the inf-over-shapes alpha' = {gi95:g} "
          f"vs the fiducial band at alpha=.05 ===")
    print(f"{'cell':<6}{'cov':>8}{'M3 area':>10}{'fid_cp':>9}{'fid_rc':>9}"
          f"{'xcp':>7}{'xrc':>7}   also at a'=0.6")
    for c in cells:
        r = rows[c]
        if c not in PUB:
            print(f"{c:<6}{r['cov'][gi95]:>8.3f}{r['area'][gi95]:>10.4f}"
                  f"{'-':>9}{'-':>9}")
            continue
        acp, ccp, arc, crc = pub_fid(c, 0.05)
        a6 = r["area"][0.6] if 0.6 in r["area"] else float("nan")
        print(f"{c:<6}{r['cov'][gi95]:>8.3f}{r['area'][gi95]:>10.4f}"
              f"{acp:>9.4f}{arc:>9.4f}"
              f"{r['area'][gi95] / acp:>7.2f}{r['area'][gi95] / arc:>7.2f}"
              f"   {a6:.4f} ({r['cov'][0.6]:.3f}) x{a6 / arc:.2f} vs rc")


# ---------------------------------------------------------------------------

def repair(files):
    for f in files:
        d = load(f)
        for cell, cd in d.items():
            m = cd["_meta"]
            print(f"\n=== repair {cell} {m['truth']} n0={m['n0']} "
                  f"reps={m['reps']} M={m['M']} ({m['runtime_s']:.0f}s) ===")
            base = cd["by"]["base"]
            arms = list(base["arms"])
            for arm in arms:
                b = base["arms"][arm]
                print(f"  base   {arm:<10} cov={b['coverage']:.3f} "
                      f"area={b['area']:.5f} w.01={b['w0.01']:.4f} "
                      f"w.05={b['w0.05']:.4f} j={b['mean_j']:.1f}")
            print(f"  {'config':<20}{'arm':<9}{'cov':>7}{'dcov':>7}"
                  f"{'area':>9}{'darea%':>8}{'w.01':>8}{'dw.01%':>8}"
                  f"{'w.05':>8}{'dw.05%':>8}{'area@matched cov':>18}")
            for c in cd["by"]:
                if c == "base":
                    continue
                for arm in arms:
                    b = base["arms"][arm]
                    v = cd["by"][c]["arms"][arm]
                    # base band whose realised coverage matches the repaired
                    # band's, from the base by_ae curve: the fair width test
                    g = np.array(cd["by"]["base"]["by_ae"]["cov"])
                    ar = np.array(cd["by"]["base"]["by_ae"]["area"])
                    j = int(np.argmin(np.abs(g - v["coverage"])))
                    print(f"  {c:<20}{arm:<9}{v['coverage']:>7.3f}"
                          f"{v['coverage'] - b['coverage']:>+7.3f}"
                          f"{v['area']:>9.5f}"
                          f"{100 * (v['area'] / b['area'] - 1):>+8.2f}"
                          f"{v['w0.01']:>8.4f}"
                          f"{100 * (v['w0.01'] / b['w0.01'] - 1):>+8.2f}"
                          f"{v['w0.05']:>8.4f}"
                          f"{100 * (v['w0.05'] / b['w0.05'] - 1):>+8.2f}"
                          f"{ar[j]:>13.5f} ({g[j]:.3f})")


if __name__ == "__main__":
    what = sys.argv[1]
    files = sys.argv[2:]
    {"fpcal": fpcal, "rough": rough, "m3grid": m3grid,
     "repair": repair}[what](files)
