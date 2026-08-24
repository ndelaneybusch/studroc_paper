"""Tables for the M3/M4 result JSONs (``res_m3_*.json``, ``res_m4_*.json``).

``m3grid`` files give M3's realised coverage/area over the nominal-level ladder,
plus the *effective* level -- the largest nominal alpha whose realised coverage
still reaches a target -- which is the direct measure of how conservative the
composition is.  ``joint`` files give the miss cap and the containment probe.
"""
import json
import sys

import numpy as np

# published benchmarks (m2_report.md; fid_* recomputed from the by_ae tables of
# res_p1diag_*/res_p2_*/res_p4_ab.json at ae = 1-(1-alpha)^C)
BENCH = {
    #          fid_cp   fid_rc   oracle    KS      WH     WHcov
    "C1":  dict(a05=(0.1394, 0.1272, 0.1277, 0.2328, 0.0803, 0.955),
                a50=(0.0947, 0.0811, 0.0863, None, None, None),
                cov05=(0.980, 0.963), cov50=(0.733, 0.545)),
    "C2":  dict(a05=(0.0634, 0.0579, 0.0515, 0.1718, 0.0313, 0.963),
                a50=(0.0418, 0.0352, 0.0352, None, None, None),
                cov05=(0.975, 0.963), cov50=(0.748, 0.540)),
    "C3":  dict(a05=(0.1137, 0.1029, 0.0740, 0.2796, 0.0567, 0.975),
                a50=(0.0719, 0.0596, 0.0518, None, None, None),
                cov05=(0.990, 0.975), cov50=(0.777, 0.573)),
    "C4":  dict(a05=(0.0856, 0.0787, 0.0737, 0.1929, 0.0485, 0.000),
                a50=(0.0571, 0.0482, 0.0499, None, None, None),
                cov05=(0.980, 0.970), cov50=(0.777, 0.550)),
    "C5":  dict(a05=(0.0775, 0.0709, 0.0547, 0.1777, 0.0571, 0.000),
                a50=(0.0509, 0.0426, 0.0410, None, None, None),
                cov05=(0.968, 0.945), cov50=(0.655, 0.435)),
    "P2c": dict(a05=(0.0288, 0.0259, 0.0155, 0.1393, 0.0094, 0.968),
                a50=(0.0179, 0.0148, 0.0109, None, None, None),
                cov05=(0.978, 0.958), cov50=(0.713, 0.510)),
    "P2d": dict(a05=(0.0613, 0.0544, 0.0191, 0.2412, 0.0178, 0.970),
                a50=(0.0361, 0.0295, 0.0135, None, None, None),
                cov05=(0.995, 0.980), cov50=(0.730, 0.545)),
    "P4b": dict(a05=(0.0320, 0.0294, None, None, None, None),
                a50=(0.0219, 0.0188, None, None, None, None),
                cov05=(0.970, 0.945), cov50=(0.715, 0.440)),
}


def eff_level(levels, cov, target):
    """Largest nominal level whose realised coverage still reaches ``target``."""
    lv = np.asarray(levels, float)
    cv = np.asarray(cov, float)
    o = np.argsort(lv)
    lv, cv = lv[o], cv[o]
    ok = np.nonzero(cv >= target - 1e-12)[0]
    return float(lv[ok.max()]) if len(ok) else None


def show_grid(path):
    res = json.load(open(path))
    for cell, cd in res.items():
        meta = cd["_meta"]
        print(f"\n=== {path}:{cell} truth={meta['truth']} n0={meta['n0']} "
              f"n1={meta['n1']} reps={meta['reps']} AUC={float(meta['true_auc']):.4f} "
              f"B=({meta['B0']},{meta['B1']}) ===")
        for sp in [k for k in cd if k not in ("_meta",)]:
            blk = cd[sp]
            lv = blk["levels"]
            cov = [blk["by_level"][str(a)]["coverage"] for a in lv]
            print(f"-- split={sp}  component cov @a=.05 "
                  f"{[round(x, 4) for x in blk['component_cov']]}")
            print(f"{'alpha':>7}{'gamma0':>10}{'gamma1':>10}{'mcidx':>7}"
                  f"{'cov':>7}{'v_lo':>7}{'v_hi':>7}{'p95d':>8}{'maxd':>8}"
                  f"{'area':>8}{'w.01':>7}{'w.05':>7}{'w.50':>7}")
            for a in lv:
                r = blk["by_level"][str(a)]
                c = blk["calib"][str(a)]
                print(f"{a:>7.3f}{c['gamma0']:>10.3g}{c['gamma1']:>10.3g}"
                      f"{min(c['mc_index']):>7d}"
                      f"{r['coverage']:>7.3f}{r['viol_low']:>7.3f}"
                      f"{r['viol_high']:>7.3f}{r['p95_depth']:>8.4f}"
                      f"{r['max_depth']:>8.4f}{r['area']:>8.4f}"
                      f"{r['w0.01']:>7.3f}{r['w0.05']:>7.3f}{r['w0.5']:>7.3f}")
            for tgt in (0.95, 0.5):
                print(f"   effective level for realised coverage >= {tgt}: "
                      f"{eff_level(lv, cov, tgt)}")
            b = BENCH.get(cell)
            if b:
                for a, key, ck in ((0.05, "a05", "cov05"), (0.5, "a50", "cov50")):
                    r = blk["by_level"][str(a)]
                    fc, fr, orc, ks, wh, whc = b[key]
                    parts = [f"M3 area {r['area']:.4f} (cov {r['coverage']:.3f})",
                             f"fid_cp {fc:.4f} (cov {b[ck][0]:.3f}) x{r['area']/fc:.2f}",
                             f"fid_rc {fr:.4f} (cov {b[ck][1]:.3f}) x{r['area']/fr:.2f}"]
                    if orc:
                        parts.append(f"oracle {orc:.4f} x{r['area']/orc:.2f}")
                    if ks:
                        parts.append(f"KS {ks:.4f} x{r['area']/ks:.2f}")
                    if wh:
                        parts.append(f"WH {wh:.4f} (cov {whc:.3f}) x{r['area']/wh:.2f}")
                    print(f"   [alpha={a}] " + " | ".join(parts))


def show_joint(path):
    res = json.load(open(path))
    for cell, cd in res.items():
        meta = cd["_meta"]
        print(f"\n=== {path}:{cell} truth={meta['truth']} n0={meta['n0']} "
              f"n1={meta['n1']} reps={meta['reps']} M={meta['M']} C={meta['C']} "
              f"capfrac={meta['cap_frac']} ({float(meta['runtime_s']):.0f}s) ===")
        keys = sorted(cd["arms"], key=lambda k: (k.split("|")[1], k))
        print(f"{'arm':<10}{'alpha':>6}{'cov':>7}{'v_lo':>7}{'v_hi':>7}"
              f"{'d|miss':>9}{'p95d':>8}{'maxd':>8}{'area':>8}{'empty':>7}")
        for k in keys:
            nm, a = k.split("|")
            r = cd["arms"][k]
            print(f"{nm:<10}{float(a):>6.2f}{r['coverage']:>7.3f}"
                  f"{r['viol_low']:>7.3f}{r['viol_high']:>7.3f}"
                  f"{r['mean_depth_missers']:>9.4f}{r['p95_depth']:>8.4f}"
                  f"{r['max_depth']:>8.4f}{r['area']:>8.4f}"
                  f"{r['frac_empty']:>7.3f}")
        for a in sorted({k.split("|")[1] for k in cd["arms"]}, key=float):
            for nm in ("cp", "rc"):
                ref = f"fp_{nm}|{a}" if f"fp_{nm}|{a}" in cd["arms"] \
                    else f"fid_{nm}|{a}"
                f = cd["arms"][ref]
                c = cd["arms"][f"cap_{nm}|{a}"]
                print(f"   cap cost alpha={a} ref=fid_{nm}: area "
                      f"{f['area']:.4f} -> {c['area']:.4f} "
                      f"({100*(c['area']/f['area']-1):+.2f}%), coverage "
                      f"{f['coverage']:.3f} -> {c['coverage']:.3f}, "
                      f"max depth {f['max_depth']:.4f} -> {c['max_depth']:.4f}, "
                      f"p95 {f['p95_depth']:.4f} -> {c['p95_depth']:.4f}")
        if "bind_frac" in cd:
            print(f"   cap binding grid fraction: "
                  f"{ {k: round(v, 4) for k, v in cd['bind_frac'].items()} }")
        for tag, lab in (("contain_in", "M3(level) INSIDE fid(alpha)"),
                         ("contain_out", "M3(level) OUTSIDE fid(alpha)")):
            print(f"   containment {lab} [fraction of reps; key alpha|ref|k0]")
            for key, dd in sorted(cd[tag].items()):
                vals = sorted(dd.items(), key=lambda x: -float(x[0]))
                if all(v == 0.0 for _, v in vals):
                    print(f"     {key}: never (0.00 at every level)")
                    continue
                print(f"     {key}: " + " ".join(f"{float(l):g}:{v:.2f}"
                                                 for l, v in vals))
        if "poke" in cd:
            print("   overhang of M3(level) outside fid(alpha) "
                  "[mean/p95/max over reps] = certified miss-depth cap")
            for key, dd in sorted(cd["poke"].items()):
                vals = sorted(dd.items(), key=lambda x: -float(x[0]))
                print(f"     {key}: " + " ".join(
                    f"{float(l):g}:{v[0]:.4f}/{v[1]:.4f}/{v[2]:.4f}"
                    for l, v in vals))
        print(f"   mean trim depth j (cp, rc): {cd['mean_j']}")


def show_bracket(path):
    res = json.load(open(path))
    for cell, cd in res.items():
        m = cd["_meta"]
        print(f"\n=== {path}:{cell} truth={m['truth']} n0={m['n0']} n1={m['n1']} "
              f"reps={m['reps']} M={m['M']} ncal={m['ncal']} m_in={m['m_in']} "
              f"set_level={m['set_level']} ({float(m['runtime_s']):.0f}s) ===")
        print(f"   oracle trim depth on the M={m['M']} scale "
              f"(j_thresh, j_quant) by inner budget: {cd['oracle_j']}")
        for a in sorted(cd["mean_j"], key=float, reverse=True):
            print(f"-- alpha={a}: mean trim depth per member / bracket")
            mj = cd["mean_j"][a]
            for k in sorted(mj):
                print(f"     {k:>10s} {mj[k]:>8.1f}")
        print(f"{'arm':<12}{'alpha':>6}{'cov':>7}{'v_lo':>7}{'v_hi':>7}"
              f"{'d|miss':>9}{'p95d':>8}{'maxd':>8}{'area':>8}")
        for k in sorted(cd["arms"], key=lambda k: (-float(k.split("|")[1]), k)):
            nm, a = k.split("|")
            r = cd["arms"][k]
            print(f"{nm:<12}{float(a):>6.2f}{r['coverage']:>7.3f}"
                  f"{r['viol_low']:>7.3f}{r['viol_high']:>7.3f}"
                  f"{r['mean_depth_missers']:>9.4f}{r['p95_depth']:>8.4f}"
                  f"{r['max_depth']:>8.4f}{r['area']:>8.4f}")
        # is the calibration functional monotone in the member's early slope?
        sl = cd["member_slope"]
        for a in sorted(cd["j_raw"], key=float, reverse=True):
            jr = cd["j_raw"][a]
            xs, ys = [], []
            for nm in sl:
                xs += sl[nm]
                ys += jr[nm]
            xs, ys = np.array(xs), np.array(ys, float)
            rk = np.corrcoef(np.argsort(np.argsort(xs)),
                             np.argsort(np.argsort(ys)))[0, 1]
            per = {nm: (float(np.mean(sl[nm])), float(np.mean(jr[nm])))
                   for nm in sl}
            print(f"   alpha={a}: Spearman(member early slope R(.05), "
                  f"calibrated j) = {rk:+.3f} over {len(xs)} member-replicates")
            print("      mean (slope, j) per member: "
                  + ", ".join(f"{nm} ({v[0]:.3f},{v[1]:.1f})"
                              for nm, v in per.items()))


def show_family(path):
    res = json.load(open(path))
    print(f"\n=== {path} ===")
    hdr = f"{'curve':<14}{'n0':>6}{'AUC':>8}{'R(.01)':>8}{'R(.05)':>8}"
    alphas = sorted(next(iter(res.values()))["recal"], key=float, reverse=True)
    for a in alphas:
        hdr += f"{'ae*@' + a:>10}{'C*':>6}{'j*':>7}"
    print(hdr)
    for name, cd in res.items():
        m = cd["_meta"]
        row = (f"{name:<14}{m['n0']:>6}{m['auc']:>8.4f}"
               f"{m['early_slope_01']:>8.3f}{m['early_slope_05']:>8.3f}")
        for a in alphas:
            r = cd["recal"][a]
            ae = r.get("ae_star")
            if ae is None:
                row += f"{'-':>10}{'-':>6}{'-':>7}"
                continue
            row += (f"{ae:>10.3f}"
                    f"{np.log(1 - ae) / np.log(1 - float(a)):>6.2f}"
                    f"{r['mean_jstar']:>7.1f}")
        print(row)
    for name, cd in res.items():
        print(f"   {name}: fid_cp " + " ".join(
            f"a={a} cov={cd['fid_cp'][a]['coverage']:.3f} "
            f"area={cd['fid_cp'][a]['area']:.4f}" for a in alphas))


def main(paths):
    for p in paths:
        d = json.load(open(p))
        first = next(iter(d.values()))
        if "oracle_j" in first:
            show_bracket(p)
        elif "recal" in first:
            show_family(p)
        elif "arms" in first:
            show_joint(p)
        else:
            show_grid(p)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("-")])
