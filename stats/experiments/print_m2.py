"""Pretty-print m2_experiments.py result JSONs."""
import json
import sys

import numpy as np

HDR = (f"{'arm':<12}{'alpha':>6}{'ae*':>7}{'cov':>7}{'v_low':>7}{'v_high':>7}"
       f"{'depth|m':>9}{'p95d':>7}{'maxd':>7}{'medFPR':>8}{'crnr':>6}"
       f"{'area':>8}{'w.01':>7}{'w.05':>7}{'w.10':>7}{'w.50':>7}{'j*':>7}")


def _f(v, fmt, dash="-"):
    if v is None:
        return f"{dash:>{int(fmt.split('.')[0].strip('>')) if '>' in fmt else 6}}"
    try:
        if isinstance(v, float) and np.isnan(v):
            raise ValueError
        return format(v, fmt)
    except Exception:
        w = fmt.split('.')[0].lstrip('>')
        return f"{dash:>{int(w) if w else 6}}"


def row(arm, a, r):
    ae = r.get("ae_star", r.get("ae"))
    return (f"{arm:<12}{float(a):>6.2f}{_f(ae, '>7.3f')}"
            f"{r['coverage']:>7.3f}{r['viol_low']:>7.3f}{r['viol_high']:>7.3f}"
            f"{r['mean_depth_missers']:>9.4f}{r['p95_depth']:>7.4f}"
            f"{r['max_depth']:>7.4f}{_f(r.get('med_worst_fpr'), '>8.4f')}"
            f"{_f(r.get('frac_miss_corner'), '>6.2f')}"
            f"{r['area']:>8.4f}{r['w0.01']:>7.3f}{r['w0.05']:>7.3f}"
            f"{r['w0.1']:>7.3f}{r['w0.5']:>7.3f}"
            f"{_f(r.get('mean_jstar'), '>7.1f')}")


def main(paths, show_curve=False):
    for path in paths:
        with open(path) as f:
            res = json.load(f)
        for cell, cd in res.items():
            meta = cd.get("_meta", {})
            print(f"\n=== {path}:{cell} truth={meta.get('truth')} "
                  f"n0={meta.get('n0')} n1={meta.get('n1')} "
                  f"reps={meta.get('reps')} M={meta.get('M')} "
                  f"AUC={float(meta.get('true_auc', float('nan'))):.4f} "
                  f"Q={meta.get('quant')} tie={meta.get('tie_mode')} "
                  f"thin={meta.get('thin')} ({float(meta.get('runtime_s', 0)):.0f}s) ===")
            blocks = ([("main", cd)] if "fid_cp" in cd else
                      [(k, cd[k]) for k in sorted(cd) if k != "_meta"])
            for key, blk in blocks:
                if not isinstance(blk, dict) or "fid_cp" not in blk:
                    continue
                print(f"-- {key} (M={blk.get('_M', meta.get('M'))})")
                print(HDR)
                for arm in ("fid_cp", "fid_rc", "recal", "fid_cal"):
                    if arm not in blk:
                        continue
                    for a in sorted(blk[arm], key=float, reverse=True):
                        r = blk[arm][a]
                        if r.get("ae_star", "x") is None:
                            print(f"{arm:<12}{float(a):>6.2f}   (no ae reaches "
                                  f"the target coverage)")
                            continue
                        print(row(arm, a, r))
                ds = blk.get("depth_stats")
                if ds:
                    print("   S(truth) quantiles:", ds["S_true_q"],
                          " mean:", round(ds["mean_S_true"], 1),
                          " | S(draw) quantiles:", ds["S_draw_q"])
                if show_curve and "by_ae" in blk:
                    ae = np.array(blk["by_ae"]["ae"])
                    cv = np.array(blk["by_ae"]["cov"])
                    ar = np.array(blk["by_ae"]["area"])
                    mj = np.array(blk["by_ae"]["mean_j"])
                    print("   ae   cov    area   meanj")
                    for i in range(len(ae)):
                        if round(ae[i], 4) in (0.01, 0.02, 0.05, 0.1, 0.15,
                                               0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                               0.8, 0.9):
                            print(f"   {ae[i]:.2f} {cv[i]:.3f} {ar[i]:.4f} "
                                  f"{mj[i]:7.1f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args, show_curve="--curve" in sys.argv)
