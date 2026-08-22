import json
import sys

import numpy as np

ARMS = ["oracle", "plug_hz", "plug_lin", "plug_step", "fiducial", "fid_cp",
        "ks", "wh"]


def main(paths):
    for path in paths:
        with open(path) as f:
            res = json.load(f)
        for cell, cd in res.items():
            meta = cd.pop("_meta", {})
            print(f"\n=== {cell} truth={meta.get('truth')} n0={meta.get('n0')} "
                  f"n1={meta.get('n1')} reps={meta.get('reps')} "
                  f"({meta.get('runtime_s', 0):.0f}s) ===")
            hdr = (f"{'arm':<10}{'alpha':>6}{'cov':>7}{'v_low':>7}{'v_high':>7}"
                   f"{'depth|m':>9}{'p95d':>7}{'maxd':>7}{'medFPR':>8}"
                   f"{'area':>8}{'w.01':>7}{'w.05':>7}{'w.10':>7}{'w.50':>7}{'j*':>6}")
            print(hdr)
            for arm in ARMS:
                if arm not in cd:
                    continue
                for a in sorted(cd[arm], key=float, reverse=True):
                    r = cd[arm][a]
                    mf = r.get("med_worst_fpr")
                    mf = "-" if mf is None or (isinstance(mf, float) and np.isnan(mf)) else f"{float(mf):.4f}"
                    js = r.get("mean_jstar")
                    js = "-" if js is None or (isinstance(js, float) and np.isnan(js)) else f"{float(js):.0f}"
                    print(f"{arm:<10}{float(a):>6.2f}{r['coverage']:>7.3f}"
                          f"{r['viol_low']:>7.3f}{r['viol_high']:>7.3f}"
                          f"{r['mean_depth_missers']:>9.4f}{r['p95_depth']:>7.4f}"
                          f"{r['max_depth']:>7.4f}{mf:>8}"
                          f"{r['area']:>8.4f}{r['w0.01']:>7.3f}{r['w0.05']:>7.3f}"
                          f"{r['w0.1']:>7.3f}{r['w0.5']:>7.3f}{js:>6}")


if __name__ == "__main__":
    main(sys.argv[1:])
