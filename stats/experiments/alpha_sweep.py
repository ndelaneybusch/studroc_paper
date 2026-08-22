"""P5: nominal-vs-actual calibration sweep for the fiducial band variants.

For each cell it reads the by_alpha_eff curve and reports, at each nominal
alpha, the coverage of
  fid_cp  : the raw fiducial rule (alpha_eff = alpha)
  fid_rc  : the recalibrated rule alpha_eff = 1 - (1-alpha)**C
plus the per-cell optimal alpha_eff.
"""
import json
import sys

import numpy as np

SWEEP = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
C = 2.2


def main(paths):
    for p in paths:
        res = json.load(open(p))
        for cell, cd in res.items():
            blocks = ([("main", cd)] if "by_ae" in cd else
                      [(k, v) for k, v in cd.items()
                       if isinstance(v, dict) and "by_ae" in v])
            for key, blk in blocks:
                ae = np.array(blk["by_ae"]["ae"])
                cv = np.array(blk["by_ae"]["cov"])
                ar = np.array(blk["by_ae"]["area"])
                mj = np.array(blk["by_ae"]["mean_j"])
                meta = cd.get("_meta", {})
                print(f"\n=== {cell}/{key} n0={meta.get('n0')} "
                      f"truth={meta.get('truth')} M={blk.get('_M')} "
                      f"reps={meta.get('reps')} ===")
                print(f"{'alpha':>6}{'nominal':>9}{'fid_cp':>9}{'fid_rc':>9}"
                      f"{'ae_rc':>8}{'area_cp':>9}{'area_rc':>9}{'j_rc':>8}"
                      f"{'ae*':>7}")
                for a in SWEEP:
                    i0 = int(np.argmin(np.abs(ae - a)))
                    aer = 1 - (1 - a) ** C
                    i1 = int(np.argmin(np.abs(ae - aer)))
                    ok = np.nonzero(cv >= 1 - a - 1e-12)[0]
                    aes = ae[ok.max()] if len(ok) else np.nan
                    print(f"{a:>6.2f}{1-a:>9.3f}{cv[i0]:>9.3f}{cv[i1]:>9.3f}"
                          f"{ae[i1]:>8.3f}{ar[i0]:>9.4f}{ar[i1]:>9.4f}"
                          f"{mj[i1]:>8.1f}{aes:>7.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
