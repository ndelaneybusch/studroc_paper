"""P1 analysis: does a single trim-level recalibration map transfer across cells?

Reads the by_alpha_eff calibration curves produced by m2_experiments.py and
evaluates candidate maps  alpha  ->  alpha_eff  used in the fiducial trim rule.
"""
import json
import sys

import numpy as np

ALPHAS = [0.5, 0.2, 0.1, 0.05]
FIT_CELLS = {"C1", "C2", "C4", "C5"}


def load(paths):
    cells = {}
    for p in paths:
        with open(p) as f:
            res = json.load(f)
        for cell, cd in res.items():
            meta = cd.get("_meta", {})
            for key, blk in cd.items():
                if key == "_meta" or "by_ae" not in blk:
                    continue
                name = cell if key == "main" else f"{cell}/{key}"
                cells[name] = dict(
                    ae=np.array(blk["by_ae"]["ae"]),
                    cov=np.array(blk["by_ae"]["cov"]),
                    area=np.array(blk["by_ae"]["area"]),
                    mean_j=np.array(blk["by_ae"]["mean_j"]),
                    vlow=np.array(blk["by_ae"].get("vlow", [np.nan])),
                    vhigh=np.array(blk["by_ae"].get("vhigh", [np.nan])),
                    recal={a: blk["recal"][a].get("ae_star")
                           for a in blk["recal"]},
                    meta=meta, M=blk.get("_M"),
                )
    return cells


def at(c, ae):
    i = int(np.argmin(np.abs(c["ae"] - ae)))
    return c["cov"][i], c["area"][i], c["mean_j"][i]


def main(paths):
    cells = load(paths)
    names = list(cells)

    print("\n### per-cell optimal alpha_eff* (largest ae with cov >= 1-alpha)")
    print(f"{'cell':<14}" + "".join(f"{a:>9.2f}" for a in ALPHAS))
    tab = {}
    for n in names:
        row = []
        for a in ALPHAS:
            v = cells[n]["recal"].get(str(a))
            row.append(float(v) if v is not None else np.nan)
        tab[n] = row
        print(f"{n:<14}" + "".join(f"{v:>9.3f}" for v in row))
    A = np.array([tab[n] for n in names])
    print(f"{'median':<14}" + "".join(f"{v:>9.3f}" for v in np.nanmedian(A, 0)))
    print(f"{'min':<14}" + "".join(f"{v:>9.3f}" for v in np.nanmin(A, 0)))
    print(f"{'max':<14}" + "".join(f"{v:>9.3f}" for v in np.nanmax(A, 0)))
    fit = [i for i, n in enumerate(names) if n.split("/")[0] in FIT_CELLS]
    print(f"{'median(fit)':<14}" +
          "".join(f"{v:>9.3f}" for v in np.nanmedian(A[fit], 0)))
    print("  implied Sidak exponent C = log(1-ae*)/log(1-alpha):")
    for i, a in enumerate(ALPHAS):
        C = np.log(1 - A[:, i]) / np.log(1 - a)
        print(f"    alpha={a:<5} median C={np.nanmedian(C):.2f} "
              f"[{np.nanmin(C):.2f}, {np.nanmax(C):.2f}]")

    for C in (1.0, 2.0, 2.2, 2.4, 2.6):
        lab = "identity (fid_cp)" if C == 1.0 else f"ae = 1-(1-a)^{C}"
        print(f"\n### map: {lab}")
        print(f"{'cell':<14}" + "".join(f"{'cov@'+str(a):>10}" for a in ALPHAS)
              + "".join(f"{'ar/'+str(a):>10}" for a in ALPHAS))
        cov_all = []
        for n in names:
            crow, arow = [], []
            for a in ALPHAS:
                ae = 1 - (1 - a) ** C
                cv, ar, _ = at(cells[n], ae)
                cv0, ar0, _ = at(cells[n], a)
                crow.append(cv)
                arow.append(ar / ar0)
            cov_all.append(crow)
            print(f"{n:<14}" + "".join(f"{v:>10.3f}" for v in crow)
                  + "".join(f"{v:>10.3f}" for v in arow))
        Cv = np.array(cov_all)
        print(f"{'min':<14}" + "".join(f"{v:>10.3f}" for v in Cv.min(0)))
        print(f"{'max':<14}" + "".join(f"{v:>10.3f}" for v in Cv.max(0)))
        print(f"{'spread(pp)':<14}" +
              "".join(f"{100*v:>10.1f}" for v in (Cv.max(0) - Cv.min(0))))
        print(f"{'err vs nom':<14}" +
              "".join(f"{v:>10.3f}" for v in
                      (Cv.mean(0) - (1 - np.array(ALPHAS)))))


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("-")])
