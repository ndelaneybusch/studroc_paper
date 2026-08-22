"""Baseline arms (oracle / KS / Working-Hotelling) on the new M2 cells.

Gives the width yardstick for the P2 vulnerability slices, using exactly the
same evaluation grid and metrics as rank_band_experiments.run_cell.
"""
import argparse
import json
import time

import numpy as np

import m2_experiments as m2
import rank_band_experiments as rbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--Mor", type=int, default=4000)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.2, 0.05])
    ap.add_argument("--arms", nargs="+", default=["oracle", "ks", "wh"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for cname in args.cells:
        spec = m2.CELLS[cname]
        curve, srepr = m2.build_truth(spec["truth"])
        print(f"== {cname} {spec['truth']} n0={spec['n0']} n1={spec['n1']} "
              f"AUC={curve.auc():.4f} ==", flush=True)
        t0 = time.time()
        res = rbe.run_cell(cname, curve, srepr, spec["n0"], spec["n1"],
                           args.reps, 100, args.Mor, args.alphas,
                           args.seed + sum(ord(c) for c in cname), args.arms)
        res["_meta"] = {**spec, "true_auc": curve.auc(), "reps": args.reps,
                        "Mor": args.Mor, "runtime_s": time.time() - t0}
        results[cname] = res
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1, default=str)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
