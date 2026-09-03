# ROCnReg `pooledROC.BB` corner check

Does the published *pointwise* Bayesian-bootstrap ROC band (Gu, Ghosal & Roy 2008, as
implemented in the `ROCnReg` R package, `pooledROC.BB`) under-cover at the FPR corners
inside this project's wedge? Theory doc §7.4(i); results in
`data/results/c_calibration_followup_20260830/corner_theory/rocnreg_bb/`.

```bash
# from this directory; needs R with ROCnReg (install.packages("ROCnReg")) and the project env
uv run --project ../../.. python gen_data.py        # 200 reps x 3 shapes, n0 = n1 = 500, seed 11
Rscript api_check.R                                  # one replicate; prints the returned structure
Rscript run_bb.R 2000 200 > run_bb.log               # B = 2000 BB draws, 200 reps, ~15 min
```

Shapes: `t2_99` (t(2) location shift, AUC .99: a wedge cell), `sliver80` (the Cor. 14.1
sliver DGP, AUC .80), `t30_95` (near-binormal, concave corners: reference). Scores are minus
the rank-space placement values; the ROC is rank-invariant so this is without loss.
Evaluation FPRs are the native grid points 1 - k/n0, k in {1,3,5,10,25,50,100,250,450}.

## GET parity (the library closest to the *band*)

`GET` (Myllymäki & Mrkvička 2024) implements the global rank envelope — our trim — on a
user-supplied cloud; it does not generate an ROC cloud. `export_cloud.py` writes one fiducial
cloud (t(2)/.99, n = 500/500, M = 2000) and our C = 1 tube; `get_parity.R` runs
`central_region(type = "rank", coverage = .95)` on the same cloud. Result
(`corner_theory/get_parity.txt`): identical lower and upper edges at all 501 grid points
(max |difference| = 0), same retained fraction .9510; `type = "erl"` is strictly narrower at
498/501 points, as the §5.1 sandwich predicts. The C = 1 band *is* GET's rank envelope of the
fiducial cloud, so the corner defect belongs to the cloud, not to the trim.

```bash
uv run --project ../../.. python export_cloud.py && Rscript get_parity.R   # needs GET (CRAN)
```
