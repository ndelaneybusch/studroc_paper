# Research work status

Audit date: 2026-09-10. This records the disposition of the eight files left
uncommitted after the exploration work; it is not a live agent registry.

| Work | Location | Disposition |
|---|---|---|
| Production hybrid floor | [implementation](../src/studroc_paper/methods/hybrid_floor.py), [tests](../tests/test_hybrid_floor.py) | Optional `m3_floor=True` / `M3Floor` support in both fiducial APIs; exact and Stage F rules. Included with this status page. The default remains the raw band. |
| Rank-likelihood theory and verification | [theory §§12.5–12.7](fiducial_band_theory.md#125-the-full-bracket-probability-is-a-likelihood-not-a-confidence-level), [script](experiments/rank_likelihood_checks_20260906.py), [results](experiments/res_rank_likelihood_checks_20260906.json) | Committed in `3f8ecb2`; exact checks reproduced the saved results. Useful inversion width remains an open research question. |
| Interior, small-n, likelihood, projection and M3 exploration harness | [spec](methods_exploration_spec.md), [runner](../scripts/methods_exploration/README.md), [pilot validation](methods_exploration_validation.md) | Committed in `5b96bae` and `39ec98a`. The validation note describes the pre-screen implementation check. |
| September 9–10 screen and large-n eligibility follow-up | [screen report](methods_exploration_report.md), [follow-up script](../scripts/methods_exploration/window_eligibility_50k.py), [follow-up results](../data/results/methods_exploration_50k_eligibility/eligibility.json) | Remote commits `66b8e7e`, `a43df7d`, `21e5307`, authored by Nate Delaney-Busch and co-authored by Claude Opus 5. Four tracks completed; interior stopped at 75,510/90,000 datasets. No method promoted. |

The production floor files were last modified September 7; the theory and its
verification files were last modified September 6. These were pre-existing changes
that the exploration commits deliberately excluded. Git does not identify the
author of an uncommitted change, so no individual or agent attribution is inferred.

At the audit, the repository had one worktree (`main`), no stashes, and no running
simulation or test workers. A separate Claude session was open in this repository;
its task ownership could not be established from the working tree. File hashes
remained stable during review. An original-file snapshot and tracked patch were
saved locally under `/tmp/studroc-cleanup-20260910` before cleanup.

## Validation

The production floor, both fiducial APIs, ladder, follow-up, M3 and exploration
suites passed **209 tests**. The only warnings were the existing deliberately
undersized-cloud checks. Ruff lint and formatting checks passed for the files
being committed. The exact-arithmetic script reproduced its saved JSON in full:
2,358 anchor/path checks, 210 path/model checks, 25 normalized laws and 544
one-cut/subdivision identities, plus refinement and enumerated-coverage results.

The API documentation now states that widening can reduce misses outside the
floor; it cannot increase them. It does not claim a distribution-free bound there.
The cleanup otherwise preserves the production implementation and theory work.

## Outstanding work and data

The full screen has already run on another checkout, according to the newly
fetched report; do not treat the earlier pilot validation note as current run
status. The full screen's raw records under `data/results/methods_exploration_screen/`
are not included in these commits and were not present in this local checkout at
the audit. Its results were not independently reproduced during cleanup. The
50,000-sample eligibility follow-up JSON is tracked and is included in the fetched
commits. Recover or locate the full screen records before attempting to resume or
independently audit that run.

The report's proposed next measurements are production-cloud coverage at n=5,000
and a targeted M3 class-allocation follow-up. These have not been started by this
cleanup. The week-long decision gate remains pending. Completing the old interior
target is optional research work, not a prerequisite silently triggered here.

Any new run needs an output directory consistent with its source fingerprint.
Historical pilot manifests intentionally refer to the source snapshots used to
generate them. Committing the production floor and adding the follow-up module
changes the fingerprint, so existing outputs cannot be resumed with the current
source without returning to their recorded source version.
