"""Refit the surfaces and the worst-case grid without the report: quick_grid.py OUT."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("stats/experiments").resolve()))
import auto_rule_surface as ars  # noqa: E402

start = time.time()
hot = [
    Path("data/results") / n
    for n in ("shape_auc_study", "shape_auc_study_p10", "nov11", "nov11_p10", "nov11_p30")
]
draws = ars.in_design_space(
    frame=ars.load_draws(hot_start=hot, out_dir=Path("data/results/auto_rule"))
)
draws["miss"] = draws["miss"].astype(bool)
print(f"{len(draws):,} draws")
surfaces = ars.fit_surfaces(draws=draws, seed=20261117)
grid = ars.worst_case_grid(surfaces=surfaces, target=ars.ALPHA)
grid.to_feather(sys.argv[1])
print(f"Saved {sys.argv[1]} in {(time.time() - start) / 60:.1f} min")
