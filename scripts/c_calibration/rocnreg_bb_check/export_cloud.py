"""Export one fiducial cloud (t(2)/.99, n=500/500, M=2000) and our C=1 trimmed tube for a GET parity check."""
import sys, os, numpy as np, torch
os.chdir('/home/nathan/Documents/studroc_paper'); sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/c_calibration')
from studroc_paper.methods.fiducial_band import _fiducial_cloud, _minp_depths, _pointwise_order_stats
from corner_mechanism import t_shape
n, M, alpha = 500, 2000, 0.05
_, R, Rinv, _, _ = t_shape(df=2.0, auc=0.99)
rng = np.random.default_rng(2026)
u = rng.random(n); w = Rinv(rng.random(n))
lab = np.concatenate([np.zeros(n, np.uint8), np.ones(n, np.uint8)])[np.argsort(np.concatenate([u, w]), kind='stable')]
grid = np.arange(n + 1) / n
cloud = _fiducial_cloud(lab, n, n, M, grid, rng, torch.device('cpu'), torch.float64)
depths = _minp_depths(cloud)                     # min-p depth S per draw (tie-inclusive ranks)
j = int(torch.sort(depths.flatten())[0][int(np.floor(alpha * M))].item())   # (floor(alpha M)+1)-th smallest depth  (Lemma 6)
lo, hi = _pointwise_order_stats(cloud, j)
out = '.'
np.savetxt(f'{out}/cloud.csv', cloud.numpy(), delimiter=',', fmt='%.10g')
np.savetxt(f'{out}/ours.csv', np.column_stack([grid, lo.numpy(), hi.numpy(), R(grid)]), delimiter=',', header='t,lo,hi,truth', comments='', fmt='%.10g')
inside = (depths.flatten() >= j).float().mean().item()
print(f'M={M} K={n+1} j*={j} retained fraction={inside:.4f} (>= 1-alpha_eff required); depth min/median={depths.min().item():.0f}/{depths.median().item():.0f}')
