"""Paired area/coverage of corner-rounded M3 compositions versus the square."""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist, norm, t as tdist
from scipy.special import ndtr
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.methods_exploration.m3 import compose
from calib import stats_cached

M_ANG = 24

def stair(p):
    th = np.linspace(0, np.pi / 2, M_ANG + 1)
    if np.isinf(p):
        return np.array([1.0]), np.array([1.0])
    x = np.cos(th) ** (2 / p); y = np.sin(th) ** (2 / p)
    return x[:-1], y[1:]

def scale(n0, n1, alpha, p):
    L0, U0 = stats_cached(n0, 400000, 1); L1, U1 = stats_cached(n1, 400000, 2)
    x, y = stair(p); pos = lambda v: np.maximum(v, 0)
    def need(a, b):
        out = np.full(a.shape, np.inf)
        for xk, yk in zip(x, y):
            out = np.minimum(out, np.maximum(a / xk, b / yk))
        return out
    s = np.maximum(need(pos(L0), pos(U1)), need(pos(U0), pos(L1)))
    return float(np.quantile(s, 1 - alpha, method="higher"))

def bounds(n, zs):
    i = np.arange(1, n + 1.0)
    lo = np.array([beta_dist.ppf(ndtr(-z), i, n + 1 - i) for z in zs])
    hi = np.array([beta_dist.ppf(ndtr(z), i, n + 1 - i) for z in zs])
    return lo, hi

class Arc:
    def __init__(self, n0, n1, alpha, p):
        x, y = stair(p); c = scale(n0, n1, alpha, p); self.c = c
        # upper edge: negatives' lower bounds at z0=c*x_k with positives' upper bounds at z1=c*y_k
        self.lo0, self.hi0 = bounds(n0, c * x)      # class-0 scale along x
        self.lo1, self.hi1 = bounds(n1, c * y)      # class-1 scale along y
        self.k = len(x)
    def band(self, labels):
        L = U = None
        for k in range(self.k):
            g, lo, up = compose(labels=labels, bounds0=(self.lo0[k], self.hi0[k]), bounds1=(self.lo1[k], self.hi1[k]))
            L = lo if L is None else np.minimum(L, lo)
            U = up if U is None else np.maximum(U, up)
        return g, L, U

SHAPES = {
    "diag": (lambda r, n: r.standard_normal(n), lambda r, n: r.standard_normal(n), lambda t: t),
    "bn80": (lambda r, n: r.standard_normal(n), lambda r, n: r.standard_normal(n) + 1.1902, lambda t: norm.sf(norm.isf(t) - 1.1902)),
    "bn95": (lambda r, n: r.standard_normal(n), lambda r, n: r.standard_normal(n) + 2.3262, lambda t: norm.sf(norm.isf(t) - 2.3262)),
    "t2s10": (lambda r, n: r.standard_t(2, n), lambda r, n: r.standard_t(2, n) + 10, lambda t: tdist.sf(tdist.isf(t, 2) - 10, 2)),
    "het3": (lambda r, n: r.standard_normal(n), lambda r, n: 3 * r.standard_normal(n) + 1.5, lambda t: norm.sf((norm.isf(t) - 1.5) / 3)),
}

def run(n0, n1, alpha, reps=300, ps=(np.inf, 4, 3, 2)):
    arcs = {p: Arc(n0, n1, alpha, p) for p in ps}
    for name, (f0, f1, R) in SHAPES.items():
        rng = np.random.default_rng(7)
        area = {p: [] for p in ps}; cov = {p: 0 for p in ps}
        grid = np.arange(n0 + 1) / n0
        with np.errstate(all="ignore"):
            truth = R(grid); truth[0] = 0 if name != "x" else truth[0]; truth[-1] = 1
        for _ in range(reps):
            x = f0(rng, n0); y = f1(rng, n1)
            s = np.r_[x, y]; lab = np.r_[np.zeros(n0, int), np.ones(n1, int)]
            labels = lab[np.argsort(-s, kind="stable")]
            for p in ps:
                g, L, U = arcs[p].band(labels)
                area[p].append(np.mean((U - L)[1:-1]))
                cov[p] += bool(np.all((L[1:-1] <= truth[1:-1] + 1e-12) & (truth[1:-1] <= U[1:-1] + 1e-12)))
        base = np.array(area[np.inf])
        print(n0, n1, alpha, name, " ".join(f"p={p}: ratio={np.mean(np.array(area[p]))/base.mean():.3f} cov={cov[p]/reps:.3f}" for p in ps), f"| sq area={base.mean():.4f}", flush=True)

if __name__ == "__main__":
    for n0, n1 in [(50, 50), (500, 500), (900, 100)]:
        for alpha in (0.05, 0.5):
            run(n0, n1, alpha)
