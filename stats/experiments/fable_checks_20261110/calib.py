"""One-sided sup statistics of uniform order statistics, and region calibration."""
import sys
from pathlib import Path
import numpy as np
from scipy.special import betainc, ndtri, ndtr

def sup_stats(n, N, seed, chunk=20000):
    rng = np.random.default_rng(seed)
    i = np.arange(1, n + 1, dtype=float)
    TL = np.empty(N); TU = np.empty(N)
    for s in range(0, N, chunk):
        m = min(chunk, N - s)
        e = rng.standard_exponential((m, n + 1))
        a = np.cumsum(e, axis=1)
        a = a[:, :n] / a[:, -1:]
        cdf = betainc(i, n + 1 - i, a)
        cdf = np.clip(cdf, 1e-300, 1 - 1e-16)
        z = ndtri(cdf)
        TL[s:s+m] = (-z).max(axis=1)
        TU[s:s+m] = z.max(axis=1)
    return TL, TU

def stats_cached(n, N=400000, seed=1):
    f = Path(__file__).parent / f"sup_{n}_{N}_{seed}.npz"
    if f.exists():
        d = np.load(f); return d["TL"], d["TU"]
    TL, TU = sup_stats(n, N, seed + 1000 * n)
    np.savez(f, TL=TL, TU=TU)
    return TL, TU

def radius(n0, n1, alpha, p, N=400000):
    """Scale c with P{(T0L,T1U) and (T0U,T1L) in the l_p quarter-ball of radius c} = 1-alpha."""
    L0, U0 = stats_cached(n0, N, 1)
    L1, U1 = stats_cached(n1, N, 2)
    pos = lambda x: np.maximum(x, 0)
    if np.isinf(p):
        s = np.maximum.reduce([pos(L0), pos(U1), pos(U0), pos(L1)])
    else:
        s = np.maximum(pos(L0)**p + pos(U1)**p, pos(U0)**p + pos(L1)**p) ** (1 / p)
    return float(np.quantile(s, 1 - alpha, method="higher"))

if __name__ == "__main__":
    for n0, n1 in [(50, 50), (500, 500), (900, 100), (15, 15)]:
        for alpha in [0.05, 0.2, 0.5]:
            cs = {p: radius(n0, n1, alpha, p) for p in [1, 2, 3, 4, np.inf]}
            cr = cs[np.inf]
            # support function h_p(phi) for the l_p ball: dual norm radius; at phi=45deg: c*2^(1/2-1/p)... h(phi)=c*||(cos,sin)||_q
            out = []
            for p, c in cs.items():
                q = 1.0 if np.isinf(p) else (np.inf if p == 1 else p / (p - 1))
                def h(phi):
                    v = np.array([np.cos(phi), np.sin(phi)])
                    return c * (v.max() if np.isinf(q) else (v**q).sum() ** (1 / q))
                out.append(f"p={p}: c={c:.3f} h45/sq={h(np.pi/4)/(cr*2**.5):.3f} h20/sq={h(np.pi/9)/(cr*(np.cos(np.pi/9)+np.sin(np.pi/9))):.3f} h0/sq={h(0)/cr:.3f}")
            print(n0, n1, alpha, " | ".join(out), flush=True)
