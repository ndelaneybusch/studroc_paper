# C-calibration screening verdict

**STOP: retain a documented fixed/default rule; full auto-map study is not justified**

- Usable alpha=.05 cells: 27
- Shape lower envelope, C* minus one SE: 0.967
- Mean oracle area gain versus C=1 on the shape screen: 9.5%
- Strong C* < 1 flags: 1
- Complete evidence arms: {'shape': True, 'taper': True, 'imbalance': True}

This is a resource-allocation gate, not a coverage guarantee. A positive verdict still requires a constrained fit and fresh confirmation.

## Stage A routing

- Large n: do not run a dense large-n arm; validate a C=1 clamp above the measured range
- Imbalance: test min(n0,n1) first; omit a broad 2-D sweep
- Alpha: defer the full alpha grid until the alpha=.05 usefulness gate passes

The JSON companion contains the taper and directional-imbalance tables.
