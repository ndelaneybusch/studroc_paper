//! Exact non-crossing probabilities for uniform order statistics.
//!
//! This is the calibration kernel of the M3 composition band
//! (`studroc_paper.methods.m3_band_rs`): the simultaneous coverage of a
//! one-sample equal-local-levels band is `P(l_i <= U_(i) <= h_i for all i)`
//! for iid Uniform(0,1) order statistics `U_(1) <= ... <= U_(n)`, and this
//! module computes that probability *exactly* (to f64 roundoff), replacing
//! the Monte Carlo calibration + safety shading of the experimental
//! implementation.
//!
//! Algorithm: sweep the sorted union of the bound values with the counting
//! process `N(c) = #{U_j <= c}`. The constraints are
//! `N(l_i) <= i - 1` (from `U_(i) >= l_i`) and `N(h_i) >= i` (from
//! `U_(i) <= h_i`), ties having probability zero. Conditionally on
//! `N(c) = j`, the remaining `n - j` points are iid uniform on `(c, 1)`, so
//! `N` is Markov across breakpoints with binomial increments. Because both
//! bound sequences are non-decreasing, the alive states at any moment form
//! the contiguous window `[#upper bounds passed, #lower bounds pending]`:
//! a state above the window would violate the next lower-bound constraint
//! (counts never decrease), and a state below it has already violated a
//! passed upper-bound constraint. The DP therefore costs
//! `O(sum of window width^2)` — `O(n)` per step at fixed local level, with
//! the window width of order `sqrt(n)` — and every intermediate quantity is
//! a probability in `[0, 1]`, so there is no underflow: expected jumps
//! between consecutive breakpoints are `O(log n)` because the bounds
//! themselves are spaced `O(polylog(n) / n)` apart.

/// Exact `P(lower[i] <= U_(i+1) <= upper[i] for all i)` for the order
/// statistics of `n = lower.len()` iid Uniform(0,1) variables.
///
/// Both bound vectors must be non-decreasing (Beta-quantile bands are) with
/// `0 <= lower[i]` and `upper[i] <= 1`. Pointwise crossed bounds
/// (`lower[i] > upper[i]`) make the event empty and return 0.
///
/// # Errors
///
/// Returns a message when the inputs are empty, of unequal length, outside
/// `[0, 1]`, or not non-decreasing.
pub fn crossing_prob(lower: &[f64], upper: &[f64]) -> Result<f64, String> {
    let n = lower.len();
    if n == 0 {
        return Err("bounds must be non-empty".to_string());
    }
    if upper.len() != n {
        return Err(format!(
            "bound lengths differ: {} vs {}",
            n,
            upper.len()
        ));
    }
    if lower[0] < 0.0 || upper[n - 1] > 1.0 {
        return Err("bounds must lie in [0, 1]".to_string());
    }
    if lower.windows(2).any(|w| w[1] < w[0]) || upper.windows(2).any(|w| w[1] < w[0]) {
        return Err("bounds must be non-decreasing".to_string());
    }
    if lower.iter().zip(upper).any(|(l, h)| l > h) {
        return Ok(0.0); // pointwise crossed bounds: the event is empty
    }

    // State: p[j - off] = P(N(c) = j, all constraints at breakpoints <= c
    // hold), for j in the alive window [off, off + p.len() - 1] = [ih, il],
    // where il = #lower bounds processed, ih = #upper bounds processed.
    let mut p = vec![1.0f64];
    let mut off = 0usize;
    let mut il = 0usize;
    let mut ih = 0usize;
    let mut c_prev = 0.0f64;

    while ih < n {
        let c = if il < n {
            lower[il].min(upper[ih])
        } else {
            upper[ih]
        };

        // Binomial transition from c_prev to c; targets capped at il (any
        // higher count is doomed at the next lower-bound breakpoint).
        if c > c_prev {
            let q = (c - c_prev) / (1.0 - c_prev);
            p = transition(&p, off, il, n, q);
        }
        c_prev = c;

        // Constraints AT c. Lower bounds: N(lower[i]) <= i held already via
        // the target cap, so processing just raises the cap for the future.
        while il < n && lower[il] <= c {
            il += 1;
        }
        // Upper bounds: N(upper[i]) >= i + 1 zeroes the low states now.
        while ih < n && upper[ih] <= c {
            ih += 1;
        }
        if ih > off {
            let cut = (ih - off).min(p.len());
            p.drain(..cut);
            off = ih;
        }
        if p.is_empty() {
            return Ok(0.0);
        }
    }
    // All n upper bounds processed forces N = n: the window is [n, n].
    Ok(p.iter().sum::<f64>().clamp(0.0, 1.0))
}

/// One binomial spreading step: mass at count `j` moves to `j + m` with
/// `m ~ Binomial(n - j, q)`, truncated to targets `<= cap` (excess mass is
/// dropped — those paths die at the next lower-bound constraint).
fn transition(p: &[f64], off: usize, cap: usize, n: usize, q: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; cap - off + 1];
    if q >= 1.0 {
        // Everything remaining falls below c: N jumps straight to n.
        if cap == n {
            let total: f64 = p.iter().sum();
            *out.last_mut().expect("non-empty window") += total;
        }
        return out;
    }
    let log1mq = (-q).ln_1p();
    let odds = q / (1.0 - q);
    for (s, &mass) in p.iter().enumerate() {
        if mass == 0.0 {
            continue;
        }
        let j = off + s;
        let k_rem = (n - j) as f64;
        let max_m = (cap - j).min(n - j);
        let mut pmf = (k_rem * log1mq).exp(); // Binomial(n - j, q) at m = 0
        for m in 0..=max_m {
            out[s + m] += mass * pmf;
            pmf *= (k_rem - m as f64) / (m as f64 + 1.0) * odds;
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::Xoshiro256pp;

    #[test]
    fn n1_is_interval_length() {
        for &(l, h) in &[(0.0, 1.0), (0.2, 0.9), (0.5, 0.5)] {
            let p = crossing_prob(&[l], &[h]).expect("valid");
            assert!((p - (h - l)).abs() < 1e-15, "l={l} h={h} p={p}");
        }
    }

    #[test]
    fn n2_matches_closed_form() {
        // P = 2 * int_{l1}^{h1} max(0, h2 - max(l2, u)) du, u constrained to
        // u <= h2 implicitly by the max.
        let cases = [
            ([0.1, 0.3], [0.6, 0.9]),
            ([0.0, 0.5], [0.5, 1.0]),
            ([0.2, 0.2], [0.8, 0.8]),
        ];
        for (low, high) in cases {
            let prob = crossing_prob(&low, &high).expect("valid");
            let steps = 2_000_000usize;
            let mut acc = 0.0f64;
            let du = (high[0] - low[0]) / steps as f64;
            for step in 0..steps {
                let mid = low[0] + (step as f64 + 0.5) * du;
                acc += (high[1] - low[1].max(mid)).max(0.0) * du;
            }
            let expect = 2.0 * acc;
            assert!(
                (prob - expect).abs() < 1e-5,
                "l={low:?} h={high:?}: dp {prob} vs quadrature {expect}"
            );
        }
    }

    #[test]
    fn trivial_and_empty_events() {
        let n = 20usize;
        let zeros = vec![0.0; n];
        let ones = vec![1.0; n];
        assert_eq!(crossing_prob(&zeros, &ones).expect("valid"), 1.0);
        // Crossed bounds somewhere: probability zero.
        let mut l = zeros.clone();
        let mut h = ones;
        l[10] = 0.9;
        l[11] = 0.9; // keep non-decreasing beyond the bump
        for slot in l.iter_mut().skip(12) {
            *slot = 0.9;
        }
        h[10] = 0.1;
        for slot in h.iter_mut().take(10) {
            *slot = 0.1;
        }
        assert_eq!(crossing_prob(&l, &h).expect("valid"), 0.0);
    }

    #[test]
    fn monotone_in_bandwidth() {
        let n = 15usize;
        let band = |w: f64| -> (Vec<f64>, Vec<f64>) {
            let l: Vec<f64> = (1..=n)
                .map(|i| (i as f64 / (n + 1) as f64 - w).max(0.0))
                .collect();
            let h: Vec<f64> = (1..=n)
                .map(|i| (i as f64 / (n + 1) as f64 + w).min(1.0))
                .collect();
            (l, h)
        };
        let mut prev = 0.0;
        for &w in &[0.05, 0.1, 0.2, 0.4, 1.0] {
            let (l, h) = band(w);
            let p = crossing_prob(&l, &h).expect("valid");
            assert!(p >= prev, "not monotone at w={w}: {p} < {prev}");
            prev = p;
        }
        assert!((prev - 1.0).abs() < 1e-12, "full-width band must be sure");
    }

    #[test]
    fn matches_monte_carlo() {
        let n = 30usize;
        let low: Vec<f64> = (1..=n)
            .map(|i| (i as f64 / (n + 1) as f64 - 0.15).max(0.0))
            .collect();
        let high: Vec<f64> = (1..=n)
            .map(|i| (i as f64 / (n + 1) as f64 + 0.15).min(1.0))
            .collect();
        let prob = crossing_prob(&low, &high).expect("valid");

        let reps = 400_000usize;
        let mut rng = Xoshiro256pp::new(20_240_823);
        let mut hits = 0usize;
        let mut sample = vec![0.0f64; n];
        for _ in 0..reps {
            for slot in &mut sample {
                *slot = rng.next_f64();
            }
            sample.sort_unstable_by(f64::total_cmp);
            if sample
                .iter()
                .zip(low.iter().zip(&high))
                .all(|(&v, (&lo, &hi))| v >= lo && v <= hi)
            {
                hits += 1;
            }
        }
        let mc = hits as f64 / reps as f64;
        let se = (mc * (1.0 - mc) / reps as f64).sqrt();
        assert!(
            (prob - mc).abs() < 5.0 * se + 1e-9,
            "dp {prob} vs mc {mc} (se {se})"
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(crossing_prob(&[], &[]).is_err());
        assert!(crossing_prob(&[0.1], &[0.5, 0.6]).is_err());
        assert!(crossing_prob(&[-0.1], &[0.5]).is_err());
        assert!(crossing_prob(&[0.1], &[1.5]).is_err());
        assert!(crossing_prob(&[0.3, 0.2], &[0.5, 0.6]).is_err());
        assert!(crossing_prob(&[0.1, 0.2], &[0.6, 0.5]).is_err());
    }
}
