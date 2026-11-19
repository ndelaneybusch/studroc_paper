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
//! the window width of order `sqrt(n)`. Every stored quantity is a
//! probability in `[0, 1]` formed from nonnegative terms.
//!
//! Each binomial transition is evaluated outward from the largest term that
//! falls inside the live window, whose value comes from a saddle-point
//! formula (`binomial_pmf`). The probability at zero successes is never the
//! starting point, so a gap between breakpoints that holds hundreds of
//! expected observations (constant-margin, released, or repeated bounds)
//! does not underflow. A transition row is dropped only when its largest
//! in-window term is itself below the smallest positive `f64`; dropped mass
//! can only lower the result.

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
        return Err(format!("bound lengths differ: {} vs {}", n, upper.len()));
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
    let odds = q / (1.0 - q);
    for (s, &mass) in p.iter().enumerate() {
        if mass == 0.0 {
            continue;
        }
        let j = off + s;
        let remaining = n - j;
        let k_rem = remaining as f64;
        let max_m = (cap - j).min(remaining);
        // The binomial terms are unimodal, so the in-window term nearest
        // the mode is the largest one in the window.
        let start = binomial_mode(remaining, q).min(max_m);
        let peak = binomial_pmf(remaining, start, q);
        if peak == 0.0 {
            continue;
        }
        out[s + start] += mass * peak;
        let mut pmf = peak;
        for m in start..max_m {
            pmf *= (k_rem - m as f64) / (m as f64 + 1.0) * odds;
            out[s + m + 1] += mass * pmf;
        }
        let mut pmf = peak;
        for m in (1..=start).rev() {
            pmf *= m as f64 / ((k_rem - m as f64 + 1.0) * odds);
            out[s + m - 1] += mass * pmf;
        }
    }
    out
}

/// Mode `floor((k + 1) q)` of the Binomial(`k`, `q`) law, capped at `k`.
// The argument is in [0, k + 1], so the cast cannot lose a sign.
#[allow(clippy::cast_sign_loss)]
fn binomial_mode(k: usize, q: f64) -> usize {
    ((((k + 1) as f64) * q).floor() as usize).min(k)
}

/// Binomial(`k`, `q`) probability of `m` successes, `0 < q < 1`.
///
/// Uses Loader's saddle-point form (the algorithm behind R's `dbinom`),
/// which keeps full relative accuracy where the direct product of a huge
/// binomial coefficient and tiny powers would overflow or underflow.
fn binomial_pmf(k: usize, m: usize, q: f64) -> f64 {
    let (kf, mf) = (k as f64, m as f64);
    if m == 0 {
        return (kf * (-q).ln_1p()).exp();
    }
    if m == k {
        return (kf * q.ln()).exp();
    }
    let log_core = stirling_error(kf)
        - stirling_error(mf)
        - stirling_error(kf - mf)
        - deviance(mf, kf * q)
        - deviance(kf - mf, kf * (1.0 - q));
    let log_scale = (2.0 * std::f64::consts::PI).ln() + mf.ln() + (-mf / kf).ln_1p();
    (log_core - 0.5 * log_scale).exp()
}

/// `ln(x!) - [(x + 1/2) ln x - x + ln(2 pi) / 2]` for a positive integer `x`:
/// the error of Stirling's approximation to the log factorial.
fn stirling_error(x: f64) -> f64 {
    if x < 16.0 {
        let mut log_factorial = 0.0f64;
        let mut i = 2.0f64;
        while i <= x {
            log_factorial += i.ln();
            i += 1.0;
        }
        return log_factorial - (x + 0.5) * x.ln() + x - 0.5 * (2.0 * std::f64::consts::PI).ln();
    }
    let x2 = x * x;
    (1.0 / 12.0
        - (1.0 / 360.0 - (1.0 / 1260.0 - (1.0 / 1680.0 - (1.0 / 1188.0) / x2) / x2) / x2) / x2)
        / x
}

/// `x ln(x / mean) + mean - x`, evaluated without cancellation when `x` is
/// close to `mean`.
// The series stops when a further term no longer changes the sum, which is
// an exact comparison by design.
#[allow(clippy::float_cmp)]
fn deviance(x: f64, mean: f64) -> f64 {
    if (x - mean).abs() < 0.1 * (x + mean) {
        let v = (x - mean) / (x + mean);
        let mut sum = (x - mean) * v;
        let mut term = 2.0 * x * v;
        let v2 = v * v;
        for order in 1..1000 {
            term *= v2;
            let next = sum + term / f64::from(2 * order + 1);
            if next == sum {
                return next;
            }
            sum = next;
        }
        return sum;
    }
    x * (x / mean).ln() + mean - x
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::Xoshiro256pp;

    #[test]
    fn binomial_pmf_matches_direct_evaluation() {
        // Direct evaluation is safe at these sizes and is an independent
        // route to the same number.
        for &(k, q) in &[
            (1usize, 0.3f64),
            (7, 0.01),
            (15, 0.5),
            (16, 0.9),
            (60, 0.37),
            (170, 0.05),
        ] {
            let mut total = 0.0f64;
            for m in 0..=k {
                let mut direct = 1.0f64;
                for step in 0..m {
                    direct *= (k - step) as f64 / (step as f64 + 1.0);
                }
                direct *= q.powi(m as i32) * (1.0 - q).powi((k - m) as i32);
                let got = binomial_pmf(k, m, q);
                assert!(
                    (got - direct).abs() <= 1e-12 * direct,
                    "k={k} m={m} q={q}: {got} vs {direct}"
                );
                total += got;
            }
            assert!((total - 1.0).abs() < 1e-12, "k={k} q={q}: total {total}");
        }
    }

    /// Bounds whose non-crossing probability is a binomial quantity with a
    /// closed form (reference values from `scipy.stats.binom`). Each has a
    /// gap between breakpoints holding hundreds of expected observations.
    #[test]
    fn large_gaps_match_binomial_closed_forms() {
        // At least 960 of 2000 uniforms below 0.5: upper bounds released
        // above rank 960.
        let n = 2000usize;
        let mut upper = vec![1.0f64; n];
        for slot in upper.iter_mut().take(960) {
            *slot = 0.5;
        }
        let got = crossing_prob(&vec![0.0; n], &upper).expect("valid");
        assert!(
            (got - 0.964_959_534_175_958_7).abs() < 1e-10,
            "sf case: {got}"
        );

        // At most 950 of 3000 uniforms below 0.3: lower bounds released
        // below rank 951.
        let n = 3000usize;
        let mut lower = vec![0.0f64; n];
        for slot in lower.iter_mut().skip(950) {
            *slot = 0.3;
        }
        let got = crossing_prob(&lower, &vec![1.0; n]).expect("valid");
        assert!(
            (got - 0.977_473_757_544_557_8).abs() < 1e-10,
            "cdf case: {got}"
        );

        // Exactly 2500 of 5000 uniforms below 0.5: every bound repeated.
        let n = 5000usize;
        let mut lower = vec![0.0f64; n];
        let mut upper = vec![1.0f64; n];
        for slot in upper.iter_mut().take(2500) {
            *slot = 0.5;
        }
        for slot in lower.iter_mut().skip(2500) {
            *slot = 0.5;
        }
        let got = crossing_prob(&lower, &upper).expect("valid");
        assert!(
            (got - 0.011_283_227_495_479_853).abs() < 1e-12,
            "pmf case: {got}"
        );

        // Rare events at large n: at most 3 of 4000 uniforms below 0.004.
        let n = 4000usize;
        let mut lower = vec![0.0f64; n];
        for slot in lower.iter_mut().skip(3) {
            *slot = 0.004;
        }
        let got = crossing_prob(&lower, &vec![1.0; n]).expect("valid");
        assert!(
            (got - 9.115_980_665_952_531e-5).abs() < 1e-14,
            "rare cdf case: {got}"
        );
    }

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
