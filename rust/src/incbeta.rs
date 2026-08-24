//! Regularized incomplete beta function and its inverse.
//!
//! The ladder-profile kernel (`ladder`) evaluates the Clopper-Pearson upper
//! allowance at every trim depth's own local level `j / (M + 1)`, which
//! requires the Beta quantile `B^{-1}(1 - j/(M+1); khat + 1, n1 - khat)` at
//! depths that are only known after the draw-depth pass. Computing it here
//! (rather than shipping tables across the FFI boundary) keeps the kernel a
//! single call. Agreement with scipy's `beta.ppf` is verified to ~1e-12 in
//! the Python test suite.
//!
//! Algorithms: Lentz continued fraction for the incomplete beta and a
//! Halley-refined initial guess for the inverse (Numerical Recipes 6.4),
//! with a bisection fallback for pathological corners.

// Conventional single-letter parameter names of the special-function
// literature (a, b, p, x, ...) and full-precision published constants are
// deliberate here.
#![allow(clippy::many_single_char_names, clippy::excessive_precision)]

use std::f64::consts::PI;

const MAX_CF_ITER: usize = 400;
const CF_EPS: f64 = f64::EPSILON;
const FPMIN: f64 = f64::MIN_POSITIVE / f64::EPSILON;

/// Lanczos (g = 7, n = 9) log-gamma, accurate to ~1e-13 relative.
fn ln_gamma(x: f64) -> f64 {
    const COEF: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula; x is never a non-positive integer in our use.
        return PI.ln() - (PI * x).sin().abs().ln() - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut acc = 0.999_999_999_999_809_93;
    for (i, &c) in COEF.iter().enumerate() {
        acc += c / (z + (i + 1) as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + acc.ln()
}

/// Lentz's continued fraction for the incomplete beta (NR `betacf`).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_CF_ITER {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < CF_EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)` for `a, b > 0`, `x` clamped to
/// `[0, 1]`.
#[must_use]
pub fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Inverse of the regularized incomplete beta: the `x` with
/// `I_x(a, b) = p`, for `a, b > 0` and `p` in `[0, 1]`.
///
/// Halley iterations from the Numerical Recipes 6.4 starting guess, with a
/// bisection fallback if the iteration fails to converge to `~1e-12`.
#[must_use]
pub fn inv_reg_inc_beta(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let mut x = initial_guess(p, a, b).clamp(1e-300, 1.0 - 1e-16);
    let afac = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let a1 = a - 1.0;
    let b1 = b - 1.0;
    let mut converged = false;
    for _ in 0..20 {
        let err = reg_inc_beta(a, b, x) - p;
        let ln_pdf = a1 * x.ln() + b1 * (1.0 - x).ln() + afac;
        let t = ln_pdf.exp();
        if t <= 0.0 || !t.is_finite() {
            break;
        }
        let u = err / t;
        // Halley step; the second-order correction is capped at 1 for
        // stability (as in NR).
        let halley = (u * (a1 / x - b1 / (1.0 - x))).min(1.0);
        let mut xnew = x - u / (1.0 - 0.5 * halley);
        if xnew <= 0.0 {
            xnew = 0.5 * x;
        }
        if xnew >= 1.0 {
            xnew = 0.5 * (x + 1.0);
        }
        let step = (xnew - x).abs();
        x = xnew;
        if step < 1e-15 * x.max(1e-15) {
            converged = true;
            break;
        }
    }
    if !converged && (reg_inc_beta(a, b, x) - p).abs() > 1e-12 {
        x = bisect(p, a, b);
    }
    x
}

fn initial_guess(p: f64, a: f64, b: f64) -> f64 {
    if a >= 1.0 && b >= 1.0 {
        // Abramowitz-Stegun 26.5.22 via the normal quantile approximation.
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut xg = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            xg = -xg;
        }
        let al = (xg * xg - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = xg * (al + h).sqrt() / h
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        a / (a + b * (2.0 * w).exp())
    } else {
        let lna = (a / (a + b)).ln();
        let lnb = (b / (a + b)).ln();
        let t = (a * lna).exp() / a;
        let u = (b * lnb).exp() / b;
        let w = t + u;
        if p < t / w {
            (a * w * p).powf(1.0 / a)
        } else {
            1.0 - (b * w * (1.0 - p)).powf(1.0 / b)
        }
    }
}

fn bisect(p: f64, a: f64, b: f64) -> f64 {
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if reg_inc_beta(a, b, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-16 * hi.max(1e-300) {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_lossless)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials() {
        for n in 1u64..=20 {
            let expect: f64 = (1..n).map(|k| (k as f64).ln()).sum();
            let got = ln_gamma(n as f64);
            assert!(
                (got - expect).abs() < 1e-10 * expect.abs().max(1.0),
                "ln_gamma({n}) = {got}, expected {expect}"
            );
        }
        // Gamma(1/2) = sqrt(pi).
        assert!((ln_gamma(0.5) - 0.5 * PI.ln()).abs() < 1e-12);
    }

    #[test]
    fn incomplete_beta_closed_forms() {
        // I_x(1, b) = 1 - (1 - x)^b; I_x(a, 1) = x^a.
        for &x in &[1e-6f64, 0.01, 0.3, 0.5, 0.9, 1.0 - 1e-9] {
            for &b in &[1.0, 2.0, 17.0, 400.0] {
                let expect = 1.0 - (1.0 - x).powf(b);
                assert!(
                    (reg_inc_beta(1.0, b, x) - expect).abs() < 1e-12,
                    "I_x(1, {b}) at x={x}"
                );
                assert!(
                    (reg_inc_beta(b, 1.0, x) - x.powf(b)).abs() < 1e-12,
                    "I_x({b}, 1) at x={x}"
                );
            }
        }
    }

    #[test]
    fn inverse_roundtrips_across_regimes() {
        // Covers the CP-allowance use (a = khat + 1 >= 1, b = n1 - khat >= 1)
        // including extreme levels and large counts, plus a < 1 corners.
        let ps = [1e-8, 1e-5, 1e-3, 0.05, 0.5, 0.95, 0.999, 1.0 - 1e-8];
        let abs = [
            (1.0, 1.0),
            (1.0, 5000.0),
            (2.0, 3.0),
            (17.0, 400.0),
            (400.0, 17.0),
            (2500.0, 2500.0),
            (50000.0, 1.0),
            (0.5, 0.5),
            (0.3, 7.0),
        ];
        for &(a, b) in &abs {
            for &p in &ps {
                let x = inv_reg_inc_beta(p, a, b);
                assert!(
                    (0.0..=1.0).contains(&x),
                    "x out of range: a={a} b={b} p={p}"
                );
                let back = reg_inc_beta(a, b, x);
                // Near x = 0 or 1 the density can be so steep that no
                // representable x roundtrips exactly; accept an answer that
                // brackets p within a relative-ulp neighborhood of x.
                let ok_direct = (back - p).abs() < 1e-10;
                let x_lo = (x - x.abs() * 4e-16).max(0.0);
                let x_hi = (x + x.abs() * 4e-16 + 1e-300).min(1.0);
                let ok_bracket =
                    reg_inc_beta(a, b, x_lo) - 1e-10 <= p && p <= reg_inc_beta(a, b, x_hi) + 1e-10;
                assert!(
                    ok_direct || ok_bracket,
                    "roundtrip a={a} b={b} p={p}: x={x} back={back}"
                );
            }
        }
    }

    #[test]
    fn inverse_is_monotone_in_p() {
        let (a, b) = (8.0, 493.0);
        let mut prev = 0.0;
        for i in 1..200 {
            let p = i as f64 / 200.0;
            let x = inv_reg_inc_beta(p, a, b);
            assert!(x >= prev, "not monotone at p={p}");
            prev = x;
        }
    }

    #[test]
    fn endpoints_are_exact() {
        assert_eq!(inv_reg_inc_beta(0.0, 3.0, 4.0), 0.0);
        assert_eq!(inv_reg_inc_beta(1.0, 3.0, 4.0), 1.0);
        assert_eq!(reg_inc_beta(3.0, 4.0, 0.0), 0.0);
        assert_eq!(reg_inc_beta(3.0, 4.0, 1.0), 1.0);
    }
}
