//! Ladder-profile kernel for the offline trim-exponent calibration study
//! (`stats/c_calibration_spec.md` section 5.6).
//!
//! For one replicate (one merged label sequence), the calibration study needs
//! the full coverage-vs-depth profile of the fiducial band: at every trim
//! depth `j` of a caller-supplied ladder, whether the allowance-augmented
//! band at depth `j` covers a reference curve, together with the band's area,
//! the draw-depth distribution, the reference curve's own tie-inclusive depth,
//! and the same statistics at "reference-map" depths selected exactly the way
//! production selects them (the `alpha_eff`-quantile of the draw depths,
//! clamped to `[1, M/2]`).
//!
//! Everything is computed from a single fiducial cloud generated and held
//! inside Rust (same draw-indexed RNG as [`crate::fiducial_cloud`], so a
//! given seed yields bit-identical draws to the production tube kernel). Only
//! O(J + M) summaries cross the FFI boundary unless per-depth band edges are
//! explicitly requested; the M x K cloud never does.
//!
//! Band assembly per depth replicates the production wrapper operation
//! order exactly (`fiducial_band_rs`): clip both edges to `[0, 1]`, union
//! the upper edge with the Clopper-Pearson upper allowance at local level
//! `j / (M + 1)` and monotonize it by a running maximum, zero the lower edge
//! wherever the empirical TPR count is zero, re-clip, and pin
//! `lower[0] = 0`, `upper[K-1] = 1`. The CP quantile is the in-crate
//! inverse regularized incomplete beta ([`crate::incbeta`]), which matches
//! scipy's `beta.ppf` to ~1e-12.

use rayon::prelude::*;

use crate::incbeta::inv_reg_inc_beta;
use crate::{fiducial_cloud, gather_block_cols, minp_depths_cols, parse_labels, CHUNK_COLS};

/// Coverage tolerance, matching the experiment harness convention.
const TOL: f64 = 1e-12;

/// Per-depth band statistics against the reference curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandStats {
    /// Reference curve inside the allowance-augmented band at every grid point.
    pub covered: bool,
    /// Reference curve dips below the lower edge somewhere.
    pub viol_low: bool,
    /// Reference curve exceeds the upper edge somewhere.
    pub viol_high: bool,
    /// Largest violation of either edge (0 when covered).
    pub miss_depth: f64,
    /// Grid index of the first largest violation, `-1` when covered.
    pub worst_k: i64,
    /// Mean band width over the grid, allowances applied.
    pub area: f64,
    /// Mean band width of the raw (pre-allowance) tube.
    pub area_raw: f64,
}

/// Per-depth pointwise band edges (raw tube, before allowances).
pub struct Edges {
    /// The depths the rows correspond to (sorted ascending, deduplicated).
    pub depths: Vec<u32>,
    /// Row-major `(depths.len(), K)` lower edges.
    pub lower: Vec<f32>,
    /// Row-major `(depths.len(), K)` upper edges.
    pub upper: Vec<f32>,
}

/// Full ladder profile of one replicate.
pub struct LadderProfile {
    /// Stats at each caller-supplied ladder depth, in ladder order.
    pub ladder_stats: Vec<BandStats>,
    /// Realized production trim depth for each requested `alpha_eff`.
    pub ref_j: Vec<u32>,
    /// Stats at each reference depth, in `alpha_effs` order.
    pub ref_stats: Vec<BandStats>,
    /// Sorted min-p depths of all draws (over the trim columns).
    pub depths_sorted: Vec<u32>,
    /// Reference curve's min tie-inclusive rank from below, over all columns.
    pub truth_depth_low: u32,
    /// Reference curve's min tie-inclusive rank from above, over all columns.
    pub truth_depth_high: u32,
    /// Raw tube edges at every evaluated depth, when requested.
    pub edges: Option<Edges>,
}

struct PassChunk {
    start: usize,
    width: usize,
    lower: Vec<f32>,
    upper: Vec<f32>,
    truth_low: u32,
    truth_high: u32,
}

/// Extract per-column order statistics at `depths` plus the reference
/// curve's tie-inclusive ranks, in parallel column chunks.
fn edge_pass(
    cloud: &[f32],
    n_draws: usize,
    n_grid: usize,
    depths: &[u32],
    rtrue: &[f64],
) -> (Vec<f32>, Vec<f32>, u32, u32) {
    let n_depths = depths.len();
    let starts: Vec<usize> = (0..n_grid).step_by(CHUNK_COLS).collect();
    let chunks: Vec<PassChunk> = starts
        .par_iter()
        .map(|&start| {
            let width = CHUNK_COLS.min(n_grid - start);
            let cols: Vec<usize> = (start..start + width).collect();
            let mut block = vec![0.0f32; width * n_draws];
            gather_block_cols(cloud, n_draws, n_grid, &cols, &mut block);
            let mut lower = vec![0.0f32; n_depths * width];
            let mut upper = vec![0.0f32; n_depths * width];
            let mut truth_low = n_draws as u32;
            let mut truth_high = n_draws as u32;
            for c in 0..width {
                let col = &mut block[c * n_draws..(c + 1) * n_draws];
                col.sort_unstable_by(f32::total_cmp);
                for (di, &d) in depths.iter().enumerate() {
                    lower[di * width + c] = col[d as usize - 1];
                    upper[di * width + c] = col[n_draws - d as usize];
                }
                let rt = rtrue[start + c];
                let le = col.partition_point(|&v| f64::from(v) <= rt) as u32;
                let ge = (n_draws - col.partition_point(|&v| f64::from(v) < rt)) as u32;
                truth_low = truth_low.min(le);
                truth_high = truth_high.min(ge);
            }
            PassChunk {
                start,
                width,
                lower,
                upper,
                truth_low,
                truth_high,
            }
        })
        .collect();

    let mut lower = vec![0.0f32; n_depths * n_grid];
    let mut upper = vec![0.0f32; n_depths * n_grid];
    let mut truth_low = n_draws as u32;
    let mut truth_high = n_draws as u32;
    for ch in chunks {
        for di in 0..n_depths {
            let dst = di * n_grid + ch.start;
            lower[dst..dst + ch.width]
                .copy_from_slice(&ch.lower[di * ch.width..(di + 1) * ch.width]);
            upper[dst..dst + ch.width]
                .copy_from_slice(&ch.upper[di * ch.width..(di + 1) * ch.width]);
        }
        truth_low = truth_low.min(ch.truth_low);
        truth_high = truth_high.min(ch.truth_high);
    }
    (lower, upper, truth_low, truth_high)
}

/// Assemble the production band at one depth and score it against `rtrue`.
#[allow(clippy::cast_possible_wrap)] // grid indices are far below i64::MAX
fn band_stats_at_depth(
    lower_row: &[f32],
    upper_row: &[f32],
    cp_row: &[f64],
    khat: &[u32],
    rtrue: &[f64],
) -> BandStats {
    let n_grid = rtrue.len();
    let mut run_upper = f64::NEG_INFINITY;
    let mut miss_depth = 0.0f64;
    let mut worst_k: i64 = -1;
    let mut viol_low = false;
    let mut viol_high = false;
    let mut area = 0.0f64;
    let mut area_raw = 0.0f64;
    for k in 0..n_grid {
        let lo_raw = f64::from(lower_row[k]).clamp(0.0, 1.0);
        let hi_raw = f64::from(upper_row[k]).clamp(0.0, 1.0);
        run_upper = run_upper.max(hi_raw.max(cp_row[khat[k] as usize]));
        let mut u = run_upper.clamp(0.0, 1.0);
        let mut l = if khat[k] == 0 { 0.0 } else { lo_raw };
        if k == 0 {
            l = 0.0;
        }
        if k == n_grid - 1 {
            u = 1.0;
        }
        area += u - l;
        area_raw += hi_raw - lo_raw;
        let d_lo = l - rtrue[k];
        let d_hi = rtrue[k] - u;
        if d_lo > TOL {
            viol_low = true;
        }
        if d_hi > TOL {
            viol_high = true;
        }
        let viol = d_lo.max(d_hi).max(0.0);
        if viol > miss_depth {
            miss_depth = viol;
            worst_k = k as i64;
        }
    }
    let kf = n_grid as f64;
    BandStats {
        covered: !(viol_low || viol_high),
        viol_low,
        viol_high,
        miss_depth,
        worst_k: if miss_depth > TOL { worst_k } else { -1 },
        area: area / kf,
        area_raw: area_raw / kf,
    }
}

/// Compute the full ladder profile of one replicate.
///
/// `rtrue` is the reference curve on the native grid `t_k = k / n0`
/// (`K = n0 + 1` values in `[0, 1]`); `khat` the empirical TPR counts at
/// the grid points (values in `0..=n1`); `ladder` the strictly increasing
/// trim depths to profile (each in `[1, max(M/2, 1)]`); `alpha_effs` the
/// effective levels at which to evaluate the production depth-selection
/// rule; `trim_cols` an optional strictly increasing subset of grid columns
/// on which the min-p depths are computed (the production thinned-grid
/// rule), with the band always evaluated on the full grid.
///
/// # Errors
///
/// Returns a message when any input is malformed (labels, dimensions,
/// ladder ordering or range, levels outside `(0, 1)`, columns out of range).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn ladder_profile_vec(
    labels: &[u8],
    n_draws: usize,
    seed: u64,
    rtrue: &[f64],
    khat: &[u32],
    ladder: &[u32],
    alpha_effs: &[f64],
    trim_cols: Option<&[usize]>,
    return_edges: bool,
) -> Result<LadderProfile, String> {
    let (n0, n1) = parse_labels(labels)?;
    if n_draws < 2 {
        return Err(format!("n_draws must be at least 2, got {n_draws}"));
    }
    let n_grid = n0 + 1;
    if rtrue.len() != n_grid {
        return Err(format!(
            "rtrue must have n0 + 1 = {n_grid} entries, got {}",
            rtrue.len()
        ));
    }
    if rtrue.iter().any(|&r| !(0.0..=1.0).contains(&r)) {
        return Err("rtrue values must lie in [0, 1]".to_string());
    }
    if khat.len() != n_grid {
        return Err(format!(
            "khat must have n0 + 1 = {n_grid} entries, got {}",
            khat.len()
        ));
    }
    if khat.iter().any(|&v| v as usize > n1) {
        return Err(format!("khat values must be at most n1 = {n1}"));
    }
    let half = (n_draws / 2).max(1);
    if ladder.is_empty() {
        return Err("ladder must be non-empty".to_string());
    }
    if ladder.windows(2).any(|w| w[1] <= w[0]) {
        return Err("ladder must be strictly increasing".to_string());
    }
    if ladder[0] < 1 || ladder[ladder.len() - 1] as usize > half {
        return Err(format!(
            "ladder depths must lie in [1, {half}] for n_draws = {n_draws}"
        ));
    }
    if alpha_effs.iter().any(|&a| !(a > 0.0 && a < 1.0)) {
        return Err("alpha_effs must lie in (0, 1)".to_string());
    }
    if let Some(cols) = trim_cols {
        if cols.is_empty() {
            return Err("trim_cols must be non-empty when given".to_string());
        }
        if cols.windows(2).any(|w| w[1] <= w[0]) {
            return Err("trim_cols must be strictly increasing".to_string());
        }
        if cols[cols.len() - 1] >= n_grid {
            return Err(format!("trim_cols must be less than n_grid = {n_grid}"));
        }
    }

    let cloud = fiducial_cloud(labels, n0, n1, n_draws, seed);

    // Draw depths over the trim columns, then the production depth rule.
    let depth_cols: Vec<usize> = trim_cols.map_or_else(|| (0..n_grid).collect(), <[usize]>::to_vec);
    let mut depths_sorted = minp_depths_cols(&cloud, n_draws, n_grid, &depth_cols);
    depths_sorted.sort_unstable();
    // alpha_eff is validated in (0, 1), so the floored index is in range.
    #[allow(clippy::cast_sign_loss)]
    let ref_j: Vec<u32> = alpha_effs
        .iter()
        .map(|&ae| {
            let q = depths_sorted[(ae * n_draws as f64).floor() as usize] as usize;
            q.clamp(1, half) as u32
        })
        .collect();

    let mut all_depths: Vec<u32> = ladder.to_vec();
    all_depths.extend_from_slice(&ref_j);
    all_depths.sort_unstable();
    all_depths.dedup();

    let (lower, upper, truth_depth_low, truth_depth_high) =
        edge_pass(&cloud, n_draws, n_grid, &all_depths, rtrue);
    drop(cloud);

    // Clopper-Pearson upper allowance per depth, tabulated by khat value.
    let mut present = vec![false; n1 + 1];
    for &v in khat {
        present[v as usize] = true;
    }
    let level_denom = n_draws as f64 + 1.0;
    let cp_rows: Vec<Vec<f64>> = all_depths
        .par_iter()
        .map(|&d| {
            let level = f64::from(d) / level_denom;
            let mut row = vec![1.0f64; n1 + 1];
            for (v, row_v) in row.iter_mut().enumerate().take(n1) {
                if present[v] {
                    *row_v = inv_reg_inc_beta(1.0 - level, (v + 1) as f64, (n1 - v) as f64);
                }
            }
            row
        })
        .collect();

    let stats: Vec<BandStats> = (0..all_depths.len())
        .into_par_iter()
        .map(|di| {
            band_stats_at_depth(
                &lower[di * n_grid..(di + 1) * n_grid],
                &upper[di * n_grid..(di + 1) * n_grid],
                &cp_rows[di],
                khat,
                rtrue,
            )
        })
        .collect();

    let stat_at = |d: u32| {
        let idx = all_depths
            .binary_search(&d)
            .expect("depth present by construction");
        stats[idx]
    };
    let ladder_stats: Vec<BandStats> = ladder.iter().map(|&d| stat_at(d)).collect();
    let ref_stats: Vec<BandStats> = ref_j.iter().map(|&d| stat_at(d)).collect();

    Ok(LadderProfile {
        ladder_stats,
        ref_j,
        ref_stats,
        depths_sorted,
        truth_depth_low,
        truth_depth_high,
        edges: return_edges.then_some(Edges {
            depths: all_depths,
            lower,
            upper,
        }),
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
mod tests {
    use super::*;
    use crate::{minp_depths, trimmed_tube_vec, Xoshiro256pp};

    fn labels_case(n0: usize, n1: usize, seed: u64) -> Vec<u8> {
        let mut rng = Xoshiro256pp::new(seed);
        let mut labels = Vec::with_capacity(n0 + n1);
        let (mut r0, mut r1) = (n0, n1);
        while r0 + r1 > 0 {
            let p1 = r1 as f64 / (r0 + r1) as f64;
            if r0 == 0 || (r1 > 0 && rng.next_f64() < p1) {
                labels.push(1);
                r1 -= 1;
            } else {
                labels.push(0);
                r0 -= 1;
            }
        }
        labels
    }

    /// Staircase-upper empirical TPR counts from the merged label sequence.
    fn khat_from_labels(labels: &[u8], n1: usize) -> Vec<u32> {
        let mut khat = Vec::new();
        let mut cpos = 0u32;
        for &l in labels {
            if l == 1 {
                cpos += 1;
            } else {
                khat.push(cpos);
            }
        }
        khat.push(n1 as u32);
        khat
    }

    /// A smooth monotone reference curve pinned at the endpoints.
    fn reference_curve(n_grid: usize) -> Vec<f64> {
        (0..n_grid)
            .map(|k| {
                let t = k as f64 / (n_grid - 1) as f64;
                t.powf(0.35)
            })
            .collect()
    }

    #[test]
    fn matches_bruteforce_profile() {
        let (n0, n1, n_draws) = (18usize, 13usize, 240usize);
        let labels = labels_case(n0, n1, 12);
        let n_grid = n0 + 1;
        let khat = khat_from_labels(&labels, n1);
        let rtrue = reference_curve(n_grid);
        let ladder: Vec<u32> = vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 120];
        let alpha_effs = [0.0975, 0.25, 0.75];
        let seed = 99u64;

        let prof = ladder_profile_vec(
            &labels,
            n_draws,
            seed,
            &rtrue,
            &khat,
            &ladder,
            &alpha_effs,
            None,
            false,
        )
        .expect("valid");

        // Independent naive reconstruction from the same cloud.
        let cloud = fiducial_cloud(&labels, n0, n1, n_draws, seed);
        let mut depths = minp_depths(&cloud, n_draws, n_grid);
        depths.sort_unstable();
        assert_eq!(prof.depths_sorted, depths, "depth distribution");

        for (&ae, (&j_got, stats_got)) in alpha_effs
            .iter()
            .zip(prof.ref_j.iter().zip(&prof.ref_stats))
        {
            let q = depths[(ae * n_draws as f64).floor() as usize] as usize;
            let j_expect = q.clamp(1, n_draws / 2) as u32;
            assert_eq!(j_got, j_expect, "ref depth at alpha_eff={ae}");
            let naive = naive_stats(&cloud, n_draws, n_grid, j_got, &khat, n1, &rtrue);
            assert_stats_close(*stats_got, naive, &format!("ref alpha_eff={ae}"));
        }
        for (&j, stats_got) in ladder.iter().zip(&prof.ladder_stats) {
            let naive = naive_stats(&cloud, n_draws, n_grid, j, &khat, n1, &rtrue);
            assert_stats_close(*stats_got, naive, &format!("ladder j={j}"));
        }

        // Truth depth: naive tie-inclusive ranks.
        let mut t_lo = n_draws as u32;
        let mut t_hi = n_draws as u32;
        for k in 0..n_grid {
            let le = (0..n_draws)
                .filter(|&m| f64::from(cloud[m * n_grid + k]) <= rtrue[k])
                .count() as u32;
            let ge = (0..n_draws)
                .filter(|&m| f64::from(cloud[m * n_grid + k]) >= rtrue[k])
                .count() as u32;
            t_lo = t_lo.min(le);
            t_hi = t_hi.min(ge);
        }
        assert_eq!(prof.truth_depth_low, t_lo);
        assert_eq!(prof.truth_depth_high, t_hi);
    }

    fn naive_stats(
        cloud: &[f32],
        n_draws: usize,
        n_grid: usize,
        j: u32,
        khat: &[u32],
        n1: usize,
        rtrue: &[f64],
    ) -> BandStats {
        let j = j as usize;
        let mut lo = vec![0.0f64; n_grid];
        let mut hi = vec![0.0f64; n_grid];
        for k in 0..n_grid {
            let mut col: Vec<f32> = (0..n_draws).map(|m| cloud[m * n_grid + k]).collect();
            col.sort_unstable_by(f32::total_cmp);
            lo[k] = f64::from(col[j - 1]).clamp(0.0, 1.0);
            hi[k] = f64::from(col[n_draws - j]).clamp(0.0, 1.0);
        }
        let level = j as f64 / (n_draws as f64 + 1.0);
        let mut u: Vec<f64> = (0..n_grid)
            .map(|k| {
                let cp = if (khat[k] as usize) < n1 {
                    inv_reg_inc_beta(
                        1.0 - level,
                        f64::from(khat[k]) + 1.0,
                        (n1 - khat[k] as usize) as f64,
                    )
                } else {
                    1.0
                };
                hi[k].max(cp)
            })
            .collect();
        for k in 1..n_grid {
            u[k] = u[k].max(u[k - 1]);
        }
        let mut l = lo.clone();
        for k in 0..n_grid {
            if khat[k] == 0 {
                l[k] = 0.0;
            }
            u[k] = u[k].clamp(0.0, 1.0);
        }
        l[0] = 0.0;
        u[n_grid - 1] = 1.0;
        let mut miss = 0.0f64;
        let mut worst = -1i64;
        let (mut vl, mut vh) = (false, false);
        let mut area = 0.0;
        let mut area_raw = 0.0;
        for k in 0..n_grid {
            area += u[k] - l[k];
            area_raw += hi[k] - lo[k];
            let d_lo = l[k] - rtrue[k];
            let d_hi = rtrue[k] - u[k];
            if d_lo > TOL {
                vl = true;
            }
            if d_hi > TOL {
                vh = true;
            }
            let v = d_lo.max(d_hi).max(0.0);
            if v > miss {
                miss = v;
                worst = k as i64;
            }
        }
        BandStats {
            covered: !(vl || vh),
            viol_low: vl,
            viol_high: vh,
            miss_depth: miss,
            worst_k: if miss > TOL { worst } else { -1 },
            area: area / n_grid as f64,
            area_raw: area_raw / n_grid as f64,
        }
    }

    fn assert_stats_close(got: BandStats, expect: BandStats, ctx: &str) {
        assert_eq!(got.covered, expect.covered, "{ctx}: covered");
        assert_eq!(got.viol_low, expect.viol_low, "{ctx}: viol_low");
        assert_eq!(got.viol_high, expect.viol_high, "{ctx}: viol_high");
        assert_eq!(got.worst_k, expect.worst_k, "{ctx}: worst_k");
        assert!(
            (got.miss_depth - expect.miss_depth).abs() < 1e-12,
            "{ctx}: miss_depth {} vs {}",
            got.miss_depth,
            expect.miss_depth
        );
        assert!(
            (got.area - expect.area).abs() < 1e-12,
            "{ctx}: area {} vs {}",
            got.area,
            expect.area
        );
        assert!(
            (got.area_raw - expect.area_raw).abs() < 1e-12,
            "{ctx}: area_raw {} vs {}",
            got.area_raw,
            expect.area_raw
        );
    }

    #[test]
    fn ref_depth_and_edges_match_production_tube() {
        let (n0, n1, n_draws) = (40usize, 30usize, 500usize);
        let labels = labels_case(n0, n1, 7);
        let n_grid = n0 + 1;
        let khat = khat_from_labels(&labels, n1);
        let rtrue = reference_curve(n_grid);
        let seed = 4242u64;
        let alpha_eff = 0.0975;

        let (tube_lo, tube_hi, tube_j) =
            trimmed_tube_vec(&labels, n_draws, alpha_eff, seed, None).expect("valid");
        let prof = ladder_profile_vec(
            &labels,
            n_draws,
            seed,
            &rtrue,
            &khat,
            &[1, 5, 20],
            &[alpha_eff],
            None,
            true,
        )
        .expect("valid");

        assert_eq!(prof.ref_j[0] as usize, tube_j, "trim depth");
        let edges = prof.edges.expect("edges requested");
        let di = edges
            .depths
            .binary_search(&(tube_j as u32))
            .expect("ref depth present");
        for k in 0..n_grid {
            assert_eq!(
                f64::from(edges.lower[di * n_grid + k]),
                tube_lo[k],
                "lower edge at k={k}"
            );
            assert_eq!(
                f64::from(edges.upper[di * n_grid + k]),
                tube_hi[k],
                "upper edge at k={k}"
            );
        }
    }

    #[test]
    fn trim_columns_weakly_deepen_depths() {
        // Min over a subset of columns dominates the min over all columns,
        // so every draw's depth on the thinned grid is >= its full-grid depth.
        let (n0, n1, n_draws) = (60usize, 45usize, 300usize);
        let labels = labels_case(n0, n1, 5);
        let n_grid = n0 + 1;
        let khat = khat_from_labels(&labels, n1);
        let rtrue = reference_curve(n_grid);
        let cols: Vec<usize> = (0..n_grid).step_by(3).collect();

        let full = ladder_profile_vec(
            &labels,
            n_draws,
            11,
            &rtrue,
            &khat,
            &[1, 4, 16],
            &[0.1],
            None,
            false,
        )
        .expect("valid");
        let thinned = ladder_profile_vec(
            &labels,
            n_draws,
            11,
            &rtrue,
            &khat,
            &[1, 4, 16],
            &[0.1],
            Some(&cols),
            false,
        )
        .expect("valid");
        for (a, b) in thinned.depths_sorted.iter().zip(&full.depths_sorted) {
            assert!(a >= b, "thinned depth {a} below full-grid depth {b}");
        }
        assert!(thinned.ref_j[0] >= full.ref_j[0]);
        // Truth depth is over all columns regardless of trimming.
        assert_eq!(thinned.truth_depth_low, full.truth_depth_low);
        assert_eq!(thinned.truth_depth_high, full.truth_depth_high);
    }

    #[test]
    fn rejects_malformed_inputs() {
        let labels = labels_case(10, 8, 1);
        let n_grid = 11usize;
        let rtrue = reference_curve(n_grid);
        let khat = khat_from_labels(&labels, 8);
        let ok = |lad: &[u32], ae: &[f64], cols: Option<&[usize]>| {
            ladder_profile_vec(&labels, 100, 0, &rtrue, &khat, lad, ae, cols, false)
        };
        assert!(ok(&[1, 2], &[0.1], None).is_ok());
        assert!(ok(&[], &[0.1], None).is_err(), "empty ladder");
        assert!(ok(&[2, 2], &[0.1], None).is_err(), "non-increasing ladder");
        assert!(ok(&[0, 2], &[0.1], None).is_err(), "depth below 1");
        assert!(ok(&[1, 51], &[0.1], None).is_err(), "depth beyond M/2");
        assert!(ok(&[1], &[0.0], None).is_err(), "alpha_eff at 0");
        assert!(ok(&[1], &[1.0], None).is_err(), "alpha_eff at 1");
        assert!(ok(&[1], &[0.1], Some(&[])).is_err(), "empty trim_cols");
        assert!(
            ok(&[1], &[0.1], Some(&[3, 3])).is_err(),
            "duplicate trim col"
        );
        assert!(
            ok(&[1], &[0.1], Some(&[0, 11])).is_err(),
            "trim col out of range"
        );
        let short_rtrue = vec![0.5; n_grid - 1];
        assert!(
            ladder_profile_vec(
                &labels,
                100,
                0,
                &short_rtrue,
                &khat,
                &[1],
                &[0.1],
                None,
                false
            )
            .is_err(),
            "rtrue length"
        );
        let bad_khat = vec![9u32; n_grid];
        assert!(
            ladder_profile_vec(
                &labels,
                100,
                0,
                &rtrue,
                &bad_khat,
                &[1],
                &[0.1],
                None,
                false
            )
            .is_err(),
            "khat above n1"
        );
    }
}
