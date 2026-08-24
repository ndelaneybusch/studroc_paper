//! Fiducial ROC band kernel: the Monte Carlo core of the rank-space fiducial
//! confidence band (`studroc_paper.methods.fiducial_band`).
//!
//! Given the merged label sequence (class labels sorted by descending score,
//! ties already broken), the kernel:
//!
//! 1. draws `n_draws` fiducial ROC curves — per class a Dirichlet(1,...,1)
//!    spacings vector (normalized cumulative exponentials) places that
//!    class's CDF at its own order statistics, the other class's within-gap
//!    elements are spread at sorted-uniform fractions of the gap, and the
//!    resulting (x, y) polyline is linearly interpolated onto the grid
//!    `t_k = k / n0`;
//! 2. computes each draw's min-p depth (minimum over grid points of its
//!    tie-inclusive rank from either end of the cloud) and the trim depth
//!    `j` = the `alpha_eff`-quantile of the depths, clamped to
//!    `[1, n_draws / 2]`;
//! 3. returns the pointwise `j`-th smallest / `j`-th largest cloud values.
//!
//! The exact binomial corner allowances, tie-breaking, and output-grid
//! resampling stay in the Python wrapper — they are O(K) and need scipy's
//! Beta quantile.
//!
//! The cloud is stored as `f32`: the band edges are order statistics whose
//! Monte Carlo resolution is `1 / n_draws >= 5e-5`, three orders above f32
//! granularity, and halving the memory traffic roughly doubles the
//! throughput of the sort-heavy rank passes. Curve generation and
//! interpolation run in `f64` and round once on store.
//!
//! Reproducibility contract: the RNG stream is a pure function of
//! `(seed, draw_index)`, so output is bit-identical regardless of thread
//! count or scheduling.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

pub mod ell;

const CHUNK_COLS: usize = 128; // grid-column chunk for the rank passes

#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        Self {
            s: [
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
            ],
        }
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    #[inline(always)]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard exponential; `1 - u` is in `(0, 1]` so the log is finite.
    #[inline(always)]
    fn next_exp(&mut self) -> f64 {
        -(1.0 - self.next_f64()).ln()
    }
}

/// Derives a unique RNG seed per fiducial draw so results are reproducible
/// across thread counts.
#[inline]
#[must_use]
pub fn draw_seed(seed: u64, draw: u64) -> u64 {
    seed ^ draw.wrapping_mul(0xA24B_AED4_963E_E407)
}

/// Normalized cumulative sums of iid exponentials: the cumulative masses of
/// a Dirichlet(1,...,1) spacings vector. The last entry is pinned to 1 so
/// downstream gap arithmetic sees an exact unit total.
fn dirichlet_cumsum(rng: &mut Xoshiro256pp, out: &mut [f64]) {
    let mut acc = 0.0;
    for slot in out.iter_mut() {
        acc += rng.next_exp();
        *slot = acc;
    }
    for slot in out.iter_mut() {
        *slot /= acc;
    }
    *out.last_mut().expect("non-empty spacings") = 1.0;
}

/// Fill one axis of the fiducial polyline (`out[1..=n]`; the caller owns the
/// 0/1 endpoints at `out[0]` and `out[n + 1]`).
///
/// Elements of the axis-owning class sit at their Dirichlet cumulative
/// masses; each maximal run of the other class is spread at sorted-uniform
/// fractions of the gap it falls in.
fn fill_axis(
    labels: &[u8],
    own_label: u8,
    spacings_cum: &[f64],
    out: &mut [f64],
    rng: &mut Xoshiro256pp,
    urun: &mut Vec<f64>,
) {
    let n = labels.len();
    let mut own_count = 0usize;
    let mut i = 0usize;
    while i < n {
        if labels[i] == own_label {
            out[1 + i] = spacings_cum[own_count];
            own_count += 1;
            i += 1;
        } else {
            let start = i;
            while i < n && labels[i] != own_label {
                i += 1;
            }
            let base = if own_count > 0 {
                spacings_cum[own_count - 1]
            } else {
                0.0
            };
            let mass = spacings_cum[own_count] - base;
            urun.clear();
            urun.extend((start..i).map(|_| rng.next_f64()));
            urun.sort_unstable_by(f64::total_cmp);
            for (offset, &u) in urun.iter().enumerate() {
                out[1 + start + offset] = base + u * mass;
            }
        }
    }
}

/// Evaluate the polyline `(xv, yv)` at the grid `t_k = k / n0` by linear
/// interpolation. `xv` is non-decreasing with `xv[0] = 0` and a final 1, so
/// a single forward pointer resolves every grid point.
fn interpolate_row(xv: &[f64], yv: &[f64], n0: usize, out_row: &mut [f32]) {
    let len = xv.len();
    let mut idx = 0usize; // number of xv elements <= t, monotone in t
    for (k, slot) in out_row.iter_mut().enumerate() {
        let t = k as f64 / n0 as f64;
        while idx < len && xv[idx] <= t {
            idx += 1;
        }
        let i = idx.clamp(1, len - 1);
        let x1 = xv[i - 1];
        let x2 = xv[i];
        let frac = ((t - x1) / (x2 - x1).max(1e-300)).clamp(0.0, 1.0);
        *slot = (yv[i - 1] + frac * (yv[i] - yv[i - 1])) as f32;
    }
}

struct DrawBufs {
    pc: Vec<f64>,
    qc: Vec<f64>,
    xv: Vec<f64>,
    yv: Vec<f64>,
    urun: Vec<f64>,
}

impl DrawBufs {
    fn new(n0: usize, n1: usize) -> Self {
        let n = n0 + n1;
        Self {
            pc: vec![0.0; n0 + 1],
            qc: vec![0.0; n1 + 1],
            xv: vec![0.0; n + 2],
            yv: vec![0.0; n + 2],
            urun: Vec::with_capacity(n),
        }
    }
}

/// One fiducial draw: Dirichlet spacings per class, within-gap spreading,
/// polyline evaluation on the grid.
fn generate_row(
    rng: &mut Xoshiro256pp,
    labels: &[u8],
    n0: usize,
    buf: &mut DrawBufs,
    out_row: &mut [f32],
) {
    let n = labels.len();
    dirichlet_cumsum(rng, &mut buf.pc);
    dirichlet_cumsum(rng, &mut buf.qc);
    buf.xv[0] = 0.0;
    buf.yv[0] = 0.0;
    fill_axis(labels, 0, &buf.pc, &mut buf.xv, rng, &mut buf.urun);
    fill_axis(labels, 1, &buf.qc, &mut buf.yv, rng, &mut buf.urun);
    buf.xv[n + 1] = 1.0;
    buf.yv[n + 1] = 1.0;
    interpolate_row(&buf.xv, &buf.yv, n0, out_row);
}

/// Draw the full fiducial cloud, row-major `(n_draws, n0 + 1)`.
pub fn fiducial_cloud(labels: &[u8], n0: usize, n1: usize, n_draws: usize, seed: u64) -> Vec<f32> {
    let n_grid = n0 + 1;
    let mut cloud = vec![0.0f32; n_draws * n_grid];
    cloud.par_chunks_mut(n_grid).enumerate().for_each_init(
        || DrawBufs::new(n0, n1),
        |buf, (draw, row)| {
            let mut rng = Xoshiro256pp::new(draw_seed(seed, draw as u64));
            generate_row(&mut rng, labels, n0, buf, row);
        },
    );
    cloud
}

/// Copy grid columns `[lo, lo + width)` of the row-major cloud into a
/// column-major block (`block[c * n_draws + m]`). The row-major reads are
/// contiguous and the `width` write streams stay cache-resident, so this is
/// the transpose that makes the per-column passes bandwidth-bound instead of
/// latency-bound.
fn gather_block(cloud: &[f32], n_draws: usize, n_grid: usize, lo: usize, block: &mut [f32]) {
    let width = block.len() / n_draws;
    for m in 0..n_draws {
        let src = &cloud[m * n_grid + lo..m * n_grid + lo + width];
        for (c, &v) in src.iter().enumerate() {
            block[c * n_draws + m] = v;
        }
    }
}

/// Min-p depth of each draw: the minimum over grid points of its
/// tie-inclusive rank from either end of the cloud (`min(#{<= v}, #{>= v})`).
pub fn minp_depths(cloud: &[f32], n_draws: usize, n_grid: usize) -> Vec<u32> {
    let starts: Vec<usize> = (0..n_grid).step_by(CHUNK_COLS).collect();
    starts
        .par_iter()
        .map(|&lo| {
            let width = CHUNK_COLS.min(n_grid - lo);
            let mut block = vec![0.0f32; width * n_draws];
            gather_block(cloud, n_draws, n_grid, lo, &mut block);
            let mut depths = vec![n_draws as u32; n_draws];
            let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(n_draws);
            for c in 0..width {
                let col = &block[c * n_draws..(c + 1) * n_draws];
                pairs.clear();
                pairs.extend(col.iter().copied().zip(0..n_draws as u32));
                pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                // One pass over tie groups: every member of a group of equal
                // values shares rank_le = group end and rank_ge = n - start.
                let mut start = 0usize;
                while start < n_draws {
                    let v = pairs[start].0;
                    let mut end = start + 1;
                    while end < n_draws && pairs[end].0.total_cmp(&v).is_eq() {
                        end += 1;
                    }
                    let d = (end as u32).min((n_draws - start) as u32);
                    for &(_, m) in &pairs[start..end] {
                        let slot = &mut depths[m as usize];
                        if d < *slot {
                            *slot = d;
                        }
                    }
                    start = end;
                }
            }
            depths
        })
        .reduce(
            || vec![n_draws as u32; n_draws],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    if y < *x {
                        *x = y;
                    }
                }
                a
            },
        )
}

/// A column chunk's start offset plus its lower/upper order-stat values.
type ChunkEdges = (usize, Vec<f64>, Vec<f64>);

/// Per-column `j`-th smallest and `j`-th largest cloud values (1-indexed).
pub fn pointwise_order_stats(
    cloud: &[f32],
    n_draws: usize,
    n_grid: usize,
    j: usize,
) -> (Vec<f64>, Vec<f64>) {
    let starts: Vec<usize> = (0..n_grid).step_by(CHUNK_COLS).collect();
    let per_chunk: Vec<ChunkEdges> = starts
        .par_iter()
        .map(|&lo| {
            let width = CHUNK_COLS.min(n_grid - lo);
            let mut block = vec![0.0f32; width * n_draws];
            gather_block(cloud, n_draws, n_grid, lo, &mut block);
            let mut lower = Vec::with_capacity(width);
            let mut upper = Vec::with_capacity(width);
            for c in 0..width {
                let col = &mut block[c * n_draws..(c + 1) * n_draws];
                let (_, lo_v, _) = col.select_nth_unstable_by(j - 1, f32::total_cmp);
                lower.push(f64::from(*lo_v));
                let (_, hi_v, _) = col.select_nth_unstable_by(n_draws - j, f32::total_cmp);
                upper.push(f64::from(*hi_v));
            }
            (lo, lower, upper)
        })
        .collect();
    let mut lower = vec![0.0f64; n_grid];
    let mut upper = vec![0.0f64; n_grid];
    for (lo, l, u) in per_chunk {
        lower[lo..lo + l.len()].copy_from_slice(&l);
        upper[lo..lo + u.len()].copy_from_slice(&u);
    }
    (lower, upper)
}

/// The trimmed fiducial tube on the native grid `t_k = k / n0`.
///
/// Returns `(lower, upper, j)`: the pointwise `j`-th smallest / largest
/// fiducial draws, where `j` is the `alpha_eff`-quantile of the min-p depths
/// clamped to `[1, n_draws / 2]`. Corner allowances are the caller's job.
///
/// # Errors
///
/// Returns a message when `labels` is empty or contains values other than
/// 0/1, either class is absent, `n_draws < 2`, or `alpha_eff` is outside
/// `(0, 1)`.
pub fn trimmed_tube_vec(
    labels: &[u8],
    n_draws: usize,
    alpha_eff: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
    if labels.iter().any(|&l| l > 1) {
        return Err("labels must contain only 0 and 1".to_string());
    }
    let n1: usize = labels.iter().map(|&l| usize::from(l)).sum();
    let n0 = labels.len() - n1;
    if n0 == 0 || n1 == 0 {
        return Err(format!("both classes must be present (n0={n0}, n1={n1})"));
    }
    if n_draws < 2 {
        return Err(format!("n_draws must be at least 2, got {n_draws}"));
    }
    if !(alpha_eff > 0.0 && alpha_eff < 1.0) {
        return Err(format!("alpha_eff must be in (0, 1), got {alpha_eff}"));
    }

    let n_grid = n0 + 1;
    let cloud = fiducial_cloud(labels, n0, n1, n_draws, seed);

    let mut depths = minp_depths(&cloud, n_draws, n_grid);
    depths.sort_unstable();
    // alpha_eff is validated positive, so the floored index cannot be negative.
    #[allow(clippy::cast_sign_loss)]
    let quantile = depths[(alpha_eff * n_draws as f64).floor() as usize] as usize;
    let j = quantile.clamp(1, (n_draws / 2).max(1));

    let (lower, upper) = pointwise_order_stats(&cloud, n_draws, n_grid, j);
    Ok((lower, upper, j))
}

/// Run the kernel, on a dedicated pool when `n_threads > 0`, else the global
/// rayon pool. Output is identical either way (draw-indexed RNG streams).
///
/// # Errors
///
/// As [`trimmed_tube_vec`], plus thread-pool construction failures.
pub fn trimmed_tube_threaded(
    labels: &[u8],
    n_draws: usize,
    alpha_eff: f64,
    seed: u64,
    n_threads: usize,
) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
    if n_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| e.to_string())?
            .install(|| trimmed_tube_vec(labels, n_draws, alpha_eff, seed))
    } else {
        trimmed_tube_vec(labels, n_draws, alpha_eff, seed)
    }
}

/// Lower edge, upper edge, and realized trim depth returned to Python.
type PyTube<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, usize);

/// Trimmed fiducial tube `(lower, upper, j)` on the grid `t = arange(n0 + 1) / n0`.
// PyO3 extracts arguments by value; passing references is not an option here.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (labels, n_draws, alpha_eff, seed, n_threads))]
fn fiducial_trimmed_tube<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray1<'py, u8>,
    n_draws: usize,
    alpha_eff: f64,
    seed: u64,
    n_threads: usize,
) -> PyResult<PyTube<'py>> {
    let labels = labels.as_slice()?.to_vec();
    let (lower, upper, j) = py
        .detach(|| trimmed_tube_threaded(&labels, n_draws, alpha_eff, seed, n_threads))
        .map_err(PyValueError::new_err)?;
    Ok((lower.into_pyarray(py), upper.into_pyarray(py), j))
}

/// Exact `P(lower[i] <= U_(i+1) <= upper[i] for all i)` for the order
/// statistics of `n = len(lower)` iid Uniform(0,1) variables — the
/// calibration kernel of the M3 composition band (see `ell::crossing_prob`).
// PyO3 extracts arguments by value; passing references is not an option here.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (lower, upper))]
fn ell_crossing_probability(
    py: Python<'_>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let lower = lower.as_slice()?.to_vec();
    let upper = upper.as_slice()?.to_vec();
    py.detach(|| ell::crossing_prob(&lower, &upper))
        .map_err(PyValueError::new_err)
}

#[pymodule]
fn fiducial_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fiducial_trimmed_tube, m)?)?;
    m.add_function(wrap_pyfunction!(ell_crossing_probability, m)?)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_sign_loss)]
mod tests {
    use super::*;

    fn labels_case(n0: usize, n1: usize, seed: u64) -> Vec<u8> {
        // Interleave with a bias so runs of both classes appear.
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

    #[test]
    fn cloud_rows_are_monotone_unit_curves() {
        for &(n0, n1) in &[(1usize, 1usize), (5, 3), (40, 25)] {
            let labels = labels_case(n0, n1, 11);
            let n_grid = n0 + 1;
            let cloud = fiducial_cloud(&labels, n0, n1, 200, 7);
            for m in 0..200 {
                let row = &cloud[m * n_grid..(m + 1) * n_grid];
                assert_eq!(row[0], 0.0, "curve must start at (0, 0)");
                assert_eq!(row[n_grid - 1], 1.0, "curve must end at (1, 1)");
                for w in row.windows(2) {
                    assert!(w[0] <= w[1], "curve must be non-decreasing");
                }
                assert!(row.iter().all(|&v| (0.0..=1.0).contains(&v)));
            }
        }
    }

    #[test]
    fn depths_match_bruteforce_ranks() {
        let (n0, n1, n_draws) = (12usize, 9usize, 64usize);
        let labels = labels_case(n0, n1, 3);
        let n_grid = n0 + 1;
        let cloud = fiducial_cloud(&labels, n0, n1, n_draws, 5);
        let depths = minp_depths(&cloud, n_draws, n_grid);
        for m in 0..n_draws {
            let mut expect = n_draws as u32;
            for k in 0..n_grid {
                let v = cloud[m * n_grid + k];
                let le = (0..n_draws).filter(|&r| cloud[r * n_grid + k] <= v).count() as u32;
                let ge = (0..n_draws).filter(|&r| cloud[r * n_grid + k] >= v).count() as u32;
                expect = expect.min(le.min(ge));
            }
            assert_eq!(depths[m], expect, "draw {m}");
        }
    }

    #[test]
    fn order_stats_match_full_sort() {
        let (n0, n1, n_draws) = (17usize, 8usize, 128usize);
        let labels = labels_case(n0, n1, 21);
        let n_grid = n0 + 1;
        let cloud = fiducial_cloud(&labels, n0, n1, n_draws, 9);
        for j in [1usize, 4, 33, n_draws / 2] {
            let (lower, upper) = pointwise_order_stats(&cloud, n_draws, n_grid, j);
            for k in 0..n_grid {
                let mut col: Vec<f32> = (0..n_draws).map(|m| cloud[m * n_grid + k]).collect();
                col.sort_unstable_by(f32::total_cmp);
                assert_eq!(lower[k], f64::from(col[j - 1]), "j={j} k={k}");
                assert_eq!(upper[k], f64::from(col[n_draws - j]), "j={j} k={k}");
            }
        }
    }

    #[test]
    fn depth_tube_duality_and_content_control() {
        // Lemma: a draw lies inside the [j-th smallest, j-th largest] tube at
        // every grid point iff its min-p depth is >= j; the trimmed tube must
        // retain at least a 1 - alpha_eff fraction of the cloud.
        let (n0, n1, n_draws) = (20usize, 15usize, 400usize);
        let alpha_eff = 0.0975;
        let labels = labels_case(n0, n1, 2);
        let n_grid = n0 + 1;
        let cloud = fiducial_cloud(&labels, n0, n1, n_draws, 31);
        let depths = minp_depths(&cloud, n_draws, n_grid);
        let (lower, upper, j) = trimmed_tube_vec(&labels, n_draws, alpha_eff, 31).expect("valid");
        let mut inside_count = 0usize;
        for m in 0..n_draws {
            let row = &cloud[m * n_grid..(m + 1) * n_grid];
            let inside = row
                .iter()
                .enumerate()
                .all(|(k, &v)| f64::from(v) >= lower[k] && f64::from(v) <= upper[k]);
            assert_eq!(
                inside,
                depths[m] as usize >= j,
                "duality violated at draw {m} (depth {}, j {j})",
                depths[m]
            );
            inside_count += usize::from(inside);
        }
        let retained = inside_count as f64 / n_draws as f64;
        assert!(
            retained >= 1.0 - alpha_eff,
            "tube content {retained} below 1 - alpha_eff"
        );
    }

    #[test]
    fn output_is_independent_of_thread_count() {
        let labels = labels_case(60, 45, 4);
        let reference = trimmed_tube_threaded(&labels, 500, 0.0975, 42, 1).expect("valid");
        for threads in [2usize, 4, 0] {
            let out = trimmed_tube_threaded(&labels, 500, 0.0975, 42, threads).expect("valid");
            assert_eq!(out, reference, "thread count {threads} changed the output");
        }
    }

    #[test]
    fn rejects_invalid_inputs() {
        let ok = labels_case(5, 5, 1);
        assert!(trimmed_tube_vec(&[], 100, 0.1, 0).is_err());
        assert!(trimmed_tube_vec(&[0, 0, 0], 100, 0.1, 0).is_err());
        assert!(trimmed_tube_vec(&[1, 1], 100, 0.1, 0).is_err());
        assert!(trimmed_tube_vec(&[0, 2, 1], 100, 0.1, 0).is_err());
        assert!(trimmed_tube_vec(&ok, 1, 0.1, 0).is_err());
        assert!(trimmed_tube_vec(&ok, 100, 0.0, 0).is_err());
        assert!(trimmed_tube_vec(&ok, 100, 1.0, 0).is_err());
    }

    #[test]
    fn beta_marginal_of_extreme_negative_gap() {
        // With all positives ranked above all negatives, the cloud value at
        // t = 1/n0 is the positive-CDF evaluated in the first negative gap;
        // sanity-check the cloud's mean curve is monotone in t and the value
        // at t = 0 is exactly 0 (no atom, per the interpolation convention).
        let n0 = 10usize;
        let n1 = 10usize;
        let labels: Vec<u8> = std::iter::repeat_n(1u8, n1)
            .chain(std::iter::repeat_n(0u8, n0))
            .collect();
        let n_draws = 4000usize;
        let n_grid = n0 + 1;
        let cloud = fiducial_cloud(&labels, n0, n1, n_draws, 77);
        let mut mean = vec![0.0f64; n_grid];
        for m in 0..n_draws {
            for k in 0..n_grid {
                mean[k] += f64::from(cloud[m * n_grid + k]);
            }
        }
        for v in &mut mean {
            *v /= n_draws as f64;
        }
        assert_eq!(mean[0], 0.0);
        for w in mean.windows(2) {
            assert!(w[0] <= w[1]);
        }
        // Perfect separation with n0 = n1 = 10: the fiducial TPR at the
        // first negative should already be high (most positive mass sits
        // above the top negative gap). Loose but directional bound.
        assert!(
            mean[1] > 0.6,
            "mean cloud at t=1/n0 unexpectedly low: {}",
            mean[1]
        );
    }
}
