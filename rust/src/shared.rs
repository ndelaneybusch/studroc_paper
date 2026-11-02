//! A fiducial cloud generated once and trimmed at several levels.
//!
//! Draw `m` depends only on `(seed, m)`, so the leading `n_draws` rows of a
//! cloud are exactly the cloud [`fiducial_cloud`] would draw with `n_draws`
//! rows. Every [`SharedCloud`] method therefore takes the number of leading
//! rows to use, and its output is bit-identical to
//! [`trimmed_tube_vec`](crate::trimmed_tube_vec) at that draw count and seed.
//! One replicate can then share a single generation across levels whose draw
//! budgets differ, and across trims of the same rows over different column
//! subsets.
//!
//! Two passes are restructured relative to the one-shot kernel, both without
//! changing any output bit:
//!
//! * per-column depth contributions are cached per draw count, so a trim over
//!   a subset of already-ranked columns (the recalibrated hybrid trim) is a
//!   min-reduction instead of a re-sort;
//! * the pointwise order statistics for every requested depth come from one
//!   row-major streaming pass that keeps, per column, bounded heaps of the
//!   smallest and largest values seen. Depths are small next to the draw
//!   count, so almost every value is rejected by one comparison with a heap
//!   root, and the cloud is never transposed.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use rayon::prelude::*;

use crate::{
    column_depths_into, fiducial_cloud, gather_block_cols, parse_labels, trim_depth_from,
    validate_trim_cols, CHUNK_COLS,
};

/// Grid columns per task in the streaming order-statistic pass.
const STREAM_COLS: usize = 512;

/// Lower and upper tube edges.
pub type Tube = (Vec<f64>, Vec<f64>);

/// Per-draw depth contribution of one column, keyed by (rows, column).
type DepthCache = HashMap<(usize, usize), Vec<u32>>;

/// A generated cloud plus the per-column depth ranks computed from it.
pub struct SharedCloud {
    cloud: Vec<f32>,
    n_draws: usize,
    n_grid: usize,
    column_depths: Mutex<DepthCache>,
}

impl SharedCloud {
    /// Generate `n_draws` rows for the merged label sequence.
    ///
    /// # Errors
    ///
    /// Returns a message when the labels are invalid or `n_draws < 2`.
    pub fn new(labels: &[u8], n_draws: usize, seed: u64) -> Result<Self, String> {
        let (n0, n1) = parse_labels(labels)?;
        if n_draws < 2 {
            return Err(format!("n_draws must be at least 2, got {n_draws}"));
        }
        Ok(Self {
            cloud: fiducial_cloud(labels, n0, n1, n_draws, seed),
            n_draws,
            n_grid: n0 + 1,
            column_depths: Mutex::new(HashMap::new()),
        })
    }

    /// Number of generated rows (the largest usable prefix).
    #[must_use]
    pub fn n_draws(&self) -> usize {
        self.n_draws
    }

    fn prefix(&self, n_draws: usize) -> Result<&[f32], String> {
        if !(2..=self.n_draws).contains(&n_draws) {
            return Err(format!(
                "n_draws must be in [2, {}], got {n_draws}",
                self.n_draws
            ));
        }
        Ok(&self.cloud[..n_draws * self.n_grid])
    }

    /// Trim depth over the leading `n_draws` rows and `trim_cols` (all
    /// columns when `None`): the `alpha_eff`-quantile of the min-p depths,
    /// clamped to `[1, n_draws / 2]`.
    ///
    /// # Errors
    ///
    /// Returns a message when `n_draws` exceeds the generated rows,
    /// `alpha_eff` is outside `(0, 1)`, or `trim_cols` is invalid.
    pub fn trim_depth(
        &self,
        n_draws: usize,
        alpha_eff: f64,
        trim_cols: Option<&[usize]>,
    ) -> Result<usize, String> {
        let cloud = self.prefix(n_draws)?;
        if !(alpha_eff > 0.0 && alpha_eff < 1.0) {
            return Err(format!("alpha_eff must be in (0, 1), got {alpha_eff}"));
        }
        validate_trim_cols(trim_cols, self.n_grid)?;
        let cols: Vec<usize> =
            trim_cols.map_or_else(|| (0..self.n_grid).collect(), <[usize]>::to_vec);

        let missing: Vec<usize> = {
            let cache = self.lock_depths()?;
            cols.iter()
                .copied()
                .filter(|&k| !cache.contains_key(&(n_draws, k)))
                .collect()
        };
        let computed: Vec<(usize, Vec<u32>)> = missing
            .par_chunks(CHUNK_COLS)
            .map(|chunk| {
                let mut block = vec![0.0f32; chunk.len() * n_draws];
                gather_block_cols(cloud, n_draws, self.n_grid, chunk, &mut block);
                let mut pairs = Vec::with_capacity(n_draws);
                chunk
                    .iter()
                    .zip(block.chunks_exact(n_draws))
                    .map(|(&k, col)| {
                        let mut depths = vec![n_draws as u32; n_draws];
                        column_depths_into(col, &mut pairs, &mut depths);
                        (k, depths)
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        let mut cache = self.lock_depths()?;
        cache.extend(computed.into_iter().map(|(k, d)| ((n_draws, k), d)));
        let mut depths = vec![n_draws as u32; n_draws];
        for k in cols {
            for (slot, &d) in depths.iter_mut().zip(&cache[&(n_draws, k)]) {
                if d < *slot {
                    *slot = d;
                }
            }
        }
        Ok(trim_depth_from(depths, alpha_eff))
    }

    /// Pointwise `j`-th smallest and largest values of the leading
    /// `n_draws` rows, for each `j` in `js`.
    ///
    /// # Errors
    ///
    /// Returns a message when `n_draws` exceeds the generated rows, `js` is
    /// empty, or any `j` is outside `[1, n_draws / 2]`.
    pub fn tubes(&self, n_draws: usize, js: &[usize]) -> Result<Vec<Tube>, String> {
        let cloud = self.prefix(n_draws)?;
        let j_max = (n_draws / 2).max(1);
        if js.is_empty() {
            return Err("js must be non-empty".to_string());
        }
        if let Some(&j) = js.iter().find(|&&j| !(1..=j_max).contains(&j)) {
            return Err(format!("j must be in [1, {j_max}], got {j}"));
        }
        Ok(order_stats_multi(cloud, n_draws, self.n_grid, js))
    }

    fn lock_depths(&self) -> Result<MutexGuard<'_, DepthCache>, String> {
        self.column_depths
            .lock()
            .map_err(|_| "column depth cache poisoned".to_string())
    }
}

/// Restore the heap property below `i`; `above(a, b)` orders parents over
/// children (max-heap for "greater", min-heap for "less").
#[inline]
fn sift_down(heap: &mut [f32], mut i: usize, above: impl Fn(f32, f32) -> bool) {
    let n = heap.len();
    loop {
        let left = 2 * i + 1;
        if left >= n {
            return;
        }
        let right = left + 1;
        let child = if right < n && above(heap[right], heap[left]) {
            right
        } else {
            left
        };
        if !above(heap[child], heap[i]) {
            return;
        }
        heap.swap(i, child);
        i = child;
    }
}

fn greater(a: f32, b: f32) -> bool {
    a.total_cmp(&b).is_gt()
}

fn less(a: f32, b: f32) -> bool {
    a.total_cmp(&b).is_lt()
}

/// Per-column `j`-th smallest and `j`-th largest values (1-indexed) of a
/// row-major cloud for each `j` in `js`, in one pass over its rows.
///
/// Identical, value for value, to [`pointwise_order_stats`] at each `j`:
/// the bounded heaps retain the `max(js)` smallest and largest values of
/// each column under the same total order, and values equal under that
/// order share their bits.
///
/// [`pointwise_order_stats`]: crate::pointwise_order_stats
#[must_use]
pub fn order_stats_multi(cloud: &[f32], n_draws: usize, n_grid: usize, js: &[usize]) -> Vec<Tube> {
    let depth = *js.iter().max().expect("js is non-empty");
    let starts: Vec<usize> = (0..n_grid).step_by(STREAM_COLS).collect();
    let per_chunk: Vec<(usize, Vec<f32>, Vec<f32>)> = starts
        .par_iter()
        .map(|&lo| {
            let width = STREAM_COLS.min(n_grid - lo);
            let mut lows = vec![0.0f32; width * depth];
            let mut highs = vec![0.0f32; width * depth];
            for m in 0..depth {
                let row = &cloud[m * n_grid + lo..m * n_grid + lo + width];
                for (c, &v) in row.iter().enumerate() {
                    lows[c * depth + m] = v;
                    highs[c * depth + m] = v;
                }
            }
            for c in 0..width {
                let low = &mut lows[c * depth..(c + 1) * depth];
                let high = &mut highs[c * depth..(c + 1) * depth];
                for i in (0..depth / 2).rev() {
                    sift_down(low, i, greater);
                    sift_down(high, i, less);
                }
            }
            for m in depth..n_draws {
                let row = &cloud[m * n_grid + lo..m * n_grid + lo + width];
                for (c, &v) in row.iter().enumerate() {
                    let low = &mut lows[c * depth..(c + 1) * depth];
                    if less(v, low[0]) {
                        low[0] = v;
                        sift_down(low, 0, greater);
                    }
                    let high = &mut highs[c * depth..(c + 1) * depth];
                    if greater(v, high[0]) {
                        high[0] = v;
                        sift_down(high, 0, less);
                    }
                }
            }
            for c in 0..width {
                lows[c * depth..(c + 1) * depth].sort_unstable_by(f32::total_cmp);
                highs[c * depth..(c + 1) * depth].sort_unstable_by(|a, b| b.total_cmp(a));
            }
            (lo, lows, highs)
        })
        .collect();

    js.iter()
        .map(|&j| {
            let mut lower = vec![0.0f64; n_grid];
            let mut upper = vec![0.0f64; n_grid];
            for (lo, lows, highs) in &per_chunk {
                for (c, (low, high)) in lows
                    .chunks_exact(depth)
                    .zip(highs.chunks_exact(depth))
                    .enumerate()
                {
                    lower[lo + c] = f64::from(low[j - 1]);
                    upper[lo + c] = f64::from(high[j - 1]);
                }
            }
            (lower, upper)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pointwise_order_stats, trimmed_tube_vec};

    /// Deterministic label sequence with runs of both classes.
    fn labels(n0: usize, n1: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let (mut r0, mut r1) = (n0, n1);
        let mut out = Vec::with_capacity(n0 + n1);
        while r0 + r1 > 0 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            let p1 = r1 as f64 / (r0 + r1) as f64;
            if r0 == 0 || (r1 > 0 && u < p1) {
                out.push(1);
                r1 -= 1;
            } else {
                out.push(0);
                r0 -= 1;
            }
        }
        out
    }

    #[test]
    fn prefixes_reproduce_independent_tubes_in_any_trim_order() {
        // Each (draw count, level, column subset) must match a fresh cloud
        // bit for bit, whether the subset's columns were ranked before (cache
        // hit) or not (cache miss).
        let labs = labels(70, 50, 6);
        let thin: Vec<usize> = (0..=70).step_by(3).collect();
        let cases = [(900usize, 0.05), (400, 0.2), (250, 0.5), (900, 0.5)];
        for subset_first in [false, true] {
            let shared = SharedCloud::new(&labs, 900, 19).expect("valid");
            let order: [Option<&[usize]>; 2] = if subset_first {
                [Some(&thin), None]
            } else {
                [None, Some(&thin)]
            };
            for (n_draws, alpha_eff) in cases {
                for cols in order {
                    let expect =
                        trimmed_tube_vec(&labs, n_draws, alpha_eff, 19, cols).expect("valid");
                    let j = shared.trim_depth(n_draws, alpha_eff, cols).expect("valid");
                    let (lower, upper) = shared.tubes(n_draws, &[j]).expect("valid").remove(0);
                    assert_eq!((lower, upper, j), expect, "n_draws={n_draws} alpha={alpha_eff}");
                }
            }
        }
    }

    #[test]
    fn streaming_order_stats_match_selection_at_every_depth() {
        // Includes j = 1, j = n_draws / 2, and the all-equal corner columns.
        for &(n0, n1, n_draws) in &[(17usize, 8usize, 128usize), (3, 40, 64), (600, 30, 50)] {
            let labs = labels(n0, n1, 21);
            let n_grid = n0 + 1;
            let cloud = fiducial_cloud(&labs, n0, n1, n_draws, 9);
            let js = [1usize, 2, 4, n_draws / 3, n_draws / 2];
            let tubes = order_stats_multi(&cloud, n_draws, n_grid, &js);
            for (&j, tube) in js.iter().zip(&tubes) {
                let expect = pointwise_order_stats(&cloud, n_draws, n_grid, j);
                assert_eq!(*tube, expect, "n0={n0} j={j}");
            }
        }
    }

    #[test]
    fn rejects_out_of_range_requests() {
        let shared = SharedCloud::new(&labels(5, 5, 1), 100, 0).expect("valid");
        assert!(shared.trim_depth(101, 0.1, None).is_err());
        assert!(shared.trim_depth(1, 0.1, None).is_err());
        assert!(shared.trim_depth(100, 0.0, None).is_err());
        assert!(shared.trim_depth(100, 0.1, Some(&[0, 6])).is_err());
        assert!(shared.tubes(100, &[0]).is_err());
        assert!(shared.tubes(100, &[3, 51]).is_err());
        assert!(shared.tubes(100, &[]).is_err());
        assert!(shared.tubes(101, &[3]).is_err());
        assert!(SharedCloud::new(&[0, 0], 100, 0).is_err());
    }
}
