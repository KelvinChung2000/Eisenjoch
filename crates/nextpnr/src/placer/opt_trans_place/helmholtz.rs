//! Helmholtz operator helpers for the optimal transport placer.

/// Per-tile neighbor count for the Helmholtz Laplacian (Neumann BCs).
/// Constant for a given grid -- compute once, reuse across iterations.
/// Stored as f64 to avoid per-iteration u8->f64 casts in the diagonal update.
pub(super) fn helmholtz_neighbor_count(w: usize, h: usize) -> Vec<f64> {
    let mut counts = vec![0.0_f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut n = 0u32;
            if x > 0 { n += 1; }
            if x + 1 < w { n += 1; }
            if y > 0 { n += 1; }
            if y + 1 < h { n += 1; }
            counts[y * w + x] = n as f64;
        }
    }
    counts
}

/// Helmholtz operator off-diagonal: -1 for each grid edge (upper triangle).
/// Constant for a given grid -- compute once, reuse across iterations.
pub(super) fn helmholtz_off_diag(w: usize, h: usize) -> Vec<(usize, usize, f64)> {
    let mut off = Vec::with_capacity(2 * w * h);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if x + 1 < w { off.push((idx, idx + 1, -1.0)); }
            if y + 1 < h { off.push((idx, idx + w, -1.0)); }
        }
    }
    off
}
