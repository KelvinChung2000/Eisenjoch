//! Fill-reducing orderings for sparse matrix factorization.
//!
//! Implements nested dissection for 2D grid graphs, which gives optimal
//! O(n log n) fill-in for grid Laplacians. The ordering is computed on
//! the tile grid and expanded to the 4-port-per-tile system.

/// Compute nested dissection ordering for a W×H grid.
///
/// Returns `perm` where `perm[new_index] = old_index`.
/// Interior nodes of each recursive subdomain are ordered first,
/// separator nodes are ordered last (minimizing fill-in).
pub fn nested_dissection_2d(width: usize, height: usize) -> Vec<usize> {
    let n = width * height;
    let mut perm = Vec::with_capacity(n);
    nd_recurse(0, 0, width, height, width, &mut perm);
    debug_assert_eq!(perm.len(), n);
    perm
}

/// Recursive nested dissection on a subgrid [x0, x0+w) × [y0, y0+h)
/// within a grid of total width `stride`.
fn nd_recurse(
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    stride: usize,
    perm: &mut Vec<usize>,
) {
    // Base case: small subgrid — use natural order.
    if w <= 3 && h <= 3 {
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                perm.push(y * stride + x);
            }
        }
        return;
    }

    if w >= h {
        // Vertical separator: bisect along x-axis.
        let mid = w / 2;
        // Left half: [x0, x0+mid) × [y0, y0+h)
        nd_recurse(x0, y0, mid, h, stride, perm);
        // Right half: [x0+mid+1, x0+w) × [y0, y0+h)
        if mid + 1 < w {
            nd_recurse(x0 + mid + 1, y0, w - mid - 1, h, stride, perm);
        }
        // Separator: column x0+mid, rows [y0, y0+h)
        for y in y0..y0 + h {
            perm.push(y * stride + (x0 + mid));
        }
    } else {
        // Horizontal separator: bisect along y-axis.
        let mid = h / 2;
        // Top half: [x0, x0+w) × [y0, y0+mid)
        nd_recurse(x0, y0, w, mid, stride, perm);
        // Bottom half: [x0, x0+w) × [y0+mid+1, y0+h)
        if mid + 1 < h {
            nd_recurse(x0, y0 + mid + 1, w, h - mid - 1, stride, perm);
        }
        // Separator: row y0+mid, columns [x0, x0+w)
        for x in x0..x0 + w {
            perm.push((y0 + mid) * stride + x);
        }
    }
}

/// Expand a tile-level permutation to a 4-port-per-tile permutation.
///
/// Given `tile_perm[new_tile] = old_tile`, produces a permutation on
/// 4*n_tiles indices where ports within each tile stay contiguous:
/// `port_perm[4*new_tile + p] = 4*old_tile + p` for p in 0..4.
pub fn expand_to_ports(tile_perm: &[usize]) -> Vec<usize> {
    let n_tiles = tile_perm.len();
    let mut port_perm = vec![0usize; n_tiles * 4];
    for (new_tile, &old_tile) in tile_perm.iter().enumerate() {
        for p in 0..4 {
            port_perm[new_tile * 4 + p] = old_tile * 4 + p;
        }
    }
    port_perm
}

/// Compute the inverse permutation: `inv_perm[old_index] = new_index`.
pub fn inverse_perm(perm: &[usize]) -> Vec<usize> {
    let n = perm.len();
    let mut inv = vec![0usize; n];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        inv[old_idx] = new_idx;
    }
    inv
}
