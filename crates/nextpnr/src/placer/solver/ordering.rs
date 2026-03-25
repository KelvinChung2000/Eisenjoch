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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd_2x2_natural_order() {
        let perm = nested_dissection_2d(2, 2);
        assert_eq!(perm.len(), 4);
        // Small grid: natural order
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn nd_8x8_valid_permutation() {
        let perm = nested_dissection_2d(8, 8);
        assert_eq!(perm.len(), 64);
        // Must be a valid permutation: all indices 0..64 appear exactly once.
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..64).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn nd_81x81_valid_permutation() {
        let perm = nested_dissection_2d(81, 81);
        assert_eq!(perm.len(), 81 * 81);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..81 * 81).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn nd_separator_last() {
        // For an 8x8 grid bisected vertically, the separator column (col 4)
        // should appear after both halves in the permutation.
        let perm = nested_dissection_2d(9, 1);
        // 1D: 9 nodes, mid=4, left=[0,3], right=[5,8], separator=4
        assert_eq!(perm.len(), 9);
        // Separator (node 4) should be the last element.
        assert_eq!(*perm.last().unwrap(), 4);
    }

    #[test]
    fn expand_ports_consistency() {
        let tile_perm = vec![2, 0, 1]; // 3 tiles
        let port_perm = expand_to_ports(&tile_perm);
        assert_eq!(port_perm.len(), 12);
        // new_tile 0 -> old_tile 2: ports 8,9,10,11
        assert_eq!(&port_perm[0..4], &[8, 9, 10, 11]);
        // new_tile 1 -> old_tile 0: ports 0,1,2,3
        assert_eq!(&port_perm[4..8], &[0, 1, 2, 3]);
        // new_tile 2 -> old_tile 1: ports 4,5,6,7
        assert_eq!(&port_perm[8..12], &[4, 5, 6, 7]);
    }

    #[test]
    fn inverse_perm_roundtrip() {
        let perm = nested_dissection_2d(8, 8);
        let inv = inverse_perm(&perm);
        for (new_idx, &old_idx) in perm.iter().enumerate() {
            assert_eq!(inv[old_idx], new_idx);
        }
    }

    #[test]
    fn nd_rectangular_grid() {
        let perm = nested_dissection_2d(20, 5);
        assert_eq!(perm.len(), 100);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(sorted, expected);
    }
}
