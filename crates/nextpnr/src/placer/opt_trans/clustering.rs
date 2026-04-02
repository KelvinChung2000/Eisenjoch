//! Multi-level cell clustering for the Beckmann OT placer.
//!
//! Instead of coarsening the grid (which changes the 2D Green's function topology),
//! we coarsen the cells: group connected cells into clusters, solve placement on
//! the full-resolution grid with fewer demand sources, then decluster.
//!
//! This preserves the Green's function at all levels, making gradients
//! resolution-invariant.

use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::CellId;

/// A cluster of cells that acts as a single demand source.
#[derive(Debug, Clone)]
pub struct CellCluster {
    /// Indices of the original cells in this cluster (into idx_to_cell).
    pub members: Vec<usize>,
    /// Centroid x position (tile coordinates, relative to network origin).
    pub cx: f64,
    /// Centroid y position.
    pub cy: f64,
}

/// A hierarchy of clustering levels.
///
/// Level 0 = coarsest clusters, level last = individual cells (finest).
pub struct ClusterHierarchy {
    /// Clustering levels, from coarsest (index 0) to finest (last).
    pub levels: Vec<Vec<CellCluster>>,
    /// For each level, maps original cell index -> cluster index at that level.
    pub cell_to_cluster: Vec<Vec<usize>>,
}

/// Build a multi-level cell clustering hierarchy based on net connectivity.
///
/// Uses size-capped union-find: edges sorted by weight (shared nets), merge
/// greedily but refuse merges that would exceed max_cluster_size.
pub fn build_cluster_hierarchy(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
    target_clusters_coarse: usize,
) -> ClusterHierarchy {
    let n = idx_to_cell.len();

    // Build cell-to-cell edge weights from shared nets.
    // Skip high-fanout nets (>10 pins) to avoid creating giant clusters.
    let mut edge_weights: FxHashMap<(usize, usize), u32> = FxHashMap::default();

    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let Some(dp) = net.driver() else { continue };
        if net.num_users() == 0 {
            continue;
        }

        // Collect movable cell indices on this net.
        let mut net_cells: Vec<usize> = Vec::new();
        if let Some(&idx) = cell_to_idx.get(&dp.cell) {
            net_cells.push(idx);
        }
        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            if let Some(&idx) = cell_to_idx.get(&user.cell) {
                if !net_cells.contains(&idx) {
                    net_cells.push(idx);
                }
            }
        }

        // Skip high-fanout nets: they create overly broad connectivity.
        if net_cells.len() > 10 {
            continue;
        }

        // Add edges between all pairs on this net.
        for i in 0..net_cells.len() {
            for j in (i + 1)..net_cells.len() {
                let (a, b) = if net_cells[i] < net_cells[j] {
                    (net_cells[i], net_cells[j])
                } else {
                    (net_cells[j], net_cells[i])
                };
                *edge_weights.entry((a, b)).or_insert(0) += 1;
            }
        }
    }

    // Sort edges by weight (heaviest first) for greedy merging.
    let mut edges: Vec<((usize, usize), u32)> = edge_weights.into_iter().collect();
    edges.sort_by(|a, b| b.1.cmp(&a.1));

    // Determine target cluster counts for each level.
    // Geometric progression: n -> n/4 -> n/16 -> ... -> target.
    let mut level_targets: Vec<usize> = Vec::new();
    {
        let mut t = n;
        while t > target_clusters_coarse {
            t = (t / 4).max(target_clusters_coarse);
            level_targets.push(t);
        }
        if level_targets.is_empty() || *level_targets.last().unwrap() != target_clusters_coarse {
            level_targets.push(target_clusters_coarse);
        }
    }

    // Build levels from finest to coarsest.
    let mut levels: Vec<Vec<CellCluster>> = Vec::new();
    let mut cell_to_cluster_maps: Vec<Vec<usize>> = Vec::new();

    // Finest level: each cell is its own cluster.
    let finest: Vec<CellCluster> = (0..n)
        .map(|i| CellCluster {
            members: vec![i],
            cx: cell_x[i],
            cy: cell_y[i],
        })
        .collect();
    let finest_map: Vec<usize> = (0..n).collect();
    levels.push(finest);
    cell_to_cluster_maps.push(finest_map);

    // Build each coarser level.
    for &target in &level_targets {
        let prev_level = levels.last().unwrap();
        let prev_map = cell_to_cluster_maps.last().unwrap();
        let n_prev = prev_level.len();

        if n_prev <= target {
            levels.push(prev_level.clone());
            cell_to_cluster_maps.push(prev_map.clone());
            continue;
        }

        // Max cluster size: cap to ensure balanced clusters.
        // At target T clusters from N cells, ideal size = N/T.
        // Allow 2x headroom.
        let ideal_size = (n + target - 1) / target;
        let max_cluster_size = (ideal_size * 2).max(4);

        // Size-capped union-find.
        let mut parent: Vec<usize> = (0..n_prev).collect();
        let mut rank: Vec<u32> = vec![0; n_prev];
        let mut size: Vec<usize> = prev_level.iter().map(|c| c.members.len()).collect();

        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while parent[r] != r {
                r = parent[r];
            }
            let mut c = x;
            while parent[c] != r {
                let next = parent[c];
                parent[c] = r;
                c = next;
            }
            r
        }

        fn union_capped(
            parent: &mut [usize],
            rank: &mut [u32],
            size: &mut [usize],
            a: usize,
            b: usize,
            max_size: usize,
        ) -> bool {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb {
                return false;
            }
            // Refuse if merged cluster would be too large.
            if size[ra] + size[rb] > max_size {
                return false;
            }
            let new_size = size[ra] + size[rb];
            if rank[ra] < rank[rb] {
                parent[ra] = rb;
                size[rb] = new_size;
            } else if rank[ra] > rank[rb] {
                parent[rb] = ra;
                size[ra] = new_size;
            } else {
                parent[rb] = ra;
                size[ra] = new_size;
                rank[ra] += 1;
            }
            true
        }

        let mut n_clusters = n_prev;
        for &((a, b), _weight) in &edges {
            if n_clusters <= target {
                break;
            }
            let ca = prev_map[a];
            let cb = prev_map[b];
            if union_capped(&mut parent, &mut rank, &mut size, ca, cb, max_cluster_size) {
                n_clusters -= 1;
            }
        }

        // Build new clusters from union-find.
        let mut root_to_new_idx: FxHashMap<usize, usize> = FxHashMap::default();
        let mut new_clusters: Vec<CellCluster> = Vec::new();

        for ci in 0..n_prev {
            let root = find(&mut parent, ci);
            let new_idx = if let Some(&idx) = root_to_new_idx.get(&root) {
                idx
            } else {
                let idx = new_clusters.len();
                root_to_new_idx.insert(root, idx);
                new_clusters.push(CellCluster {
                    members: Vec::new(),
                    cx: 0.0,
                    cy: 0.0,
                });
                idx
            };

            for &member in &prev_level[ci].members {
                new_clusters[new_idx].members.push(member);
            }
        }

        // Compute centroids.
        for cluster in &mut new_clusters {
            let n_members = cluster.members.len() as f64;
            let sum_x: f64 = cluster.members.iter().map(|&i| cell_x[i]).sum();
            let sum_y: f64 = cluster.members.iter().map(|&i| cell_y[i]).sum();
            cluster.cx = sum_x / n_members;
            cluster.cy = sum_y / n_members;
        }

        let mut new_map = vec![0usize; n];
        for (cluster_idx, cluster) in new_clusters.iter().enumerate() {
            for &member in &cluster.members {
                new_map[member] = cluster_idx;
            }
        }

        levels.push(new_clusters);
        cell_to_cluster_maps.push(new_map);
    }

    // Reverse: levels[0] = coarsest, levels[last] = finest.
    levels.reverse();
    cell_to_cluster_maps.reverse();

    ClusterHierarchy {
        levels,
        cell_to_cluster: cell_to_cluster_maps,
    }
}

/// After solving at a coarse cluster level, initialize finer positions.
///
/// For each cell, compute its coarse cluster centroid from the CURRENT positions
/// (post-solve), then add perturbation to break symmetry. This preserves the
/// global structure found at the coarse level.
pub fn decluster_positions(
    cell_to_coarse: &[usize],
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    n_coarse_clusters: usize,
    perturbation: f64,
    seed: u64,
) {
    let n = cell_x.len();

    // First compute cluster centroids from current positions.
    let mut cx = vec![0.0f64; n_coarse_clusters];
    let mut cy = vec![0.0f64; n_coarse_clusters];
    let mut counts = vec![0usize; n_coarse_clusters];
    for i in 0..n {
        let ci = cell_to_coarse[i];
        cx[ci] += cell_x[i];
        cy[ci] += cell_y[i];
        counts[ci] += 1;
    }
    for ci in 0..n_coarse_clusters {
        if counts[ci] > 0 {
            cx[ci] /= counts[ci] as f64;
            cy[ci] /= counts[ci] as f64;
        }
    }

    // Place each cell at its cluster centroid + perturbation.
    for cell_idx in 0..n {
        let coarse_idx = cell_to_coarse[cell_idx];
        let h = hash_u64(seed ^ (cell_idx as u64) ^ ((coarse_idx as u64) << 32));
        let dx = ((h & 0xFFFF) as f64 / 65536.0 - 0.5) * 2.0 * perturbation;
        let dy = (((h >> 16) & 0xFFFF) as f64 / 65536.0 - 0.5) * 2.0 * perturbation;
        cell_x[cell_idx] = cx[coarse_idx] + dx;
        cell_y[cell_idx] = cy[coarse_idx] + dy;
    }
}

/// Update cluster centroids from current cell positions.
pub fn update_centroids(clusters: &mut [CellCluster], cell_x: &[f64], cell_y: &[f64]) {
    for cluster in clusters.iter_mut() {
        if cluster.members.is_empty() {
            continue;
        }
        let n = cluster.members.len() as f64;
        let sum_x: f64 = cluster.members.iter().map(|&i| cell_x[i]).sum();
        let sum_y: f64 = cluster.members.iter().map(|&i| cell_y[i]).sum();
        cluster.cx = sum_x / n;
        cluster.cy = sum_y / n;
    }
}

/// Apply gradient to clustered cells: all members of a cluster move together.
///
/// Accumulates per-cell gradient into per-cluster gradient, then applies
/// the cluster gradient to all members.
pub fn apply_clustered_gradient(
    clusters: &[CellCluster],
    cell_to_cluster: &[usize],
    grad: &mut [f64],
    n: usize,
) {
    let n_clusters = clusters.len();

    let mut cluster_gx = vec![0.0; n_clusters];
    let mut cluster_gy = vec![0.0; n_clusters];
    let mut cluster_count = vec![0usize; n_clusters];

    let (gx, gy) = grad.split_at_mut(n);
    for i in 0..n {
        let ci = cell_to_cluster[i];
        cluster_gx[ci] += gx[i];
        cluster_gy[ci] += gy[i];
        cluster_count[ci] += 1;
    }

    // Average gradient per cluster, assign to all members.
    for i in 0..n {
        let ci = cell_to_cluster[i];
        let count = cluster_count[ci] as f64;
        gx[i] = cluster_gx[ci] / count;
        gy[i] = cluster_gy[ci] / count;
    }
}

/// Print clustering statistics.
pub fn print_cluster_stats(hierarchy: &ClusterHierarchy) {
    for (level_idx, clusters) in hierarchy.levels.iter().enumerate() {
        let n_clusters = clusters.len();
        let total_cells: usize = clusters.iter().map(|c| c.members.len()).sum();
        let max_size = clusters.iter().map(|c| c.members.len()).max().unwrap_or(0);
        let min_size = clusters.iter().map(|c| c.members.len()).min().unwrap_or(0);
        let avg_size = if n_clusters > 0 {
            total_cells as f64 / n_clusters as f64
        } else {
            0.0
        };
        eprintln!(
            "  Cluster level {}: {} clusters, avg={:.1} min={} max={} cells (total={})",
            level_idx, n_clusters, avg_size, min_size, max_size, total_cells,
        );
    }
}

/// Simple deterministic hash for perturbation.
#[inline]
fn hash_u64(mut x: u64) -> u64 {
    x = x.wrapping_mul(0x517cc1b727220a95);
    x ^= x >> 32;
    x = x.wrapping_mul(0x6c62272e07bb0142);
    x ^= x >> 32;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(hash_u64(42), hash_u64(42));
        assert_ne!(hash_u64(42), hash_u64(43));
    }

    #[test]
    fn update_centroids_computes_mean() {
        let mut clusters = vec![CellCluster {
            members: vec![0, 1, 2],
            cx: 0.0,
            cy: 0.0,
        }];
        let cell_x = vec![1.0, 3.0, 5.0];
        let cell_y = vec![2.0, 4.0, 6.0];
        update_centroids(&mut clusters, &cell_x, &cell_y);
        assert!((clusters[0].cx - 3.0).abs() < 1e-12);
        assert!((clusters[0].cy - 4.0).abs() < 1e-12);
    }

    #[test]
    fn apply_clustered_gradient_averages() {
        let clusters = vec![
            CellCluster {
                members: vec![0, 1],
                cx: 0.0,
                cy: 0.0,
            },
            CellCluster {
                members: vec![2],
                cx: 0.0,
                cy: 0.0,
            },
        ];
        let cell_to_cluster = vec![0, 0, 1];
        let n = 3;
        let mut grad = vec![2.0, 4.0, 10.0, 6.0, 8.0, 20.0];
        apply_clustered_gradient(&clusters, &cell_to_cluster, &mut grad, n);
        assert!((grad[0] - 3.0).abs() < 1e-12);
        assert!((grad[1] - 3.0).abs() < 1e-12);
        assert!((grad[2] - 10.0).abs() < 1e-12);
        assert!((grad[3] - 7.0).abs() < 1e-12);
        assert!((grad[4] - 7.0).abs() < 1e-12);
        assert!((grad[5] - 20.0).abs() < 1e-12);
    }
}
