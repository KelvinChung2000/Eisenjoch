//! Shift-invariant displacement table.
//!
//! Built once per outer iteration. Drives the `DisplacementPure / Sparse /
//! Warmstart` path-solver variants.
//!
//! Idea: in sv3-like designs the global pipe graph is near-uniform (empirical
//! `p50=p90=p99 = 1.0000` for `R_eff / base`), so the shortest-path distance
//! from a CLB tile to any tile within a radius `R` depends mostly on the
//! relative offset `(dx, dy)` and not on the absolute source position. One
//! Dijkstra from a centrally-located CLB node fills a `(2R+1) × (2R+1)`
//! distance grid that all uniform (src, sink) pairs can look up in O(1).

use crate::placer::opt_trans::network::{PipeNetwork, DIST_SCALE};
use crate::placer::opt_trans::path_solver;
use rustc_hash::FxHashMap;

/// Half-width of the displacement table in tiles. Covers the vast majority of
/// sv3 nets (most HPWL bboxes are < 30 tiles wide).
pub const DISP_RADIUS: i32 = 20;

pub struct DisplacementTable {
    pub radius: i32,
    pub stride: i32,
    /// `dist[(dy + R) * stride + (dx + R)]` = quantized distance (same units
    /// as `pipe_costs_int`). `u32::MAX` means no path within the radius from
    /// the center to that offset.
    pub dist: Vec<u32>,
    pub center_node: usize,
    pub center_tile_x: i32,
    pub center_tile_y: i32,
    /// Dominant tile type in the fabric; used as the "uniform" class.
    pub uniform_tile_type: u16,
}

impl DisplacementTable {
    pub fn build(network: &PipeNetwork) -> Option<Self> {
        let uniform_tile_type = dominant_tile_type(network)?;
        let center_node = find_center_uniform_node(network, uniform_tile_type)?;

        let n_nodes = network.nodes.len();
        let mut dist_nodes = vec![f64::INFINITY; n_nodes];
        // Run on *real* pipe costs (which are base resistance in the near-
        // quiescent regime sv3 spends most iterations in). Any hot pipe near
        // the center will bias the table, but on sv3 that bias is measurement
        // noise.
        path_solver::dijkstra_single_source(
            network,
            center_node,
            |pipe| pipe.base_resistance,
            &mut dist_nodes,
        );

        let stride = 2 * DISP_RADIUS + 1;
        let tbl_size = (stride * stride) as usize;
        let mut dist = vec![u32::MAX; tbl_size];

        let center_x = network.nodes[center_node].tile_x;
        let center_y = network.nodes[center_node].tile_y;

        for (node_idx, &d) in dist_nodes.iter().enumerate() {
            if !d.is_finite() {
                continue;
            }
            let node = &network.nodes[node_idx];
            let dx = node.tile_x - center_x;
            let dy = node.tile_y - center_y;
            if dx.abs() > DISP_RADIUS || dy.abs() > DISP_RADIUS {
                continue;
            }
            let idx = ((dy + DISP_RADIUS) * stride + (dx + DISP_RADIUS)) as usize;
            let q = (d * DIST_SCALE).round();
            let q32 = if q >= (u32::MAX as f64) {
                u32::MAX - 1
            } else {
                (q as u32).max(1)
            };
            if q32 < dist[idx] {
                dist[idx] = q32;
            }
        }
        // Distance from center to itself is zero.
        let self_idx = (DISP_RADIUS * stride + DISP_RADIUS) as usize;
        dist[self_idx] = 0;

        Some(DisplacementTable {
            radius: DISP_RADIUS,
            stride,
            dist,
            center_node,
            center_tile_x: center_x,
            center_tile_y: center_y,
            uniform_tile_type,
        })
    }

    #[inline]
    pub fn lookup(&self, dx: i32, dy: i32) -> Option<u32> {
        if dx.abs() > self.radius || dy.abs() > self.radius {
            return None;
        }
        let idx = ((dy + self.radius) * self.stride + (dx + self.radius)) as usize;
        let d = self.dist[idx];
        if d == u32::MAX {
            None
        } else {
            Some(d)
        }
    }

    /// Returns `true` iff the source + every non-zero-demand sink sits on a
    /// node of the dominant tile type AND every sink is within the table's
    /// radius of the source.
    pub fn is_uniform_pair(
        &self,
        network: &PipeNetwork,
        src_node: usize,
        sinks: &[(usize, f64)],
    ) -> bool {
        if src_node >= network.tile_type_by_node.len() {
            return false;
        }
        if network.tile_type_by_node[src_node] != self.uniform_tile_type {
            return false;
        }
        let src = &network.nodes[src_node];
        for &(node, demand) in sinks {
            if demand == 0.0 {
                continue;
            }
            if node >= network.tile_type_by_node.len() {
                return false;
            }
            if network.tile_type_by_node[node] != self.uniform_tile_type {
                return false;
            }
            let sink = &network.nodes[node];
            let dx = sink.tile_x - src.tile_x;
            let dy = sink.tile_y - src.tile_y;
            if dx.abs() > self.radius || dy.abs() > self.radius {
                return false;
            }
            let idx = ((dy + self.radius) * self.stride + (dx + self.radius)) as usize;
            if self.dist[idx] == u32::MAX {
                return false;
            }
        }
        true
    }
}

/// Most-common `tile_type_by_node` entry. Robust enough for xc7 where CLB
/// interior dominates the node count.
fn dominant_tile_type(network: &PipeNetwork) -> Option<u16> {
    let mut counts: FxHashMap<u16, usize> = FxHashMap::default();
    for &tt in &network.tile_type_by_node {
        *counts.entry(tt).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|&(_, c)| c).map(|(tt, _)| tt)
}

/// Node of the dominant tile type nearest to the grid center. Ties broken by
/// lower node index.
fn find_center_uniform_node(network: &PipeNetwork, tt: u16) -> Option<usize> {
    let cx = network.width / 2;
    let cy = network.height / 2;
    let mut best: Option<(usize, i32)> = None;
    for (i, node) in network.nodes.iter().enumerate() {
        if network.tile_type_by_node.get(i).copied() != Some(tt) {
            continue;
        }
        let d = (node.tile_x - cx).abs() + (node.tile_y - cy).abs();
        match best {
            Some((_, bd)) if bd <= d => {}
            _ => best = Some((i, d)),
        }
    }
    best.map(|(i, _)| i)
}
