//! Discrete Fast Sweeping Method (FSM) for shortest-path labels on the pipe
//! graph. Drop-in replacement for the bucket-Dial Dijkstra; produces identical
//! `dist_int` labels (same min-plus semiring).
//!
//! When a corridor is active the sweep iterates the sparse `corridor_marked`
//! list directly in the four raster orderings, avoiding any bbox-wide scan.
//! Long-range pipes are relaxed as a sparse step between sweep rounds and may
//! trigger another round. A post-sweep forward pass over bucket-sorted
//! `settle_order` computes `path_weight[v]` exactly as the fused Dijkstra.
//!
//! Full-graph (no-corridor) paths fall back to a raster sweep of the whole
//! grid; that path is only used when `build_corridor` declines.
//!
//! See `path_solver::dijkstra_and_forward` for the original fused reference.

use super::network::{PipeNetwork, DIST_SCALE};
use super::path_solver::{link_likelihood_from_ints, PathSolverWorkspace};

/// Hard cap on sweep rounds to avoid pathological loops. In practice FSM
/// converges in 1-3 rounds on stereovision3; MAX=8 is a loud safety net.
const MAX_SWEEP_ROUNDS: usize = 8;

/// Discrete FSM shortest-path + Dial forward-pass solve, matching the output
/// of `dijkstra_and_forward`.
pub(crate) fn fsm_dist_and_forward(
    network: &PipeNetwork,
    source_node: usize,
    ws: &mut PathSolverWorkspace,
    use_corridor: bool,
) -> usize {
    let w = network.width;
    let h = network.height;
    let inv_scale = 1.0 / DIST_SCALE;

    // Seed source.
    ws.dist_int[source_node] = 0;

    let mut relax_count = 0usize;
    let mut max_d: u32 = 0;

    for _round in 0..MAX_SWEEP_ROUNDS {
        let mut changed = false;

        for dir in 0..4u8 {
            let x_fwd = dir & 1 == 0;
            let y_fwd = dir & 2 == 0;
            let (count, max_after) = if use_corridor {
                sparse_sweep(network, ws, x_fwd, y_fwd, max_d)
            } else {
                full_sweep(network, ws, x_fwd, y_fwd, max_d)
            };
            relax_count += count;
            max_d = max_after;
            if count > 0 {
                changed = true;
            }
        }

        // Long-range pipe relaxation (bidirectional).
        let (count, max_after) = long_pipe_relax(network, ws, use_corridor, max_d);
        relax_count += count;
        max_d = max_after;
        if count > 0 {
            changed = true;
        }

        if !changed {
            break;
        }
    }

    // Collect settled nodes into `settle_order`, ascending by dist_int.
    bucket_collect_settle_order(network, ws, max_d, use_corridor);

    // Forward pass in ascending-dist order: replicate `dijkstra_and_forward`'s
    // accumulation of `path_weight` from every upstream-settled neighbour.
    ws.path_weight[source_node] = 1.0;
    ws.dist[source_node] = 0.0;
    let adj = &network.flat_adjacency;
    let adj_neighbors = adj.neighbors.as_slice();
    let adj_pipe_idx = adj.pipe_idx.as_slice();
    let adj_offsets = adj.offsets.as_slice();
    for &node in ws.settle_order.iter() {
        let cur_int = ws.dist_int[node];
        ws.dist[node] = (cur_int as f64) * inv_scale;
        if node == source_node {
            continue;
        }
        let mut acc = 0.0f64;
        let edge_start = adj_offsets[node] as usize;
        let edge_end = adj_offsets[node + 1] as usize;
        for e in edge_start..edge_end {
            let neighbour = adj_neighbors[e] as usize;
            if use_corridor && !ws.in_corridor[neighbour] {
                continue;
            }
            let up_int = ws.dist_int[neighbour];
            if up_int >= cur_int {
                continue;
            }
            let up_weight = ws.path_weight[neighbour];
            if up_weight <= 0.0 || !up_weight.is_finite() {
                continue;
            }
            let pipe_idx = adj_pipe_idx[e] as usize;
            let cost_int = network.pipe_cost_int(pipe_idx);
            let likelihood = link_likelihood_from_ints(up_int, cur_int, cost_int);
            if likelihood > 0.0 && likelihood.is_finite() {
                acc += up_weight * likelihood;
            }
        }
        ws.path_weight[node] = acc;
    }

    ws.max_bucket = max_d;
    let _ = (w, h);
    relax_count
}

/// One FSM sweep restricted to the sparse `corridor_marked` list.
///
/// `corridor_marked` is populated by `mark_corridor` in (y asc, x asc) order.
/// The four sweep directions each need a particular visit ordering:
/// - SW→NE (x_fwd, y_fwd): iterate as-is.
/// - NE→SW (!x_fwd, !y_fwd): iterate reversed.
/// - SE→NW (!x_fwd, y_fwd): per row, iterate reversed within row.
/// - NW→SE (x_fwd, !y_fwd): iterate rows reversed, in-row asc.
///
/// Each tile relaxes from its two sweep-predecessor cardinal neighbours; the
/// other two are handled by the complementary sweep directions.
fn sparse_sweep(
    network: &PipeNetwork,
    ws: &mut PathSolverWorkspace,
    x_fwd: bool,
    y_fwd: bool,
    mut max_d: u32,
) -> (usize, u32) {
    let w = network.width;
    let marked = &ws.corridor_marked;
    let w_us = w as usize;

    // Build row boundaries: marked is already (y asc, x asc). Find the
    // (start, end) index for each contiguous y block.
    //
    // We iterate marked without materialising the row list, by tracking
    // current y and scanning ahead. For performance, do the scanning via
    // iterator windows.
    let mut count = 0usize;

    // Convert direction flags to a visit-order iterator that yields usize
    // node indices. Four combinations:
    //
    //   SW→NE (x_fwd, y_fwd): forward iteration (y asc, x asc).
    //   NE→SW: reverse iteration (y desc, x desc).
    //   SE→NW (!x_fwd, y_fwd): rows asc, within-row reversed.
    //   NW→SE (x_fwd, !y_fwd): rows desc, within-row asc.
    //
    // For same-direction (both fwd or both rev), a single iter or iter().rev()
    // over the whole slice works. For mixed, walk per-row.
    match (x_fwd, y_fwd) {
        (true, true) => {
            let nodes: Vec<usize> = marked.clone();
            for node in nodes {
                let (did_relax, nd) = relax_tile(network, ws, node, x_fwd, y_fwd, w_us);
                if did_relax {
                    count += 1;
                    if nd > max_d {
                        max_d = nd;
                    }
                }
            }
        }
        (false, false) => {
            let nodes: Vec<usize> = marked.iter().rev().copied().collect();
            for node in nodes {
                let (did_relax, nd) = relax_tile(network, ws, node, x_fwd, y_fwd, w_us);
                if did_relax {
                    count += 1;
                    if nd > max_d {
                        max_d = nd;
                    }
                }
            }
        }
        (false, true) => {
            let nodes = per_row_reversed(marked, w_us);
            for node in nodes {
                let (did_relax, nd) = relax_tile(network, ws, node, x_fwd, y_fwd, w_us);
                if did_relax {
                    count += 1;
                    if nd > max_d {
                        max_d = nd;
                    }
                }
            }
        }
        (true, false) => {
            let nodes = rows_reversed(marked, w_us);
            for node in nodes {
                let (did_relax, nd) = relax_tile(network, ws, node, x_fwd, y_fwd, w_us);
                if did_relax {
                    count += 1;
                    if nd > max_d {
                        max_d = nd;
                    }
                }
            }
        }
    }

    (count, max_d)
}

/// Build a visit order: rows ascending, within each row the x coordinates are
/// traversed in descending order (SE→NW sweep). Works from the (y asc, x asc)
/// invariant of `corridor_marked`.
fn per_row_reversed(marked: &[usize], w_us: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(marked.len());
    let mut i = 0;
    while i < marked.len() {
        let y = marked[i] / w_us;
        let start = i;
        while i < marked.len() && marked[i] / w_us == y {
            i += 1;
        }
        for &node in marked[start..i].iter().rev() {
            out.push(node);
        }
    }
    out
}

/// Build a visit order: rows descending, within each row x ascending
/// (NW→SE sweep).
fn rows_reversed(marked: &[usize], w_us: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(marked.len());
    // Find row boundaries in a first pass, then emit in reverse row order.
    let mut row_starts: Vec<usize> = Vec::with_capacity(32);
    let mut i = 0;
    while i < marked.len() {
        row_starts.push(i);
        let y = marked[i] / w_us;
        while i < marked.len() && marked[i] / w_us == y {
            i += 1;
        }
    }
    row_starts.push(marked.len());
    for win in row_starts.windows(2).rev() {
        let (start, end) = (win[0], win[1]);
        for &node in &marked[start..end] {
            out.push(node);
        }
    }
    out
}

/// Relax a single tile from its two sweep-predecessor cardinal neighbours.
/// Returns `(did_relax, new_dist)` where `new_dist` is the post-update label
/// (only meaningful when `did_relax` is true).
#[inline]
fn relax_tile(
    network: &PipeNetwork,
    ws: &mut PathSolverWorkspace,
    node: usize,
    x_fwd: bool,
    y_fwd: bool,
    w_us: usize,
) -> (bool, u32) {
    let y = node / w_us;
    let x = node - y * w_us;
    let tg = &network.tile_grid;
    let pipe_e: &[u32] = &tg.pipe_e;
    let pipe_s: &[u32] = &tg.pipe_s;

    let cur = ws.dist_int[node];
    let mut new_d = cur;

    // West predecessor: pipe_e[y, x-1] carries (x-1, y) → (x, y).
    if x_fwd && x > 0 {
        let slot = node - 1;
        let pipe = pipe_e[slot];
        if pipe != u32::MAX {
            let up = ws.dist_int[slot];
            if up != u32::MAX {
                let cand = up.saturating_add(network.pipe_cost_int(pipe as usize));
                if cand < new_d {
                    new_d = cand;
                }
            }
        }
    }
    // East predecessor.
    if !x_fwd && x + 1 < w_us {
        let slot = node;
        let pipe = pipe_e[slot];
        if pipe != u32::MAX {
            let up = ws.dist_int[slot + 1];
            if up != u32::MAX {
                let cand = up.saturating_add(network.pipe_cost_int(pipe as usize));
                if cand < new_d {
                    new_d = cand;
                }
            }
        }
    }
    // North predecessor.
    if y_fwd && y > 0 {
        let slot = node - w_us;
        let pipe = pipe_s[slot];
        if pipe != u32::MAX {
            let up = ws.dist_int[slot];
            if up != u32::MAX {
                let cand = up.saturating_add(network.pipe_cost_int(pipe as usize));
                if cand < new_d {
                    new_d = cand;
                }
            }
        }
    }
    // South predecessor.
    if !y_fwd && y + 1 < network.height as usize {
        let slot = node;
        let pipe = pipe_s[slot];
        if pipe != u32::MAX {
            let up = ws.dist_int[slot + w_us];
            if up != u32::MAX {
                let cand = up.saturating_add(network.pipe_cost_int(pipe as usize));
                if cand < new_d {
                    new_d = cand;
                }
            }
        }
    }

    if new_d < cur {
        ws.dist_int[node] = new_d;
        (true, new_d)
    } else {
        (false, cur)
    }
}

/// Full-graph raster sweep used when no corridor is active (fallback path).
fn full_sweep(
    network: &PipeNetwork,
    ws: &mut PathSolverWorkspace,
    x_fwd: bool,
    y_fwd: bool,
    mut max_d: u32,
) -> (usize, u32) {
    let w = network.width;
    let h = network.height;
    let w_us = w as usize;
    let mut count = 0usize;

    for yi in 0..h {
        let y = if y_fwd { yi } else { h - 1 - yi };
        let row_base = (y * w) as usize;
        for xi in 0..w {
            let x = if x_fwd { xi } else { w - 1 - xi };
            let node = row_base + x as usize;
            let (did_relax, nd) = relax_tile(network, ws, node, x_fwd, y_fwd, w_us);
            if did_relax {
                count += 1;
                if nd > max_d {
                    max_d = nd;
                }
            }
        }
    }

    (count, max_d)
}

/// Sparse relaxation of every `LongRange` pipe in scope.
///
/// With a corridor active we iterate the pre-filtered `corridor_long_pipes`
/// list (populated once per net in `mark_corridor`), avoiding a full scan of
/// every long pipe on the chip per round.
fn long_pipe_relax(
    network: &PipeNetwork,
    ws: &mut PathSolverWorkspace,
    use_corridor: bool,
    mut max_d: u32,
) -> (usize, u32) {
    let pipes = &network.pipes;
    let mut count = 0usize;

    let pipe_iter: &[u32] = if use_corridor {
        &ws.corridor_long_pipes
    } else {
        &network.tile_grid.long_pipes
    };

    for &pipe_idx in pipe_iter {
        let pipe = &pipes[pipe_idx as usize];
        let cost = network.pipe_cost_int(pipe_idx as usize);
        let from = pipe.from;
        let to = pipe.to;
        let df = ws.dist_int[from];
        let dt = ws.dist_int[to];
        if df != u32::MAX {
            let cand = df.saturating_add(cost);
            if cand < dt {
                ws.dist_int[to] = cand;
                if cand > max_d {
                    max_d = cand;
                }
                count += 1;
            }
        }
        if dt != u32::MAX {
            let cand = dt.saturating_add(cost);
            if cand < ws.dist_int[from] {
                ws.dist_int[from] = cand;
                if cand > max_d {
                    max_d = cand;
                }
                count += 1;
            }
        }
    }

    (count, max_d)
}

/// Bucket-sort every reached tile into `ws.settle_order` in ascending dist
/// order. Reuses Dial's `ws.buckets` storage; left empty on exit.
fn bucket_collect_settle_order(
    network: &PipeNetwork,
    ws: &mut PathSolverWorkspace,
    max_d: u32,
    use_corridor: bool,
) {
    let total = (network.width * network.height) as usize;
    let need_len = (max_d as usize) + 1;
    if ws.buckets.len() < need_len {
        ws.buckets.resize(need_len, Vec::new());
    }

    let use_marked = use_corridor && !ws.corridor_marked.is_empty();
    if use_marked {
        let marked_len = ws.corridor_marked.len();
        for i in 0..marked_len {
            let node = ws.corridor_marked[i];
            let d = ws.dist_int[node];
            if d != u32::MAX {
                ws.buckets[d as usize].push(node as u32);
            }
        }
    } else {
        for node in 0..total {
            let d = ws.dist_int[node];
            if d != u32::MAX {
                ws.buckets[d as usize].push(node as u32);
            }
        }
    }

    for d in 0..=max_d as usize {
        let bucket_len = ws.buckets[d].len();
        for i in 0..bucket_len {
            let tile = ws.buckets[d][i];
            ws.settle_order.push(tile as usize);
        }
        ws.buckets[d].clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::network::{Direction, Node, Pipe, PipeNetwork, PipeType, TileGrid};
    use super::super::path_solver::{dijkstra_single_source_for_test, PathSolverWorkspace};
    use super::*;
    use rustc_hash::FxHashMap;

    fn build_grid(
        w: i32,
        h: i32,
        span1_cost: f64,
        long_pipes: &[(i32, i32, i32, i32, f64)],
    ) -> PipeNetwork {
        let mut nodes = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                nodes.push(Node {
                    tile_x: x,
                    tile_y: y,
                    pressure: 0.0,
                });
            }
        }
        let mut pipes = Vec::new();
        let mut node_pipes = vec![Vec::new(); (w * h) as usize];
        let idx = |x: i32, y: i32| (y * w + x) as usize;

        for y in 0..h {
            for x in 0..w - 1 {
                let i = pipes.len();
                pipes.push(Pipe {
                    from: idx(x, y),
                    to: idx(x + 1, y),
                    base_resistance: span1_cost,
                    capacity: 10.0,
                    flow: 0.0,
                    net_count: 0.0,
                    raw_cell_density: 0.0,
                    cell_density: 0.0,
                    eff_conductance: 1.0 / span1_cost,
                    pipe_type: PipeType::InterTile(Direction::East),
                });
                node_pipes[idx(x, y)].push(i);
                node_pipes[idx(x + 1, y)].push(i);
            }
        }
        for y in 0..h - 1 {
            for x in 0..w {
                let i = pipes.len();
                pipes.push(Pipe {
                    from: idx(x, y),
                    to: idx(x, y + 1),
                    base_resistance: span1_cost,
                    capacity: 10.0,
                    flow: 0.0,
                    net_count: 0.0,
                    raw_cell_density: 0.0,
                    cell_density: 0.0,
                    eff_conductance: 1.0 / span1_cost,
                    pipe_type: PipeType::InterTile(Direction::South),
                });
                node_pipes[idx(x, y)].push(i);
                node_pipes[idx(x, y + 1)].push(i);
            }
        }
        for &(sx, sy, dx, dy, cost) in long_pipes {
            let i = pipes.len();
            pipes.push(Pipe {
                from: idx(sx, sy),
                to: idx(sx + dx, sy + dy),
                base_resistance: cost,
                capacity: 10.0,
                flow: 0.0,
                net_count: 0.0,
                raw_cell_density: 0.0,
                cell_density: 0.0,
                eff_conductance: 1.0 / cost,
                pipe_type: PipeType::LongRange { dx, dy },
            });
            node_pipes[idx(sx, sy)].push(i);
            node_pipes[idx(sx + dx, sy + dy)].push(i);
        }

        let pipe_costs: Vec<f64> = pipes
            .iter()
            .map(|p| 1.0 / p.eff_conductance.max(1e-12))
            .collect();
        let pipe_costs_int: Vec<u32> = pipe_costs
            .iter()
            .map(|&c| ((c * DIST_SCALE).round() as u32).max(1))
            .collect();
        let tile_grid = TileGrid::build(&pipes, &nodes, w, h);
        let flat_adjacency =
            crate::placer::opt_trans::network::FlatAdjacency::build(&node_pipes, &pipes);
        let tile_templates = std::sync::Arc::new(Vec::new());
        let n_pipe_entries = pipes.len();
        let n_node_entries = nodes.len();

        let mut net = PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_costs,
            pipe_history: vec![0.0; pipe_costs_int.len()],
            pipe_costs_int,
            span_cost_table: crate::placer::opt_trans::tile_cache::SpanCostTable::disabled(
                n_pipe_entries,
            ),
            flat_adjacency,
            tile_templates,
            tile_grid,
            pipe_lookup: FxHashMap::default(),
            tile_type_by_node: vec![0; n_node_entries],
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: (w * h) as usize,
            coarsen: 1,
        };
        net.refresh_flat_edge_costs();
        net
    }

    #[test]
    fn fsm_matches_dijkstra_on_uniform_grid() {
        let network = build_grid(8, 8, 1.0, &[]);
        let mut ws = PathSolverWorkspace::new(network.num_nodes(), network.num_pipes());
        fsm_dist_and_forward(&network, 0, &mut ws, false);
        let dij = dijkstra_single_source_for_test(&network, 0);
        for n in 0..network.num_nodes() {
            assert_eq!(ws.dist_int[n], dij[n], "mismatch at node {}", n);
        }
    }

    #[test]
    fn fsm_matches_dijkstra_with_long_pipe() {
        let network = build_grid(6, 6, 1.0, &[(0, 0, 5, 5, 1.5)]);
        let mut ws = PathSolverWorkspace::new(network.num_nodes(), network.num_pipes());
        fsm_dist_and_forward(&network, 0, &mut ws, false);
        let dij = dijkstra_single_source_for_test(&network, 0);
        for n in 0..network.num_nodes() {
            assert_eq!(ws.dist_int[n], dij[n], "mismatch at node {}", n);
        }
    }

    #[test]
    fn fsm_matches_dijkstra_with_nonuniform_costs() {
        let mut network = build_grid(5, 5, 1.0, &[]);
        let slot = (2 * 5 + 2) as usize;
        let pipe_idx = network.tile_grid.pipe_e[slot] as usize;
        network.pipe_costs[pipe_idx] = 10.0;
        network.pipe_costs_int[pipe_idx] =
            ((network.pipe_costs[pipe_idx] * DIST_SCALE).round() as u32).max(1);

        let mut ws = PathSolverWorkspace::new(network.num_nodes(), network.num_pipes());
        fsm_dist_and_forward(&network, 0, &mut ws, false);
        let dij = dijkstra_single_source_for_test(&network, 0);
        for n in 0..network.num_nodes() {
            assert_eq!(ws.dist_int[n], dij[n], "mismatch at node {}", n);
        }
    }
}
