//! Global spreading potential from routing-demand overflow.
//!
//! # Why this exists
//!
//! BPR (`resistance.rs`) prices congestion as a *local static penalty* on each
//! pipe. That tells a cell "the wires next to you are full". It cannot tell it
//! "there is spare capacity forty tiles east", because the information never
//! propagates: a penalty is evaluated pointwise. Measured consequence, from the
//! FPGA01 A/B, is that scaling the penalty 200x changes nothing —
//! alpha=0.05 gives 20,500 over-capacity tile edges and alpha=10 gives 20,830.
//! Neither the magnitude nor the shape of a pointwise penalty produces a
//! coordinated multi-cell spread.
//!
//! An elliptic solve does propagate it. Solving
//!
//! ```text
//!     laplacian(phi) = -(demand - capacity)
//! ```
//!
//! turns the local overflow measurement into a global potential whose gradient
//! has chip-wide reach, which is the mechanism that makes electrostatic
//! placers spread. This module is that solve, with one deliberate difference
//! from eDensity: the right-hand side is **routing demand minus routing
//! capacity**, not cell area minus a uniform target.
//!
//! That difference is the point, not a detail:
//!
//! - It is the constraint that actually has to hold. elfPlace approximates it
//!   by spreading cell *area* uniformly and then retrofits congestion
//!   awareness via RUDY-driven cell inflation.
//! - Capacity is per-tile and heterogeneous, so a wire-poor BRAM/DSP column
//!   reads as repulsive at modest usage while a wire-rich CLB region reads as
//!   attractive. A uniform area target cannot express that.
//!
//! # Relationship to the abandoned global Kirchhoff solve
//!
//! An aggregate field was tried before and dropped, recorded as "demand
//! cancellation in global solve: different nets' demands cancel at cell
//! positions". That finding is about **wirelength**, and it is correct:
//! superimpose many nets' signed +driver/-sink demands and the per-net
//! attractive pull cancels. It does not apply here. This field's charge is
//! `usage - capacity`, an aggregate imbalance that does not cancel between
//! nets, and it is used only for **feasibility**. Wirelength stays with the
//! per-net Dijkstra star cost, which the true-HPWL median-move diagnostic
//! already showed earns its place.
//!
//! For the same reason the historical solver failures do not transfer. AMG
//! failed here on "sparse per-net RHS localized at pin positions"; this RHS is
//! dense and smooth, the regime spectral methods are best at.

use super::network::PipeNetwork;
use crate::placer::electro_place::density::compute_potential_from_charge;

/// Per-node spreading potential plus the diagnostics needed to tell whether it
/// is doing anything.
pub(crate) struct SpreadField {
    /// Potential per network node, RMS-normalized. Indexed by node, matching
    /// `PipeNetwork` node ordering (`node = y * width + x`).
    pub(crate) potential: Vec<f64>,
    /// Pipes with `net_count > capacity` at the time the field was built.
    pub(crate) overflow_pipes: usize,
    /// Total `sum(max(usage - capacity, 0))` over all pipes, in wire units.
    pub(crate) overflow_total: f64,
    /// RMS of the raw potential before normalization. Zero means the charge
    /// grid was uniform and the field carries no information.
    pub(crate) raw_rms: f64,
}

/// Accumulate per-tile routing overflow and solve for its potential.
///
/// The charge at a node is the **mean** signed capacity imbalance over the
/// pipes touching it, `sum(usage - capacity) / degree`. Signed, not clamped:
/// negative charge marks spare capacity and makes the field attract toward it,
/// which is what gives cells somewhere to go rather than merely somewhere not
/// to be.
///
/// Dividing by degree rather than summing is load-bearing. Chip-edge nodes
/// touch fewer pipes than interior ones, so a raw sum makes a uniformly loaded
/// grid look like a bowl and produces a permanent pull toward the interior
/// even when nothing is over capacity. That would perturb uncongested designs
/// for no reason. The mean is flat whenever utilization is flat, so the term
/// is inert until there is real congestion to relieve.
///
/// The result is RMS-normalized so the caller's weight has a comparable
/// meaning across designs of different size and utilization.
/// Poisson-smoothed per-pipe capacity residual, in the same units as the raw
/// relative residual `(usage - capacity) / capacity`.
///
/// This is a *preconditioner for dual ascent*, not a second objective. Plain
/// subgradient ascent, `lambda += step * residual`, only raises the price on
/// pipes that are themselves over capacity, so the signal diffuses outward one
/// pipe per outer iteration and a net whose path could detour around a hotspot
/// never learns the hotspot is there. Applying the inverse Laplacian first
/// spreads that residual across the grid in a single solve.
///
/// The fixed point is unchanged, which is the whole reason to do it this way:
/// `(-laplacian)^-1` is positive definite, so the smoothed residual is zero
/// exactly when the raw one is, and the multiplier still stops growing exactly
/// when the constraint stops being violated. Contrast the cell-side potential
/// term in `evaluate_cell_at`, which adds a *different* objective whose own
/// minimiser is a flat imbalance rather than a feasible one.
///
/// Relative residual, not raw: the dual update prices a 2-wire pipe and a
/// 40-wire pipe on the same scale, and mixing raw wire counts into the charge
/// grid would let wide pipes dominate the solve for no physical reason.
///
/// Rescaled to preserve mean |residual|. The elliptic solve changes WHERE the
/// violation is felt, not HOW MUCH of it there is, and without the rescale the
/// operator's length-squared units would silently rescale `dual_step` by a
/// factor that grows with the die.
pub(crate) fn smoothed_pipe_residual(network: &PipeNetwork) -> Vec<f64> {
    let grid_w = network.width as usize;
    let grid_h = network.height as usize;
    let n_cells = grid_w * grid_h;

    let mut charge = vec![0.0f64; n_cells];
    let mut degree = vec![0u32; n_cells];
    for pipe in &network.pipes {
        if pipe.capacity <= 0.0 {
            continue;
        }
        let rel = (pipe.net_count - pipe.capacity) / pipe.capacity;
        if pipe.from < n_cells {
            charge[pipe.from] += rel;
            degree[pipe.from] += 1;
        }
        if pipe.to < n_cells {
            charge[pipe.to] += rel;
            degree[pipe.to] += 1;
        }
    }
    // Mean, not sum: chip-edge nodes touch fewer pipes, and a raw sum would
    // read a uniformly loaded grid as a bowl. Same reasoning as
    // `compute_spread_field`.
    for (c, &d) in charge.iter_mut().zip(degree.iter()) {
        if d > 0 {
            *c /= d as f64;
        }
    }

    let raw_scale: f64 =
        charge.iter().map(|c| c.abs()).sum::<f64>() / (n_cells.max(1) as f64);
    if raw_scale <= 0.0 {
        return vec![0.0; network.pipes.len()];
    }

    let potential = compute_potential_from_charge(charge, grid_w, grid_h);
    let pot_scale: f64 =
        potential.iter().map(|v| v.abs()).sum::<f64>() / (potential.len().max(1) as f64);
    if pot_scale <= 0.0 {
        return vec![0.0; network.pipes.len()];
    }
    let renorm = raw_scale / pot_scale;

    network
        .pipes
        .iter()
        .map(|pipe| {
            if pipe.capacity <= 0.0 {
                return 0.0;
            }
            let a = potential.get(pipe.from).copied().unwrap_or(0.0);
            let b = potential.get(pipe.to).copied().unwrap_or(0.0);
            0.5 * (a + b) * renorm
        })
        .collect()
}

pub(crate) fn compute_spread_field(network: &PipeNetwork) -> SpreadField {
    let n_nodes = network.num_nodes();
    let grid_w = network.width as usize;
    let grid_h = network.height as usize;

    let mut charge = vec![0.0f64; grid_w * grid_h];
    let mut degree = vec![0u32; grid_w * grid_h];
    let mut overflow_pipes = 0usize;
    let mut overflow_total = 0.0f64;

    for pipe in &network.pipes {
        // A pipe with no declared capacity has no defined overflow. Skipping
        // matches `ResistanceModel::effective_resistance`, which returns the
        // base resistance unmodified in the same case.
        if pipe.capacity <= 0.0 {
            continue;
        }
        let excess = pipe.net_count - pipe.capacity;
        if excess > 0.0 {
            overflow_pipes += 1;
            overflow_total += excess;
        }
        if pipe.from < charge.len() {
            charge[pipe.from] += excess;
            degree[pipe.from] += 1;
        }
        if pipe.to < charge.len() {
            charge[pipe.to] += excess;
            degree[pipe.to] += 1;
        }
    }

    for (c, &d) in charge.iter_mut().zip(degree.iter()) {
        if d > 0 {
            *c /= d as f64;
        }
    }

    // Only the variation in charge produces a field; the DC mode is dropped by
    // the solve. If utilization is flat there is nothing to relieve, and the
    // solve would return a numerically-zero potential that RMS normalization
    // below would then amplify back to unit magnitude — turning float noise
    // into a real force on an uncongested design. Detect that here and return
    // an exactly-zero field instead of solving.
    let n_cells = charge.len().max(1) as f64;
    let mean_charge: f64 = charge.iter().sum::<f64>() / n_cells;
    let charge_dev: f64 = (charge
        .iter()
        .map(|c| (c - mean_charge) * (c - mean_charge))
        .sum::<f64>()
        / n_cells)
        .sqrt();
    let charge_scale = charge.iter().fold(0.0f64, |m, c| m.max(c.abs()));
    if charge_dev <= 1e-12 * charge_scale.max(1.0) {
        return SpreadField {
            potential: vec![0.0; n_nodes.max(grid_w * grid_h)],
            overflow_pipes,
            overflow_total,
            raw_rms: 0.0,
        };
    }

    let mut potential = compute_potential_from_charge(charge, grid_w, grid_h);

    let sum_sq: f64 = potential.iter().map(|v| v * v).sum();
    let raw_rms = (sum_sq / potential.len().max(1) as f64).sqrt();
    if raw_rms > 0.0 {
        let inv = 1.0 / raw_rms;
        for v in potential.iter_mut() {
            *v *= inv;
        }
    }

    // `evaluate_cell_at` indexes this by node. The DCT grid and the node grid
    // are the same shape, but assert rather than assume.
    debug_assert_eq!(potential.len(), grid_w * grid_h);
    if potential.len() < n_nodes {
        potential.resize(n_nodes, 0.0);
    }

    SpreadField {
        potential,
        overflow_pipes,
        overflow_total,
        raw_rms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::{Direction, Node, Pipe, PipeNetwork, PipeType};
    use rustc_hash::FxHashMap;

    /// Build a `w x h` grid network with east/south pipes of uniform capacity.
    fn grid_network(w: i32, h: i32, capacity: f64) -> PipeNetwork {
        let mut nodes = Vec::new();
        for y in 0..h {
            for x in 0..w {
                nodes.push(Node {
                    tile_x: x,
                    tile_y: y,
                    pressure: 0.0,
                });
            }
        }
        let idx = |x: i32, y: i32| (y * w + x) as usize;
        let mut pipes = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if x + 1 < w {
                    pipes.push(make_pipe(idx(x, y), idx(x + 1, y), capacity));
                }
                if y + 1 < h {
                    pipes.push(make_pipe(idx(x, y), idx(x, y + 1), capacity));
                }
            }
        }
        assemble(nodes, pipes, w, h)
    }

    fn make_pipe(from: usize, to: usize, capacity: f64) -> Pipe {
        Pipe {
            from,
            to,
            base_resistance: 1.0,
            capacity,
            flow: 0.0,
            net_count: 0.0,
            raw_cell_density: 0.0,
            cell_density: 0.0,
            eff_conductance: 1.0,
            dual_lambda: 0.0,
            pipe_type: PipeType::InterTile(Direction::East),
        }
    }

    fn assemble(nodes: Vec<Node>, pipes: Vec<Pipe>, w: i32, h: i32) -> PipeNetwork {
        let n = nodes.len();
        let mut node_pipes = vec![Vec::new(); n];
        for (i, pipe) in pipes.iter().enumerate() {
            node_pipes[pipe.from].push(i);
            node_pipes[pipe.to].push(i);
        }
        let pipe_costs: Vec<f64> = pipes.iter().map(|_| 1.0).collect();
        let pipe_costs_int: Vec<u32> = pipes.iter().map(|_| 1u32).collect();
        let tile_grid = crate::placer::opt_trans::network::TileGrid::build(&pipes, &nodes, w, h);
        let flat_adjacency =
            crate::placer::opt_trans::network::FlatAdjacency::build(&node_pipes, &pipes);
        let n_pipes = pipes.len();
        PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_costs,
            pipe_costs_int,
            span_cost_table: crate::placer::opt_trans::tile_cache::SpanCostTable::disabled(n_pipes),
            flat_adjacency,
            tile_templates: std::sync::Arc::new(Vec::new()),
            tile_grid,
            pipe_lookup: FxHashMap::default(),
            tile_type_by_node: vec![0; n],
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: 0,
            coarsen: 1,
        }
    }

    /// The property that makes this a preconditioner and not a second
    /// objective: no violation anywhere => no price anywhere. If the smoothed
    /// residual could be nonzero on a feasible network, dual ascent would keep
    /// raising lambda after the constraint was satisfied and the fixed point
    /// would move.
    #[test]
    fn feasible_network_gets_zero_smoothed_residual() {
        let mut net = grid_network(16, 16, 10.0);
        for pipe in net.pipes.iter_mut() {
            pipe.net_count = 4.0; // uniformly under capacity
        }
        let r = smoothed_pipe_residual(&net);
        assert_eq!(r.len(), net.pipes.len());
        for (i, v) in r.iter().enumerate() {
            assert!(
                v.abs() < 1e-9,
                "pipe {i} priced at {v} on a network with no violation"
            );
        }
    }

    /// The reason to precondition at all: a pipe that is itself WITHIN capacity
    /// but sits next to an overloaded region must still be priced, so routes
    /// are steered around the region instead of off one saturated pipe. Raw
    /// subgradient ascent gives such a pipe exactly zero.
    #[test]
    fn smoothing_prices_pipes_that_are_not_themselves_over() {
        let w = 32;
        let mut net = grid_network(w, 32, 10.0);
        let hot = (16 * w + 16) as usize;
        for pipe in net.pipes.iter_mut() {
            if pipe.from == hot || pipe.to == hot {
                pipe.net_count = 500.0;
            }
        }
        let r = smoothed_pipe_residual(&net);

        // A pipe five tiles from the hotspot, well under capacity itself.
        let near_from = (16 * w + 21) as usize;
        let near = net
            .pipes
            .iter()
            .position(|p| p.from == near_from && p.net_count < p.capacity)
            .expect("a slack pipe near the hotspot");
        let far_from = (16 * w + 30) as usize;
        let far = net
            .pipes
            .iter()
            .position(|p| p.from == far_from && p.net_count < p.capacity)
            .expect("a slack pipe far from the hotspot");

        assert!(
            r[near] > 0.0,
            "slack pipe next to a hotspot must still be priced, got {}",
            r[near]
        );
        assert!(
            r[near] > r[far],
            "price must decay with distance: near={} far={}",
            r[near],
            r[far]
        );
    }

    /// The rescale is what keeps `dual_step` meaning the same thing after the
    /// solve; without it the inverse Laplacian's length-squared units would
    /// scale the step by a factor that grows with the die.
    #[test]
    fn smoothing_preserves_residual_magnitude() {
        let w = 24;
        let mut net = grid_network(w, 24, 10.0);
        for (i, pipe) in net.pipes.iter_mut().enumerate() {
            if i % 7 == 0 {
                pipe.net_count = 30.0;
            }
        }
        let r = smoothed_pipe_residual(&net);
        let raw_mean: f64 = net
            .pipes
            .iter()
            .map(|p| ((p.net_count - p.capacity) / p.capacity).abs())
            .sum::<f64>()
            / net.pipes.len() as f64;
        let sm_mean: f64 = r.iter().map(|v| v.abs()).sum::<f64>() / r.len() as f64;
        // Node-mean charge and the pipe-midpoint read-back both average, so
        // this is a same-order check, not an identity.
        assert!(
            sm_mean > 0.1 * raw_mean && sm_mean < 10.0 * raw_mean,
            "smoothed magnitude {sm_mean} should stay within an order of magnitude of raw {raw_mean}"
        );
    }

    #[test]
    fn uniform_usage_gives_no_field() {
        // Every pipe equally loaded: the charge grid is constant, so after the
        // DC mode is dropped there is nothing left. A spreading force must not
        // invent a direction when no region is worse than any other.
        let mut net = grid_network(8, 8, 10.0);
        for pipe in net.pipes.iter_mut() {
            pipe.net_count = 5.0;
        }
        let field = compute_spread_field(&net);
        assert_eq!(field.overflow_pipes, 0);
        assert!(
            field.raw_rms < 1e-9,
            "expected a flat potential, got rms {}",
            field.raw_rms
        );
    }

    #[test]
    fn hotspot_is_a_potential_maximum() {
        // Overload the pipes around one interior node. That node must end up
        // at higher potential than the far corner, so a cost term that adds
        // `potential[node]` pushes cells away from it.
        let w = 16;
        let mut net = grid_network(w, 16, 10.0);
        let hot = (8 * w + 8) as usize;
        for pipe in net.pipes.iter_mut() {
            if pipe.from == hot || pipe.to == hot {
                pipe.net_count = 100.0;
            }
        }
        let field = compute_spread_field(&net);
        assert!(field.overflow_pipes > 0, "test setup produced no overflow");

        let far = 0usize;
        assert!(
            field.potential[hot] > field.potential[far],
            "hotspot potential {} should exceed corner {}",
            field.potential[hot],
            field.potential[far]
        );
    }

    #[test]
    fn potential_reaches_beyond_the_overloaded_pipes() {
        // The whole point of the elliptic solve: a node that touches NO
        // overloaded pipe still feels the hotspot, and feels it more the
        // closer it is. A pointwise penalty like BPR is identically zero at
        // both of these nodes.
        let w = 32;
        let mut net = grid_network(w, 32, 10.0);
        let hot = (16 * w + 16) as usize;
        for pipe in net.pipes.iter_mut() {
            if pipe.from == hot || pipe.to == hot {
                pipe.net_count = 500.0;
            }
        }
        let field = compute_spread_field(&net);

        // Neither of these touches an overloaded pipe.
        let near = (16 * w + 21) as usize; // 5 tiles away
        let far = (16 * w + 30) as usize; // 14 tiles away
        assert!(
            field.potential[near] > field.potential[far],
            "potential must decay with distance from the hotspot: near={} far={}",
            field.potential[near],
            field.potential[far]
        );
    }

    #[test]
    fn spare_capacity_reads_as_attractive() {
        // Half the chip is overloaded and half is empty. The empty half must
        // sit BELOW the mean potential, i.e. be somewhere cells want to go,
        // not merely somewhere neutral.
        let w = 16;
        let h = 16;
        let mut net = grid_network(w, h, 10.0);
        for pipe in net.pipes.iter_mut() {
            let x = net_node_x(pipe.from, w);
            if x < w / 2 {
                pipe.net_count = 40.0;
            }
        }
        let field = compute_spread_field(&net);
        let mean: f64 = field.potential.iter().sum::<f64>() / field.potential.len() as f64;
        let left = field.potential[(8 * w + 2) as usize];
        let right = field.potential[(8 * w + 13) as usize];
        assert!(
            left > mean && right < mean,
            "loaded half should be above mean and empty half below: left={} right={} mean={}",
            left,
            right,
            mean
        );
    }

    fn net_node_x(node: usize, w: i32) -> i32 {
        (node as i32) % w
    }
}
