//! Hydraulic placer state: cell positions, network, solver state.
//!
//! All positions are continuous (floating-point). Demand injection and pressure
//! gradient use bilinear interpolation — no tile snapping until final legalization.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::{CellId, NetId};

use super::config::InitStrategy;
use super::network::{PipeNetwork, Port};

const PARALLEL_THRESHOLD: usize = 4096;
const ALL_PORTS: [Port; 4] = [Port::North, Port::East, Port::South, Port::West];

pub struct HydraulicState {
    pub cell_x: Vec<f64>,
    pub cell_y: Vec<f64>,
    pub cell_to_idx: FxHashMap<CellId, usize>,
    pub idx_to_cell: Vec<CellId>,
    pub network: PipeNetwork,
    /// IO centroid and initial box half-size (for expanding box).
    pub box_center: (f64, f64),
    pub box_initial_half: f64,
}

impl HydraulicState {
    pub fn new(ctx: &Context, init: InitStrategy) -> Self {
        let (cell_to_idx, idx_to_cell) = crate::placer::common::collect_movable_cells(ctx);
        let n = idx_to_cell.len();

        let network = PipeNetwork::from_context(ctx);

        // Compute IO centroid for bounding box center.
        let box_center = Self::compute_io_centroid(ctx, &network);

        // Minimum box: just enough BELs to cover all cells.
        // Average BEL density = total_bels / total_tiles.
        let mut total_bels = 0usize;
        for y in 0..network.height {
            for x in 0..network.width {
                let tile = ctx.chipdb().tile_by_xy(x, y);
                total_bels += ctx.chipdb().tile_type(tile).bels.len();
            }
        }
        let total_tiles = (network.width * network.height) as f64;
        let bel_density = (total_bels as f64 / total_tiles).max(1.0);
        let tiles_needed = n as f64 / bel_density;
        let box_initial_half = (tiles_needed.sqrt() / 2.0).max(2.0);

        // For Centroid strategy: distribute cells uniformly within the initial tight box
        // centered at the IO centroid. Distinct positions avoid demand cancellation.
        let (cell_x, cell_y) = match init {
            InitStrategy::Uniform => Self::init_uniform(n, &network),
            InitStrategy::Centroid => {
                let (mut xs, mut ys) = Self::init_uniform(n, &network);
                // Remap from full grid to initial box around IO centroid.
                let (cx, cy) = box_center;
                let half = box_initial_half;
                let max_x = (network.width - 1) as f64;
                let max_y = (network.height - 1) as f64;
                for i in 0..n {
                    let fx = xs[i] / network.width as f64;
                    let fy = ys[i] / network.height as f64;
                    xs[i] = (cx - half + 2.0 * half * fx).clamp(0.0, max_x);
                    ys[i] = (cy - half + 2.0 * half * fy).clamp(0.0, max_y);
                }
                (xs, ys)
            }
            InitStrategy::RandomBel => Self::init_from_bels(ctx, &idx_to_cell, n),
            InitStrategy::RadialCapacity => Self::init_radial_capacity(ctx, n, &network, box_center),
        };

        Self {
            cell_x,
            cell_y,
            cell_to_idx,
            idx_to_cell,
            network,
            box_center,
            box_initial_half,
        }
    }

    /// Center of mass of all fixed (locked/IO) cells. Falls back to grid center.
    fn compute_io_centroid(ctx: &Context, network: &PipeNetwork) -> (f64, f64) {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0usize;
        for (_cell_idx, cell) in ctx.design.iter_alive_cells() {
            if !cell.bel_strength.is_locked() {
                continue;
            }
            let Some(bel) = cell.bel else { continue };
            let loc = ctx.bel(bel).loc();
            sum_x += loc.x as f64;
            sum_y += loc.y as f64;
            count += 1;
        }
        if count > 0 {
            (sum_x / count as f64, sum_y / count as f64)
        } else {
            ((network.width - 1) as f64 / 2.0, (network.height - 1) as f64 / 2.0)
        }
    }

    /// Uniform grid: cells evenly distributed across the chip.
    fn init_uniform(n: usize, network: &PipeNetwork) -> (Vec<f64>, Vec<f64>) {
        let mut cell_x = vec![0.0; n];
        let mut cell_y = vec![0.0; n];
        if n > 0 {
            let w = network.width as f64;
            let h = network.height as f64;
            let cols = (n as f64).sqrt().ceil() as usize;
            let rows = (n + cols - 1) / cols;
            let dx = w / (cols as f64 + 1.0);
            let dy = h / (rows as f64 + 1.0);
            for i in 0..n {
                cell_x[i] = dx * ((i % cols) as f64 + 1.0);
                cell_y[i] = dy * ((i / cols) as f64 + 1.0);
            }
        }
        (cell_x, cell_y)
    }

    /// Random BEL: read positions from the BEL assignment done by initial_placement.
    fn init_from_bels(ctx: &Context, idx_to_cell: &[CellId], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut cell_x = vec![0.0; n];
        let mut cell_y = vec![0.0; n];
        crate::placer::common::init_positions_from_bels(ctx, idx_to_cell, &mut cell_x, &mut cell_y);
        (cell_x, cell_y)
    }

    /// Capacity-aware radial init: spread cells outward from IO centroid,
    /// filling each tile up to its BEL capacity before moving to the next ring.
    ///
    /// This gives a compact starting position (cells near centroid for strong
    /// Kirchhoff gradients) with no overlap (each tile <= capacity). Tiles are
    /// filled closest-to-centroid first, so the placement radiates outward.
    fn init_radial_capacity(
        ctx: &Context,
        n: usize,
        network: &PipeNetwork,
        center: (f64, f64),
    ) -> (Vec<f64>, Vec<f64>) {
        struct TileSlot {
            x: i32,
            y: i32,
            capacity: usize,
            dist_sq: f64,
        }

        let (cx, cy) = center;

        let mut tiles: Vec<TileSlot> = Vec::new();
        for y in 0..network.height {
            for x in 0..network.width {
                let tile = ctx.chipdb().tile_by_xy(x, y);
                let capacity = ctx.chipdb().tile_type(tile).bels.len();
                if capacity > 0 {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    tiles.push(TileSlot { x, y, capacity, dist_sq: dx * dx + dy * dy });
                }
            }
        }

        tiles.sort_by(|a, b| a.dist_sq.total_cmp(&b.dist_sq));

        let total_capacity: usize = tiles.iter().map(|t| t.capacity).sum();
        assert!(
            n <= total_capacity,
            "init_radial_capacity: {} cells exceed total BEL capacity {}",
            n,
            total_capacity,
        );

        let mut cell_x = Vec::with_capacity(n);
        let mut cell_y = Vec::with_capacity(n);
        let mut placed = 0;

        for slot in &tiles {
            if placed >= n {
                break;
            }
            let to_place = slot.capacity.min(n - placed);
            let cols = (to_place as f64).sqrt().ceil() as usize;
            let rows = (to_place + cols - 1) / cols;
            for i in 0..to_place {
                let lx = (i % cols + 1) as f64 / (cols + 1) as f64;
                let ly = (i / cols + 1) as f64 / (rows + 1) as f64;
                cell_x.push(slot.x as f64 + lx);
                cell_y.push(slot.y as f64 + ly);
            }
            placed += to_place;
        }

        (cell_x, cell_y)
    }

    pub fn num_cells(&self) -> usize {
        self.idx_to_cell.len()
    }

    /// Continuous position of a pin (movable: from cell_x/y, fixed: from BEL).
    pub fn pin_pos(&self, ctx: &Context, cell_id: CellId) -> (f64, f64) {
        if let Some(&idx) = self.cell_to_idx.get(&cell_id) {
            (self.cell_x[idx], self.cell_y[idx])
        } else {
            let cell = ctx.design.cell(cell_id);
            if let Some(bel) = cell.bel {
                let loc = ctx.bel(bel).loc();
                (loc.x as f64, loc.y as f64)
            } else {
                (self.network.width as f64 / 2.0, self.network.height as f64 / 2.0)
            }
        }
    }

    /// Nearest tile for a pin (for timing BFS and legalization diagnostics).
    pub fn pin_tile(&self, ctx: &Context, cell_id: CellId) -> (i32, i32) {
        let (x, y) = self.pin_pos(ctx, cell_id);
        let tx = (x.round() as i32).clamp(0, self.network.width - 1);
        let ty = (y.round() as i32).clamp(0, self.network.height - 1);
        (tx, ty)
    }

    /// Clamp continuous position to grid and compute bilinear cell coordinates.
    /// Returns (x0, y0, fx, fy) where x0/y0 are the lower-left tile indices
    /// and fx/fy are the fractional offsets within the cell.
    fn bilinear_cell(&self, x: f64, y: f64) -> (i32, i32, f64, f64) {
        let max_x = (self.network.width - 1) as f64;
        let max_y = (self.network.height - 1) as f64;
        let x = x.clamp(0.0, max_x);
        let y = y.clamp(0.0, max_y);

        let x0 = (x.floor() as i32).clamp(0, self.network.width - 2);
        let y0 = (y.floor() as i32).clamp(0, self.network.height - 2);
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        (x0, y0, fx, fy)
    }

    /// Bilinear weights: maps continuous (x, y) to 4 surrounding tiles with weights.
    /// Returns [(tile_x, tile_y, weight); 4].
    fn bilinear_weights(&self, x: f64, y: f64) -> [(i32, i32, f64); 4] {
        let (x0, y0, fx, fy) = self.bilinear_cell(x, y);
        [
            (x0, y0, (1.0 - fx) * (1.0 - fy)),
            (x0 + 1, y0, fx * (1.0 - fy)),
            (x0, y0 + 1, (1.0 - fx) * fy),
            (x0 + 1, y0 + 1, fx * fy),
        ]
    }

    /// Build demand vector using bilinear interpolation (continuous positions).
    ///
    /// IOs are the pumps — nets with fixed (locked) pins get boosted demand
    /// because they provide the strongest placement signal (fixed endpoints
    /// never cancel). Internal nets contribute at base level.
    ///
    /// Additional scaling: span (sqrt) and criticality (viscosity).
    /// Star model force: WA wirelength gradient pulling cells toward connected pin centroids.
    /// Fixed (IO) pins act as immovable attractors. Returns (grad_x, grad_y).
    #[allow(dead_code)]
    pub fn compute_star_force(
        &self,
        ctx: &Context,
        wl_coeff: f64,
        net_weights: Option<&FxHashMap<NetId, f64>>,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.num_cells();
        let mut grad_x = vec![0.0; n];
        let mut grad_y = vec![0.0; n];
        crate::placer::common::add_wa_wirelength_gradient(
            ctx,
            &self.cell_to_idx,
            &self.cell_x,
            &self.cell_y,
            wl_coeff,
            &mut grad_x,
            &mut grad_y,
            net_weights,
        );
        (grad_x, grad_y)
    }

    pub fn compute_net_demands(
        &self,
        ctx: &Context,
        criticality: &FxHashMap<NetId, f64>,
        timing_weight: f64,
        io_boost: f64,
        pump_gain: f64,
    ) -> Vec<f64> {
        let n_j = self.network.num_junctions();
        let mut demand = vec![0.0; n_j];
        let grid_span = (self.network.width + self.network.height) as f64;

        for (net_id, net) in ctx.design.iter_alive_nets() {
            let Some(dp) = net.driver() else { continue };

            let mut has_fixed_sink = false;
            let mut sink_positions: Vec<(f64, f64)> = Vec::new();
            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                has_fixed_sink |= ctx.design.cell(user.cell).bel_strength.is_locked();
                sink_positions.push(self.pin_pos(ctx, user.cell));
            }
            if sink_positions.is_empty() {
                continue;
            }

            let (dx, dy) = self.pin_pos(ctx, dp.cell);
            let fanout = sink_positions.len() as f64;

            // IO boost: nets with a fixed (locked) pin pump harder.
            let has_fixed_pin =
                ctx.design.cell(dp.cell).bel_strength.is_locked() || has_fixed_sink;
            let io_factor = if has_fixed_pin { io_boost } else { 1.0 };

            // Sink centroid determines net span.
            let (sum_x, sum_y) = sink_positions
                .iter()
                .fold((0.0, 0.0), |(ax, ay), &(x, y)| (ax + x, ay + y));
            let (cx, cy) = (sum_x / fanout, sum_y / fanout);

            // Span factor: sqrt for sublinear scaling.
            let span = (dx - cx).abs() + (dy - cy).abs();
            let span_factor = 1.0 + (span / grid_span).sqrt();

            // Criticality factor: viscous nets pump harder.
            let crit = criticality.get(&net_id).copied().unwrap_or(0.0);
            let crit_factor = 1.0 + crit * timing_weight;

            // Dynamic pump: nets violating timing get amplified demand.
            // Quadratic ramp steepens local gradients for stuck long-distance nets.
            let transit_factor = 1.0 + pump_gain * crit.powi(2);

            // Combined scale: IO boost × span × criticality × pump.
            let port_share = 0.25 * io_factor * span_factor * crit_factor * transit_factor;

            // Driver injects +scale, bilinearly spread across 4 ports.
            for (tx, ty, bw) in self.bilinear_weights(dx, dy) {
                let share = port_share * bw;
                for &port in &ALL_PORTS {
                    demand[self.network.junction_index(tx, ty, port)] += share;
                }
            }

            // Each sink extracts scale/fanout, bilinearly spread across 4 ports.
            let sink_share = port_share / fanout;
            for &(sx, sy) in &sink_positions {
                for (tx, ty, bw) in self.bilinear_weights(sx, sy) {
                    let share = sink_share * bw;
                    for &port in &ALL_PORTS {
                        demand[self.network.junction_index(tx, ty, port)] -= share;
                    }
                }
            }
        }

        demand
    }

    /// Per-cell demand sign for asymmetric pressure gradient.
    ///
    /// Returns a smooth sign factor per cell: positive for net sinks, negative
    /// for net sources. Used to flip the pressure gradient direction:
    /// - Source cells (drivers): negative sign → force = -∇P (toward sinks)
    /// - Sink cells: positive sign → force = +∇P (toward drivers)
    ///
    /// Uses tanh for smooth transition. The result is in [-1, 1].
    pub fn compute_cell_demand_sign(
        &self,
        ctx: &Context,
        criticality: &FxHashMap<NetId, f64>,
        timing_weight: f64,
        io_boost: f64,
    ) -> Vec<f64> {
        let n = self.num_cells();
        let mut cell_demand = vec![0.0; n];

        for (net_id, net) in ctx.design.iter_alive_nets() {
            let Some(dp) = net.driver() else { continue };
            let users = net.users();

            let has_fixed_sink = users.iter().any(|u| {
                u.is_valid() && ctx.design.cell(u.cell).bel_strength.is_locked()
            });
            let has_fixed_pin =
                ctx.design.cell(dp.cell).bel_strength.is_locked() || has_fixed_sink;
            let io_factor = if has_fixed_pin { io_boost } else { 1.0 };

            let crit = criticality.get(&net_id).copied().unwrap_or(0.0);
            let crit_factor = 1.0 + crit * timing_weight;
            let weight = io_factor * crit_factor;

            let fanout = users.iter().filter(|u| u.is_valid()).count() as f64;
            if fanout == 0.0 {
                continue;
            }

            // Driver contributes positive demand (source).
            if let Some(&idx) = self.cell_to_idx.get(&dp.cell) {
                cell_demand[idx] += weight;
            }
            // Sinks contribute negative demand (extraction).
            let sink_weight = weight / fanout;
            for user in users {
                if !user.is_valid() {
                    continue;
                }
                if let Some(&idx) = self.cell_to_idx.get(&user.cell) {
                    cell_demand[idx] -= sink_weight;
                }
            }
        }

        // Smooth sign: tanh(demand / scale).
        // Scale by median absolute value to normalize.
        let mut abs_vals: Vec<f64> = cell_demand.iter().map(|d| d.abs()).filter(|&a| a > 1e-12).collect();
        let scale = if abs_vals.is_empty() {
            1.0
        } else {
            abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            abs_vals[abs_vals.len() / 2].max(1e-6)
        };

        // Negate and smooth: sources (d>0) get sign<0, sinks (d<0) get sign>0.
        // Used as: grad = sign * ∇P, so sources move down-gradient, sinks up-gradient.
        cell_demand.iter().map(|&d| -(d / scale).tanh()).collect()
    }

    /// Per-cell viscosity from net criticality.
    ///
    /// Viscosity = 1 + alpha * max_criticality across all nets touching this cell.
    /// Critical cells (high viscosity) move slowly and settle first.
    /// Non-critical cells (low viscosity) flow freely around them.
    pub fn compute_cell_viscosity(
        &self,
        ctx: &Context,
        criticality: &FxHashMap<NetId, f64>,
        alpha: f64,
    ) -> Vec<f64> {
        let n = self.num_cells();
        let mut max_crit = vec![0.0_f64; n];

        for (net_id, net) in ctx.design.iter_alive_nets() {
            let crit = criticality.get(&net_id).copied().unwrap_or(0.0);
            if crit <= 0.0 {
                continue;
            }
            let Some(dp) = net.driver() else { continue };

            if let Some(&idx) = self.cell_to_idx.get(&dp.cell) {
                max_crit[idx] = max_crit[idx].max(crit);
            }
            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                if let Some(&idx) = self.cell_to_idx.get(&user.cell) {
                    max_crit[idx] = max_crit[idx].max(crit);
                }
            }
        }

        max_crit.iter().map(|&c| 1.0 + alpha * c).collect()
    }

    /// Bilinear gradient of a scalar field at cell position.
    ///
    /// Given 4 corner values (f00, f10, f01, f11) and fractional offsets (fx, fy):
    ///   df/dx = (1-fy)(f10-f00) + fy(f11-f01)
    ///   df/dy = (1-fx)(f01-f00) + fx(f11-f10)
    #[inline]
    fn bilinear_gradient(fx: f64, fy: f64, f00: f64, f10: f64, f01: f64, f11: f64) -> (f64, f64) {
        let gx = (1.0 - fy) * (f10 - f00) + fy * (f11 - f01);
        let gy = (1.0 - fx) * (f01 - f00) + fx * (f11 - f10);
        (gx, gy)
    }

    /// Compute per-cell gradients in parallel (or sequentially for small N),
    /// returning separate x and y component vectors.
    fn parallel_gradient(&self, f: impl Fn(usize) -> (f64, f64) + Sync) -> (Vec<f64>, Vec<f64>) {
        let n = self.num_cells();
        let pairs: Vec<(f64, f64)> = if n >= PARALLEL_THRESHOLD {
            (0..n).into_par_iter().map(&f).collect()
        } else {
            (0..n).map(f).collect()
        };
        pairs.into_iter().unzip()
    }

    /// Per-cell bilinear gradient of a 2D scalar field stored in row-major order.
    fn field_gradient(&self, field: &[f64], w: usize) -> (Vec<f64>, Vec<f64>) {
        self.parallel_gradient(|i| {
            let (x0, y0, fx, fy) = self.bilinear_cell(self.cell_x[i], self.cell_y[i]);
            let row0 = y0 as usize * w;
            let row1 = (y0 + 1) as usize * w;
            let col0 = x0 as usize;
            let col1 = (x0 + 1) as usize;
            Self::bilinear_gradient(fx, fy, field[row0 + col0], field[row0 + col1], field[row1 + col0], field[row1 + col1])
        })
    }

    /// Pressure gradient with optional Gaussian blur for multi-resolution.
    ///
    /// sigma = 0: raw pressure field (fine detail)
    /// sigma > 0: blurred pressure field (global structure, coarse-to-fine)
    ///
    /// Large sigma lets cells see long-range pressure signals and converge
    /// to global positions quickly. As sigma anneals to 0, cells refine locally.
    pub fn compute_pressure_gradient(&self, sigma: f64) -> (Vec<f64>, Vec<f64>) {
        let w = self.network.width as usize;
        let h = self.network.height as usize;

        // Build pressure map from junction pressures.
        let pressure_map: Vec<f64> = (0..h)
            .flat_map(|y| (0..w).map(move |x| self.pressure_at(x as i32, y as i32)))
            .collect();

        // Blur for multi-resolution (skip if sigma < 0.5 to avoid unnecessary copy).
        let field = if sigma >= 0.5 {
            gaussian_blur_2d(&pressure_map, w, h, sigma)
        } else {
            pressure_map
        };

        self.field_gradient(&field, w)
    }

    /// Average pressure across all 4 ports at tile (x, y).
    #[inline]
    pub fn pressure_at(&self, x: i32, y: i32) -> f64 {
        let junctions = &self.network.junctions;
        let sum: f64 = ALL_PORTS
            .iter()
            .map(|&port| junctions[self.network.junction_index(x, y, port)].pressure)
            .sum();
        sum / 4.0
    }

    /// Bilinearly interpolated pressure at a continuous position.
    fn pressure_at_continuous(&self, x: f64, y: f64) -> f64 {
        self.bilinear_weights(x, y)
            .iter()
            .map(|&(tx, ty, w)| w * self.pressure_at(tx, ty))
            .sum()
    }

    /// Pump-cost energy: sum of pressure drops from driver to sinks per net.
    ///
    /// E = sum_nets (P_driver - avg(P_sinks))
    ///
    /// A longer pipe needs a stronger pump (higher driver pressure).
    /// Always non-negative. Directly proportional to wirelength.
    pub fn compute_pump_energy(&self, ctx: &Context) -> f64 {
        let mut energy = 0.0;
        for (_net_id, net) in ctx.design.iter_alive_nets() {
            let users = net.users();
            if users.is_empty() {
                continue;
            }
            let Some(dp) = net.driver() else { continue };

            let (dx, dy) = self.pin_pos(ctx, dp.cell);
            let p_driver = self.pressure_at_continuous(dx, dy);

            let mut p_sink_sum = 0.0;
            let mut sink_count = 0usize;
            for user in users {
                if !user.is_valid() {
                    continue;
                }
                let (sx, sy) = self.pin_pos(ctx, user.cell);
                p_sink_sum += self.pressure_at_continuous(sx, sy);
                sink_count += 1;
            }

            if sink_count > 0 {
                energy += (p_driver - p_sink_sum / sink_count as f64).abs();
            }
        }
        energy
    }

    /// Continuous HPWL: sum of half-perimeter bounding boxes from continuous positions.
    /// No legalization needed — uses cell_x/cell_y directly.
    pub fn continuous_hpwl(&self, ctx: &Context) -> f64 {
        let mut total = 0.0;
        for (_, net) in ctx.design.iter_alive_nets() {
            let Some(dp) = net.driver() else { continue };

            let (dx, dy) = self.pin_pos(ctx, dp.cell);
            let (mut min_x, mut max_x) = (dx, dx);
            let (mut min_y, mut max_y) = (dy, dy);

            let mut has_valid_sink = false;
            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                has_valid_sink = true;
                let (x, y) = self.pin_pos(ctx, user.cell);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            if !has_valid_sink {
                continue;
            }

            total += (max_x - min_x) + (max_y - min_y);
        }
        total
    }

    /// Electrical transport energy: E = ½ · P^T · S.
    ///
    /// This is the total energy dissipated in the resistor network when
    /// demand S is driven through conductances to produce potentials P.
    /// Equivalent to Σ R_e · f_e² (sum of resistance × flow² over all edges).
    ///
    /// This is the natural objective of the Kirchhoff system LP = S.
    /// Lower energy = shorter effective routing paths.
    pub fn transport_energy(&self, demand: &[f64]) -> f64 {
        let mut energy = 0.0;
        for (junction, &d) in self.network.junctions.iter().zip(demand.iter()) {
            energy += junction.pressure * d;
        }
        0.5 * energy
    }

    /// Build normalized density field: bilinear splat of cell positions,
    /// divided by per-tile BEL capacity. Shared by `density_entropy` and
    /// `compute_gas_gradient`.
    fn build_density_field(&self, ctx: &Context) -> Vec<f64> {
        let w = self.network.width as usize;
        let h = self.network.height as usize;
        let n = self.num_cells();

        let mut density = vec![0.0; w * h];
        for i in 0..n {
            for (tx, ty, weight) in self.bilinear_weights(self.cell_x[i], self.cell_y[i]) {
                density[ty as usize * w + tx as usize] += weight;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let tile = ctx.chipdb().tile_by_xy(x as i32, y as i32);
                let n_bels = ctx.chipdb().tile_type(tile).bels.len().max(1) as f64;
                density[y * w + x] /= n_bels;
            }
        }
        density
    }

    /// Density entropy: S = Σ_tiles ρ·ln(ρ) + hard_wall_penalty(ρ).
    ///
    /// Measures how far the cell distribution is from uniform.
    /// ρ=1 everywhere → S ≈ 0 (well-spread). ρ >> 1 somewhere → S >> 0 (clustered).
    /// The hard wall penalty adds α·(ρ-1)³ for ρ > 1, preventing overlap.
    pub fn density_entropy(&self, ctx: &Context) -> f64 {
        let density = self.build_density_field(ctx);

        let mut entropy = 0.0;
        for &rho in &density {
            if rho > 1e-12 {
                entropy += rho * rho.ln();
            }
            if rho > 1.0 {
                let excess = rho - 1.0;
                entropy += 10.0 * excess * excess * excess;
            }
        }
        entropy
    }

    /// Cell overlap metrics from the density field.
    ///
    /// Returns (overflow_ratio, max_density, overflow_count):
    /// - overflow_ratio: fraction of occupied tiles where ρ > 1.0 (above capacity)
    /// - max_density: highest ρ value across all tiles
    /// - overflow_count: number of tiles exceeding capacity
    pub fn overlap_metrics(&self, ctx: &Context) -> (f64, f64, usize) {
        let density = self.build_density_field(ctx);
        let mut max_rho = 0.0_f64;
        let mut overflow_count = 0usize;
        let mut occupied_count = 0usize;
        for &rho in &density {
            max_rho = max_rho.max(rho);
            if rho > 1e-6 {
                occupied_count += 1;
                if rho > 1.0 {
                    overflow_count += 1;
                }
            }
        }
        let overflow_ratio = if occupied_count > 0 {
            overflow_count as f64 / occupied_count as f64
        } else {
            0.0
        };
        (overflow_ratio, max_rho, overflow_count)
    }

    /// Free energy: F = E_transport + λ · S_density.
    ///
    /// The unified objective of the placement system.
    /// Minimizing F simultaneously reduces routing energy and cell overlap.
    pub fn free_energy(&self, ctx: &Context, demand: &[f64], lambda: f64) -> f64 {
        self.transport_energy(demand) + lambda * self.density_entropy(ctx)
    }

    pub fn clamp_positions(&mut self) {
        let max_x = (self.network.width - 1) as f64;
        let max_y = (self.network.height - 1) as f64;
        crate::placer::common::clamp_positions(&mut self.cell_x, &mut self.cell_y, max_x, max_y);
    }

    /// Clamp positions to an expanding bounding box centered on the IO centroid.
    /// `progress` goes from 0.0 (initial tight box) to 1.0 (full grid).
    /// The box starts at `box_initial_half` (just enough BELs for all cells)
    /// and expands to the full grid.
    pub fn clamp_to_box(&mut self, progress: f64) {
        let (cx, cy) = self.box_center;
        let grid_x = (self.network.width - 1) as f64;
        let grid_y = (self.network.height - 1) as f64;

        // Interpolate half-extent from initial tight box to full grid extent.
        let half_x = self.box_initial_half + (grid_x - self.box_initial_half) * progress;
        let half_y = self.box_initial_half + (grid_y - self.box_initial_half) * progress;

        let (min_x, max_x) = ((cx - half_x).max(0.0), (cx + half_x).min(grid_x));
        let (min_y, max_y) = ((cy - half_y).max(0.0), (cy + half_y).min(grid_y));
        for i in 0..self.cell_x.len() {
            self.cell_x[i] = self.cell_x[i].clamp(min_x, max_x);
            self.cell_y[i] = self.cell_y[i].clamp(min_y, max_y);
        }
    }

    /// Compute gas pressure gradient from cell density and thermodynamic temperature.
    ///
    /// Models cells as ideal gas: P = κ · ρ · T where:
    /// - ρ = cell density per tile (bilinear splatted, normalized by BEL capacity)
    /// - T = base_temperature + mean(|v|^2) at each tile from cell velocities
    ///
    /// Hot regions (cells still moving) spread more; cold regions (cells settled) freeze.
    /// This naturally anneals without an artificial temperature schedule.
    ///
    /// If no velocities are provided (iter 0), uses uniform base temperature.
    pub fn compute_gas_gradient(
        &self,
        ctx: &Context,
        base_temperature: f64,
        sigma: f64,
        velocities: Option<(&[f64], &[f64])>,
    ) -> (Vec<f64>, Vec<f64>) {
        let w = self.network.width as usize;
        let h = self.network.height as usize;
        let n = self.num_cells();
        let num_tiles = w * h;

        // 1-2. Build density field: bilinear splat, normalized by BEL capacity.
        let density = self.build_density_field(ctx);

        // 3. Build temperature field from cell velocities: T(tile) = base + mean(|v|^2).
        let temperature_field = self.build_temperature_field(
            w, h, n, base_temperature, velocities,
        );

        // 4. Ideal gas pressure with hard-wall penalty at capacity.
        //    Below capacity: P = ρ · T (linear, soft spreading)
        //    Above capacity: P = ρ · T + α·(ρ-1)³ (cubic hard wall)
        const HARD_WALL_STIFFNESS: f64 = 10.0;
        let mut pressure = vec![0.0; num_tiles];
        for i in 0..num_tiles {
            let rho = density[i];
            let soft = rho * temperature_field[i];
            let hard_wall = if rho > 1.0 {
                let excess = rho - 1.0;
                HARD_WALL_STIFFNESS * excess * excess * excess
            } else {
                0.0
            };
            pressure[i] = soft + hard_wall;
        }

        // 5. Gaussian blur + gradient.
        let blurred = gaussian_blur_2d(&pressure, w, h, sigma);
        self.field_gradient(&blurred, w)
    }

    /// Build per-tile temperature field from cell velocities.
    ///
    /// Splats |v|^2 onto tiles using bilinear weights, then averages per tile.
    /// Returns base_temperature for tiles with no nearby cells.
    fn build_temperature_field(
        &self,
        w: usize,
        h: usize,
        n: usize,
        base_temperature: f64,
        velocities: Option<(&[f64], &[f64])>,
    ) -> Vec<f64> {
        let num_tiles = w * h;
        let Some((vx, vy)) = velocities else {
            return vec![base_temperature; num_tiles];
        };

        let mut ke_sum = vec![0.0; num_tiles];
        let mut cell_count = vec![0.0; num_tiles];
        for i in 0..n {
            let speed_sq = vx[i] * vx[i] + vy[i] * vy[i];
            for (tx, ty, weight) in self.bilinear_weights(self.cell_x[i], self.cell_y[i]) {
                let idx = ty as usize * w + tx as usize;
                ke_sum[idx] += weight * speed_sq;
                cell_count[idx] += weight;
            }
        }

        ke_sum
            .iter()
            .zip(&cell_count)
            .map(|(&ke, &cnt)| {
                if cnt > 1e-12 {
                    base_temperature + ke / cnt
                } else {
                    base_temperature
                }
            })
            .collect()
    }
}

/// Separable Gaussian blur on a 2D grid.
fn gaussian_blur_2d(input: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    if sigma < 0.5 {
        return input.to_vec();
    }
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = (0..=radius)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let norm: f64 = kernel[0] + 2.0 * kernel[1..].iter().sum::<f64>();

    // Horizontal pass.
    let mut temp = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = input[y * w + x] * kernel[0];
            for k in 1..=radius {
                let left = if x >= k { x - k } else { 0 };
                let right = (x + k).min(w - 1);
                sum += input[y * w + left] * kernel[k];
                sum += input[y * w + right] * kernel[k];
            }
            temp[y * w + x] = sum / norm;
        }
    }

    // Vertical pass.
    let mut output = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = temp[y * w + x] * kernel[0];
            for k in 1..=radius {
                let up = if y >= k { y - k } else { 0 };
                let down = (y + k).min(h - 1);
                sum += temp[up * w + x] * kernel[k];
                sum += temp[down * w + x] * kernel[k];
            }
            output[y * w + x] = sum / norm;
        }
    }

    output
}
