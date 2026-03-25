//! Optimal transport placer state: cell positions, network, solver state.
//!
//! All positions are continuous (floating-point). Demand injection and pressure
//! gradient use bilinear interpolation -- no tile snapping until final legalization.

mod demand;
mod density;
mod geometry;
mod pressure;

use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::CellId;

use super::config::InitStrategy;
use super::network::PipeNetwork;

pub struct OptTransState {
    pub cell_x: Vec<f64>,
    pub cell_y: Vec<f64>,
    pub cell_to_idx: FxHashMap<CellId, usize>,
    pub idx_to_cell: Vec<CellId>,
    pub network: PipeNetwork,
    /// IO centroid and initial box half-size (for expanding box).
    pub box_center: (f64, f64),
    pub box_initial_half: f64,
    /// Average BEL count per occupied tile (tiles with capacity > 0).
    pub(super) avg_bels: f64,
    /// Cached per-tile BEL capacity for density normalization.
    /// Tiles with BELs use their real count. Tiles without BELs use a large
    /// sentinel (1e6) so density there is effectively zero -- this prevents
    /// the optimizer from parking cells on unbuildable tiles.
    pub tile_bel_cap: Vec<f64>,
}

impl OptTransState {
    pub fn new(ctx: &Context, init: InitStrategy) -> Self {
        let (cell_to_idx, idx_to_cell) = crate::placer::common::collect_movable_cells(ctx);
        let n = idx_to_cell.len();

        // Compute IO centroid in physical coords for box centering.
        // Do this on a temporary full-grid to find the center, then create
        // a cropped network around it.
        let full_w = ctx.chipdb().width();
        let full_h = ctx.chipdb().height();

        // Count total BELs and compute bel_density on full grid.
        let mut total_bels_full = 0usize;
        let mut total_tiles_with_bels = 0usize;
        for y in 0..full_h {
            for x in 0..full_w {
                let tile = ctx.chipdb().tile_by_xy(x, y);
                let bc = ctx.chipdb().tile_type(tile).bels.len();
                if bc > 0 {
                    total_bels_full += bc;
                    total_tiles_with_bels += 1;
                }
            }
        }
        let bel_density = (total_bels_full as f64 / (full_w * full_h) as f64).max(0.01);
        let tiles_needed = n as f64 / bel_density;
        let initial_half = (tiles_needed.sqrt() / 2.0).max(5.0) as i32;

        // Find center: use placed/locked cells if any, else center of LOGIC tiles.
        let (io_cx, io_cy) = {
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            let mut cnt = 0usize;
            // Try locked/placed cells first
            for (_cell_idx, cell) in ctx.design.iter_alive_cells() {
                if let Some(bel) = cell.bel {
                    let loc = ctx.bel(bel).loc();
                    sx += loc.x as f64;
                    sy += loc.y as f64;
                    cnt += 1;
                }
            }
            if cnt == 0 {
                // Fall back to center of LOGIC tiles (tiles with >2 BELs)
                for y in 0..full_h {
                    for x in 0..full_w {
                        let tile = ctx.chipdb().tile_by_xy(x, y);
                        if ctx.chipdb().tile_type(tile).bels.len() > 2 {
                            sx += x as f64;
                            sy += y as f64;
                            cnt += 1;
                        }
                    }
                }
            }
            if cnt > 0 {
                (sx / cnt as f64, sy / cnt as f64)
            } else {
                (full_w as f64 / 2.0, full_h as f64 / 2.0)
            }
        };

        // Create cropped network centered on design center.
        // Size must be large enough for cells to spread and avoid congestion.
        // Use design size x headroom factor, capped at full grid.
        let headroom = 4;  // 4x the minimum needed area
        let crop_half = (initial_half * headroom).max(15);
        let network = PipeNetwork::from_context_with_bounds(
            ctx,
            Some((io_cx as i32, io_cy as i32, crop_half)),
        );

        // Box center in virtual coords
        let box_center = (io_cx - network.x0 as f64, io_cy - network.y0 as f64);

        // Pre-compute per-tile BEL capacity and average BEL density.
        let w = network.width as usize;
        let h = network.height as usize;

        // Count distinct movable cell types to compute per-type capacity.
        let mut cell_type_set = rustc_hash::FxHashSet::default();
        for &cell_id in &idx_to_cell {
            cell_type_set.insert(ctx.design.cell(cell_id).cell_type);
        }
        let n_cell_types = cell_type_set.len().max(1);

        // Per-tile capacity = total BELs / number of cell types.
        // This ensures the continuous density field prevents any single type
        // from exceeding its fair share of BELs at each tile.
        // Tiles without BELs get tiny capacity (0.01) to repel cells.
        let mut tile_bel_cap = vec![0.01_f64; w * h];
        let mut total_bels = 0usize;
        let mut occupied_tiles = 0usize;
        for y in 0..network.height {
            for x in 0..network.width {
                let tile = ctx.chipdb().tile_by_xy(x + network.x0, y + network.y0);
                let bel_count = ctx.chipdb().tile_type(tile).bels.len();
                if bel_count > 0 {
                    tile_bel_cap[y as usize * w + x as usize] =
                        bel_count as f64 / n_cell_types as f64;
                    total_bels += bel_count;
                    occupied_tiles += 1;
                }
            }
        }
        let avg_bels = (total_bels as f64 / occupied_tiles.max(1) as f64).max(1.0);

        // Minimum box: just enough BELs to cover all cells.
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
            InitStrategy::RandomBel => Self::init_from_bels(ctx, &idx_to_cell, n, &network),
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
            avg_bels,
            tile_bel_cap,
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
            sum_x += (loc.x - network.x0) as f64;
            sum_y += (loc.y - network.y0) as f64;
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
    fn init_from_bels(ctx: &Context, idx_to_cell: &[CellId], n: usize, network: &PipeNetwork) -> (Vec<f64>, Vec<f64>) {
        let mut cell_x = vec![0.0; n];
        let mut cell_y = vec![0.0; n];
        crate::placer::common::init_positions_from_bels(ctx, idx_to_cell, &mut cell_x, &mut cell_y);
        // Convert from physical to virtual coords
        let x0 = network.x0 as f64;
        let y0 = network.y0 as f64;
        let max_x = (network.width - 1) as f64;
        let max_y = (network.height - 1) as f64;
        for i in 0..n {
            cell_x[i] = (cell_x[i] - x0).clamp(0.0, max_x);
            cell_y[i] = (cell_y[i] - y0).clamp(0.0, max_y);
        }
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
                let tile = ctx.chipdb().tile_by_xy(x + network.x0, y + network.y0);
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
}
