//! HeAP (Heterogeneous Analytical Placer) for FPGA cell placement.
//!
//! This implements the HeAP algorithm: cells are positioned by solving a
//! quadratic optimization problem (analytical placement), then spread to
//! reduce overlap via recursive bisection, and finally legalized by snapping
//! each cell to the nearest available BEL of matching type.
//!
//! The cost function minimizes squared wirelength. For each net, connections
//! between cell pairs contribute quadratic terms. Multi-pin nets use a star
//! model with a virtual node at the net centroid. Anchor forces pull cells
//! toward their current spread positions, growing stronger each iteration.

use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::CellId;
use log::{debug, info};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::metrics::{accumulate_edge_crossings, bresenham_line};

use super::common;
use super::common::initial_placement;
use super::solver::{Solver, SparseSystem};
use super::PlacerError;

/// HeAP analytical placer.
pub struct PlacerHeap;

impl super::Placer for PlacerHeap {
    type Config = PlacerHeapCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), super::PlacerError> {
        place_heap(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[crate::netlist::CellId],
    ) -> Result<(), super::PlacerError> {
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        common::with_locked_others(ctx, &cells_set, |ctx| place_heap(ctx, cfg))
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the HeAP placer.
#[derive(Clone)]
pub struct PlacerHeapCfg {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Weight for timing cost (currently reserved, unused).
    pub timing_weight: f64,
    /// Initial spreading force multiplier. Grows by 1.5x each iteration.
    pub alpha: f64,
    /// Weight for net connections in the quadratic system.
    pub beta: f64,
    /// Maximum number of outer iterations.
    pub max_iterations: usize,
    /// Quality threshold at which spreading is considered good enough.
    pub spreading_threshold: f64,
    /// Conjugate gradient solver convergence tolerance.
    pub solver_tolerance: f64,
    /// Maximum CG solver iterations.
    pub max_solver_iters: usize,
    /// Weight for congestion-aware forces (0.0 = no congestion awareness).
    pub congestion_weight: f64,
}

impl Default for PlacerHeapCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            timing_weight: 0.5,
            alpha: 0.1,
            beta: 1.0,
            max_iterations: 20,
            spreading_threshold: 0.95,
            solver_tolerance: 1e-5,
            max_solver_iters: 100,
            congestion_weight: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// HeAP state
// ---------------------------------------------------------------------------

/// A rectangular region used during the spreading phase.
struct Region {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    /// Indices into HeapState::movable_cells (the movable cell index, not CellIdx).
    cells: Vec<usize>,
    /// Number of available BELs in this region.
    bel_count: usize,
}

/// Internal state for the HeAP algorithm.
pub struct HeapState {
    pub cfg: PlacerHeapCfg,
    /// Movable cells (alive, not locked).
    pub movable_cells: Vec<CellId>,
    /// Map from CellIdx to index in movable_cells.
    pub cell_to_idx: FxHashMap<CellId, usize>,
    /// Current X positions (continuous).
    pub cell_x: Vec<f64>,
    /// Current Y positions (continuous).
    pub cell_y: Vec<f64>,
    /// Region constraint for each movable cell (parallel to movable_cells).
    pub cell_region: Vec<Option<u32>>,
    /// Current spreading force multiplier.
    pub alpha: f64,
    /// Grid dimensions.
    pub grid_w: i32,
    pub grid_h: i32,
    /// Congestion-aware displacement targets (target_x, target_y, force_weight).
    pub congestion_targets: Option<Vec<(f64, f64, f64)>>,
}

struct BelPrefixGrid {
    width: i32,
    height: i32,
    prefix: Vec<usize>,
}

impl BelPrefixGrid {
    fn build(ctx: &Context) -> Self {
        let width = ctx.chipdb().width().max(0);
        let height = ctx.chipdb().height().max(0);
        let stride = (width + 1) as usize;
        let mut prefix = vec![0usize; ((height + 1) as usize) * stride];

        for bel in ctx.bels() {
            let loc = bel.loc();
            if loc.x < 0 || loc.y < 0 || loc.x >= width || loc.y >= height {
                continue;
            }
            let idx = ((loc.y + 1) as usize) * stride + (loc.x + 1) as usize;
            prefix[idx] += 1;
        }

        for y in 1..=height as usize {
            for x in 1..=width as usize {
                let idx = y * stride + x;
                let left = y * stride + (x - 1);
                let up = (y - 1) * stride + x;
                let up_left = (y - 1) * stride + (x - 1);
                prefix[idx] += prefix[left] + prefix[up] - prefix[up_left];
            }
        }

        Self {
            width,
            height,
            prefix,
        }
    }

    #[inline]
    fn count_in_region(&self, x0: i32, y0: i32, x1: i32, y1: i32) -> usize {
        if self.width <= 0 || self.height <= 0 {
            return 0;
        }

        let xa = x0.clamp(0, self.width - 1);
        let ya = y0.clamp(0, self.height - 1);
        let xb = x1.clamp(0, self.width - 1);
        let yb = y1.clamp(0, self.height - 1);

        if xa > xb || ya > yb {
            return 0;
        }

        let stride = (self.width + 1) as usize;
        let x0p = xa as usize;
        let y0p = ya as usize;
        let x1p = (xb + 1) as usize;
        let y1p = (yb + 1) as usize;

        let a = self.prefix[y1p * stride + x1p];
        let b = self.prefix[y0p * stride + x1p];
        let c = self.prefix[y1p * stride + x0p];
        let d = self.prefix[y0p * stride + x0p];
        a - b - c + d
    }
}

impl HeapState {
    /// Build a new HeapState from the context and configuration.
    pub fn new(ctx: &Context, cfg: &PlacerHeapCfg) -> Result<Self, PlacerError> {
        let mut movable_cells = Vec::new();
        let mut cell_to_idx = FxHashMap::default();
        let mut cell_region = Vec::new();

        for (ci, cell) in ctx.design.iter_alive_cells() {
            if !cell.bel_strength.is_locked() {
                let idx = movable_cells.len();
                cell_to_idx.insert(ci, idx);
                movable_cells.push(ci);
                cell_region.push(cell.region);
            }
        }

        let n = movable_cells.len();
        let grid_w = ctx.chipdb().width();
        let grid_h = ctx.chipdb().height();

        // Initialize cell positions: region-constrained cells start at region center,
        // unconstrained cells at grid center.
        let cx = (grid_w as f64 - 1.0) / 2.0;
        let cy = (grid_h as f64 - 1.0) / 2.0;
        let mut cell_x = vec![cx; n];
        let mut cell_y = vec![cy; n];

        for i in 0..n {
            if let Some(region_idx) = cell_region[i] {
                if let Some(bbox) = ctx.design.region(region_idx).bounding_box() {
                    cell_x[i] = (bbox.x0 as f64 + bbox.x1 as f64) / 2.0;
                    cell_y[i] = (bbox.y0 as f64 + bbox.y1 as f64) / 2.0;
                }
            }
        }

        Ok(Self {
            cfg: cfg.clone(),
            movable_cells,
            cell_to_idx,
            cell_x,
            cell_y,
            cell_region,
            alpha: cfg.alpha,
            grid_w,
            grid_h,
            congestion_targets: None,
        })
    }

    /// Perform initial random placement of all unplaced cells, then sync
    /// analytical positions from the placed BEL locations.
    fn do_initial_placement(&mut self, ctx: &mut Context) -> Result<(), PlacerError> {
        initial_placement(ctx)?;

        // Initialize analytical positions from the initial placement.
        for (idx, &cell_idx) in self.movable_cells.iter().enumerate() {
            let cell = ctx.cell(cell_idx);
            if let Some(bel) = cell.bel() {
                let loc = bel.loc();
                self.cell_x[idx] = loc.x as f64;
                self.cell_y[idx] = loc.y as f64;
            }
        }

        Ok(())
    }

    /// Build and solve the quadratic wirelength minimization system.
    ///
    /// For 2-pin nets: direct connection between the two cells.
    /// For multi-pin nets (>2 pins): star model with virtual center node.
    /// Anchor forces pull cells toward their current spread positions.
    fn solve_analytical(&mut self, ctx: &Context) -> Result<(), PlacerError> {
        let n = self.movable_cells.len();
        if n == 0 {
            return Ok(());
        }

        let mut sys_x = SparseSystem::new(n);
        let mut sys_y = SparseSystem::new(n);

        let weight = self.cfg.beta;

        // Process each net.
        for (_net_idx, net) in ctx.design.iter_alive_nets() {
            if !net.driver.is_connected() || net.users.is_empty() {
                continue;
            }

            // Collect all movable cell indices on this net, and fixed positions.
            let mut movable_on_net: Vec<usize> = Vec::new();
            let mut movable_seen: FxHashSet<usize> = FxHashSet::default();
            let mut fixed_positions: Vec<(f64, f64)> = Vec::new();

            // Driver cell.
            let drv_cell_idx = net.driver.cell;
            if let Some(&idx) = self.cell_to_idx.get(&drv_cell_idx) {
                if movable_seen.insert(idx) {
                    movable_on_net.push(idx);
                }
            } else {
                // Fixed cell: get its location.
                let cell = ctx.cell(drv_cell_idx);
                if let Some(bel) = cell.bel() {
                    let loc = bel.loc();
                    fixed_positions.push((loc.x as f64, loc.y as f64));
                }
            }

            // User cells.
            for user in &net.users {
                if !user.is_connected() {
                    continue;
                }
                let user_cell_idx = user.cell;
                if let Some(&idx) = self.cell_to_idx.get(&user_cell_idx) {
                    if movable_seen.insert(idx) {
                        movable_on_net.push(idx);
                    }
                } else {
                    let cell = ctx.cell(user_cell_idx);
                    if let Some(bel) = cell.bel() {
                        let loc = bel.loc();
                        fixed_positions.push((loc.x as f64, loc.y as f64));
                    }
                }
            }

            let total_pins = movable_on_net.len() + fixed_positions.len();
            if total_pins < 2 {
                continue;
            }

            if total_pins == 2 && movable_on_net.len() == 2 {
                // 2-pin net, both movable: direct connection.
                let i = movable_on_net[0];
                let j = movable_on_net[1];
                sys_x.add_connection(i, j, weight);
                sys_y.add_connection(i, j, weight);
            } else if total_pins == 2 && movable_on_net.len() == 1 && fixed_positions.len() == 1 {
                // 2-pin net, one movable and one fixed: anchor.
                let i = movable_on_net[0];
                let (fx, fy) = fixed_positions[0];
                sys_x.add_anchor(i, fx, weight);
                sys_y.add_anchor(i, fy, weight);
            } else {
                // Multi-pin net: star model.
                // Compute the centroid of all pins.
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                for &idx in &movable_on_net {
                    sum_x += self.cell_x[idx];
                    sum_y += self.cell_y[idx];
                }
                for &(fx, fy) in &fixed_positions {
                    sum_x += fx;
                    sum_y += fy;
                }
                let centroid_x = sum_x / total_pins as f64;
                let centroid_y = sum_y / total_pins as f64;

                // Connect each movable cell to the centroid with weight
                // proportional to 1/(num_pins - 1) to normalize.
                let star_weight = weight * (total_pins as f64) / ((total_pins - 1) as f64);
                for &idx in &movable_on_net {
                    sys_x.add_anchor(idx, centroid_x, star_weight);
                    sys_y.add_anchor(idx, centroid_y, star_weight);
                }
            }
        }

        // Add anchor forces toward current positions (spreading forces).
        for i in 0..n {
            sys_x.add_anchor(i, self.cell_x[i], self.alpha);
            sys_y.add_anchor(i, self.cell_y[i], self.alpha);
        }

        // Add congestion-aware forces.
        if let Some(ref targets) = self.congestion_targets {
            for i in 0..n {
                let (tx, ty, w) = targets[i];
                if w > 0.0 {
                    sys_x.add_anchor(i, tx, w);
                    sys_y.add_anchor(i, ty, w);
                }
            }
        }

        // Solve X and Y systems in parallel using rayon::join.
        // Split borrows so each closure gets its own &mut slice.
        let tol = self.cfg.solver_tolerance;
        let max_si = self.cfg.max_solver_iters;
        let cell_x = &mut self.cell_x;
        let cell_y = &mut self.cell_y;
        let (iters_x, iters_y) = rayon::join(
            || sys_x.solve(cell_x, tol, max_si),
            || sys_y.solve(cell_y, tol, max_si),
        );

        debug!(
            "HeAP: analytical solve: CG iters x={}, y={}",
            iters_x, iters_y
        );

        // Clamp positions to grid bounds, and region-constrained cells to their region bbox.
        let max_x = (self.grid_w - 1) as f64;
        let max_y = (self.grid_h - 1) as f64;
        for i in 0..n {
            self.cell_x[i] = self.cell_x[i].clamp(0.0, max_x);
            self.cell_y[i] = self.cell_y[i].clamp(0.0, max_y);

            if let Some(region_idx) = self.cell_region[i] {
                if let Some(bbox) = ctx.design.region(region_idx).bounding_box() {
                    self.cell_x[i] = self.cell_x[i].clamp(bbox.x0 as f64, bbox.x1 as f64);
                    self.cell_y[i] = self.cell_y[i].clamp(bbox.y0 as f64, bbox.y1 as f64);
                }
            }
        }

        Ok(())
    }

    /// Spread cells via recursive bisection to reduce overlap.
    ///
    /// Returns a quality metric in [0, 1] where 1.0 means no overlap.
    pub fn spread(&mut self, ctx: &Context) -> Result<f64, PlacerError> {
        let n = self.movable_cells.len();
        if n == 0 {
            return Ok(1.0);
        }

        let bel_grid = BelPrefixGrid::build(ctx);

        let total_bels = bel_grid.count_in_region(0, 0, self.grid_w - 1, self.grid_h - 1);

        let initial_region = Region {
            x0: 0,
            y0: 0,
            x1: self.grid_w - 1,
            y1: self.grid_h - 1,
            cells: (0..n).collect(),
            bel_count: total_bels,
        };

        let mut leaf_regions: Vec<Region> = Vec::new();
        let mut stack: Vec<Region> = vec![initial_region];

        while let Some(region) = stack.pop() {
            if region.cells.is_empty() {
                continue;
            }

            if region.cells.len() <= region.bel_count {
                leaf_regions.push(region);
                continue;
            }

            let width = region.x1 - region.x0;
            let height = region.y1 - region.y0;

            if width <= 0 && height <= 0 {
                leaf_regions.push(region);
                continue;
            }

            let split_horizontal = width >= height;

            // Compute the split midpoint along the chosen axis.
            let (lo, hi) = if split_horizontal {
                (region.x0, region.x1)
            } else {
                (region.y0, region.y1)
            };
            let mid = (lo + hi) / 2;

            if mid == lo && hi > lo {
                leaf_regions.push(region);
                continue;
            }

            // Sort cells along the split axis.
            let positions = if split_horizontal {
                &self.cell_x
            } else {
                &self.cell_y
            };
            let mut sorted_cells = region.cells.clone();
            sorted_cells.sort_by(|&a, &b| {
                positions[a]
                    .partial_cmp(&positions[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Count BELs in each sub-region.
            let (lo_bels, hi_bels) = if split_horizontal {
                (
                    bel_grid.count_in_region(region.x0, region.y0, mid, region.y1),
                    bel_grid.count_in_region(mid + 1, region.y0, region.x1, region.y1),
                )
            } else {
                (
                    bel_grid.count_in_region(region.x0, region.y0, region.x1, mid),
                    bel_grid.count_in_region(region.x0, mid + 1, region.x1, region.y1),
                )
            };

            // Split cells proportionally to BEL counts.
            let total_bels_here = lo_bels + hi_bels;
            let lo_capacity = if total_bels_here > 0 {
                (sorted_cells.len() * lo_bels) / total_bels_here
            } else {
                sorted_cells.len() / 2
            };
            let lo_capacity = lo_capacity.min(sorted_cells.len());

            let hi_cells = sorted_cells.split_off(lo_capacity);
            let lo_cells = sorted_cells;

            // Clamp cell positions into their assigned sub-region.
            let positions = if split_horizontal {
                &mut self.cell_x
            } else {
                &mut self.cell_y
            };
            for &idx in &lo_cells {
                positions[idx] = positions[idx].clamp(lo as f64, mid as f64);
            }
            for &idx in &hi_cells {
                positions[idx] = positions[idx].clamp((mid + 1) as f64, hi as f64);
            }

            // Push the two sub-regions.
            let (lo_region, hi_region) = if split_horizontal {
                (
                    Region {
                        x0: region.x0,
                        y0: region.y0,
                        x1: mid,
                        y1: region.y1,
                        cells: lo_cells,
                        bel_count: lo_bels,
                    },
                    Region {
                        x0: mid + 1,
                        y0: region.y0,
                        x1: region.x1,
                        y1: region.y1,
                        cells: hi_cells,
                        bel_count: hi_bels,
                    },
                )
            } else {
                (
                    Region {
                        x0: region.x0,
                        y0: region.y0,
                        x1: region.x1,
                        y1: mid,
                        cells: lo_cells,
                        bel_count: lo_bels,
                    },
                    Region {
                        x0: region.x0,
                        y0: mid + 1,
                        x1: region.x1,
                        y1: region.y1,
                        cells: hi_cells,
                        bel_count: hi_bels,
                    },
                )
            };
            stack.push(lo_region);
            stack.push(hi_region);
        }

        // Quality: ratio of cells that fit into their leaf regions.
        let cells_fitting: usize = leaf_regions
            .iter()
            .map(|r| r.cells.len().min(r.bel_count))
            .sum();
        let quality = cells_fitting as f64 / n as f64;

        debug!("HeAP: spreading quality = {:.4}", quality);
        Ok(quality)
    }

    /// Legalize the placement: assign each movable cell to the nearest
    /// available BEL of matching bucket type.
    ///
    /// Two-phase approach:
    ///   Phase A (parallel): compute distance-sorted BEL candidate lists per cell
    ///   Phase B (sequential): assign cells to BELs, skipping already-taken ones
    fn legalize(&mut self, ctx: &mut Context) -> Result<(), PlacerError> {
        use crate::chipdb::BelId;
        use crate::common::IdString;

        let n = self.movable_cells.len();
        if n == 0 {
            return Ok(());
        }

        // First, unbind all movable cells.
        for &cell_idx in &self.movable_cells {
            if let Some(bel_id) = ctx.cell(cell_idx).bel().map(|b| b.id()) {
                ctx.unbind_bel(bel_id);
            }
        }

        // Pre-collect BEL data per cell type into plain data (BelId, x, y)
        // so we can share across rayon threads without lifetime issues.
        let mut bel_data_cache: FxHashMap<IdString, Vec<(BelId, i32, i32)>> = FxHashMap::default();
        for &cell_idx in &self.movable_cells {
            let cell_type_id = ctx.cell(cell_idx).cell_type_id();
            bel_data_cache.entry(cell_type_id).or_insert_with(|| {
                ctx.bels_for_bucket(cell_type_id)
                    .map(|bel| {
                        let loc = bel.loc();
                        (bel.id(), loc.x, loc.y)
                    })
                    .collect()
            });
        }

        // Sort movable cells by distance from center (place outer cells first).
        let cx = (self.grid_w as f64 - 1.0) / 2.0;
        let cy = (self.grid_h as f64 - 1.0) / 2.0;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let da = (self.cell_x[a] - cx).powi(2) + (self.cell_y[a] - cy).powi(2);
            let db = (self.cell_x[b] - cx).powi(2) + (self.cell_y[b] - cy).powi(2);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Gather per-cell info for parallel phase.
        struct CellLegalizeInfo {
            idx: usize,
            cell_idx: CellId,
            cell_type_id: IdString,
            cell_type_name: String,
            cell_name: String,
            cell_region: Option<u32>,
            target_x: f64,
            target_y: f64,
        }

        let cell_infos: Vec<CellLegalizeInfo> = order
            .iter()
            .map(|&idx| {
                let cell_idx = self.movable_cells[idx];
                let cell = ctx.cell(cell_idx);
                CellLegalizeInfo {
                    idx,
                    cell_idx,
                    cell_type_id: cell.cell_type_id(),
                    cell_type_name: cell.cell_type().to_owned(),
                    cell_name: cell.name().to_owned(),
                    cell_region: self.cell_region[idx],
                    target_x: self.cell_x[idx],
                    target_y: self.cell_y[idx],
                }
            })
            .collect();

        // Phase A (parallel): compute distance-sorted BEL candidate lists.
        // Each candidate list is sorted by distance to the cell's target position.
        // Region filtering is applied here using precomputed region data.
        let region_bel_sets: FxHashMap<u32, FxHashSet<BelId>> = {
            let mut map = FxHashMap::default();
            for info in &cell_infos {
                if let Some(rid) = info.cell_region {
                    map.entry(rid).or_insert_with(|| {
                        let region = ctx.design.region(rid);
                        let mut set = FxHashSet::default();
                        if let Some(bbox) = region.bounding_box() {
                            // Collect all BELs in the region.
                            for bel in ctx.bels() {
                                let loc = bel.loc();
                                if region.contains(loc.x, loc.y)
                                    && loc.x >= bbox.x0
                                    && loc.x <= bbox.x1
                                    && loc.y >= bbox.y0
                                    && loc.y <= bbox.y1
                                {
                                    set.insert(bel.id());
                                }
                            }
                        }
                        set
                    });
                }
            }
            map
        };

        let sorted_candidates: Vec<Vec<BelId>> = cell_infos
            .par_iter()
            .map(|info| {
                let bels = match bel_data_cache.get(&info.cell_type_id) {
                    Some(b) => b,
                    None => return Vec::new(),
                };

                // Filter by region if needed, then sort by distance.
                let mut candidates: Vec<(BelId, f64)> = bels
                    .iter()
                    .filter(|&&(bel_id, _, _)| {
                        if let Some(rid) = info.cell_region {
                            region_bel_sets
                                .get(&rid)
                                .map_or(false, |s| s.contains(&bel_id))
                        } else {
                            true
                        }
                    })
                    .map(|&(bel_id, bx, by)| {
                        let dx = bx as f64 - info.target_x;
                        let dy = by as f64 - info.target_y;
                        (bel_id, dx * dx + dy * dy)
                    })
                    .collect();

                candidates.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.into_iter().map(|(id, _)| id).collect()
            })
            .collect();

        // Phase B (sequential): assign cells to nearest available BEL.
        for (i, info) in cell_infos.iter().enumerate() {
            let candidates = &sorted_candidates[i];

            if candidates.is_empty() {
                return Err(PlacerError::NoBelsAvailable(info.cell_type_name.clone()));
            }

            let mut bound = false;
            for &bel in candidates {
                if ctx.bel(bel).is_available() {
                    if !ctx.bind_bel(bel, info.cell_idx, PlaceStrength::Placer) {
                        return Err(PlacerError::PlacementFailed(format!(
                            "Failed to bind cell {} to BEL {}",
                            info.cell_name, bel,
                        )));
                    }
                    let loc = ctx.bel(bel).loc();
                    self.cell_x[info.idx] = loc.x as f64;
                    self.cell_y[info.idx] = loc.y as f64;
                    bound = true;
                    break;
                }
            }

            if !bound {
                return Err(PlacerError::NoBelsAvailable(format!(
                    "{} (no available BELs for cell {})",
                    info.cell_type_name, info.cell_name,
                )));
            }
        }

        Ok(())
    }

    /// Compute congestion-aware displacement targets for each movable cell.
    ///
    /// For each cell, estimates the local congestion gradient from the edge-demand
    /// grid and computes a target position shifted away from congested edges.
    /// Returns (target_x, target_y, force_weight) for each movable cell.
    fn compute_congestion_targets(&self, ctx: &Context) -> Vec<(f64, f64, f64)> {
        let n = self.movable_cells.len();
        let grid_w = self.grid_w;
        let grid_h = self.grid_h;
        let wu = grid_w as usize;
        let hu = grid_h as usize;
        let congestion_weight = self.cfg.congestion_weight;

        // Build capacity grids (total_wires / 4 per direction).
        let mut h_capacity = vec![vec![1.0f64; wu]; hu];
        let mut v_capacity = vec![vec![1.0f64; wu]; hu];
        for ty in 0..grid_h {
            for tx in 0..grid_w {
                let tile_idx = ty * grid_w + tx;
                let tt = ctx.chipdb().tile_type(tile_idx);
                let nwires = tt.wires.get().len() as f64;
                let cap = (nwires / 4.0).max(1.0);
                h_capacity[ty as usize][tx as usize] = cap;
                v_capacity[ty as usize][tx as usize] = cap;
            }
        }

        // Build demand grids by iterating all alive nets and tracing Bresenham lines.
        let mut h_demand = vec![vec![0.0f64; wu]; hu];
        let mut v_demand = vec![vec![0.0f64; wu]; hu];

        for (_net_idx, net) in ctx.design.iter_alive_nets() {
            if !net.driver.is_connected() || net.users.is_empty() {
                continue;
            }

            let drv_cell = ctx.cell(net.driver.cell);
            let Some(drv_bel) = drv_cell.bel() else {
                continue;
            };
            let drv_loc = drv_bel.loc();

            for user in &net.users {
                if !user.is_connected() {
                    continue;
                }
                let user_cell = ctx.cell(user.cell);
                let Some(user_bel) = user_cell.bel() else {
                    continue;
                };
                let user_loc = user_bel.loc();

                let points = bresenham_line(drv_loc.x, drv_loc.y, user_loc.x, user_loc.y);
                accumulate_edge_crossings(&points, grid_w, grid_h, &mut h_demand, &mut v_demand, 1.0);
            }
        }

        // Compute per-cell congestion displacement targets.
        let max_x = (grid_w - 1) as f64;
        let max_y = (grid_h - 1) as f64;

        (0..n)
            .map(|i| {
                let cx = self.cell_x[i];
                let cy = self.cell_y[i];
                let ix = cx.round() as i32;
                let iy = cy.round() as i32;

                // Get congestion at surrounding edges (0.0 if at boundary).
                let east_c =
                    if ix >= 0 && (ix as usize) + 1 < wu && iy >= 0 && (iy as usize) < hu {
                        h_demand[iy as usize][ix as usize]
                            / h_capacity[iy as usize][ix as usize]
                    } else {
                        0.0
                    };
                let west_c =
                    if ix > 0 && (ix as usize) < wu && iy >= 0 && (iy as usize) < hu {
                        h_demand[iy as usize][(ix - 1) as usize]
                            / h_capacity[iy as usize][(ix - 1) as usize]
                    } else {
                        0.0
                    };
                let south_c =
                    if iy >= 0 && (iy as usize) + 1 < hu && ix >= 0 && (ix as usize) < wu {
                        v_demand[iy as usize][ix as usize]
                            / v_capacity[iy as usize][ix as usize]
                    } else {
                        0.0
                    };
                let north_c =
                    if iy > 0 && (iy as usize) < hu && ix >= 0 && (ix as usize) < wu {
                        v_demand[(iy - 1) as usize][ix as usize]
                            / v_capacity[(iy - 1) as usize][ix as usize]
                    } else {
                        0.0
                    };

                // Displacement: push away from congested edges.
                let dx = west_c - east_c; // positive = push east (away from west congestion)
                let dy = north_c - south_c; // positive = push south (away from north congestion)
                let max_c = east_c.max(west_c).max(south_c).max(north_c);

                // Only apply force if congestion is above 1.0 (over-capacity).
                if max_c > 1.0 {
                    let target_x = (cx + dx).clamp(0.0, max_x);
                    let target_y = (cy + dy).clamp(0.0, max_y);
                    let force = self.alpha * congestion_weight * (max_c - 1.0);
                    (target_x, target_y, force)
                } else {
                    (cx, cy, 0.0)
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count how many BELs fall within the given rectangular region.
#[cfg(feature = "test-utils")]
pub fn count_bels_in_region(ctx: &Context, x0: i32, y0: i32, x1: i32, y1: i32) -> usize {
    let mut count = 0;
    for bel in ctx.bels() {
        let loc = bel.loc();
        if loc.x >= x0 && loc.x <= x1 && loc.y >= y0 && loc.y <= y1 {
            count += 1;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the HeAP placer on the given context.
///
/// Steps:
/// 1. Initial random placement of all unplaced cells.
/// 2. Iteratively: solve quadratic system, spread, legalize.
/// 3. Stop when spreading quality exceeds threshold or max iterations reached.
pub fn place_heap(ctx: &mut Context, cfg: &PlacerHeapCfg) -> Result<(), PlacerError> {
    ctx.reseed_rng(cfg.seed);

    info!("HeAP Placer: starting...");

    let mut state = HeapState::new(ctx, cfg)?;

    let num_cells = state.movable_cells.len();
    if num_cells == 0 {
        info!("HeAP Placer: no moveable cells, nothing to do.");
        return Ok(());
    }
    info!("HeAP Placer: {} moveable cells.", num_cells);

    state.do_initial_placement(ctx)?;
    info!("HeAP Placer: initial placement done.");

    // Track whether the solver has run with congestion targets at least once.
    // When convergence is reached before congestion forces have been applied,
    // we force one additional iteration so that congestion-aware placement is
    // reflected in the final result.
    let mut solver_used_congestion = false;

    for iter in 0..cfg.max_iterations {
        solver_used_congestion |= state.congestion_targets.is_some();
        state.solve_analytical(ctx)?;

        let quality = state.spread(ctx)?;
        state.legalize(ctx)?;

        // Compute congestion forces for the next iteration.
        if cfg.congestion_weight > 0.0 {
            state.congestion_targets = Some(state.compute_congestion_targets(ctx));
        }

        debug!(
            "HeAP Placer: iter={}, quality={:.4}, alpha={:.4}",
            iter, quality, state.alpha
        );

        if quality > cfg.spreading_threshold {
            if cfg.congestion_weight > 0.0 && !solver_used_congestion {
                // Force one more iteration so congestion targets are applied.
                state.alpha *= 1.5;
                continue;
            }
            info!(
                "HeAP Placer: converged at iteration {} (quality={:.4}).",
                iter, quality
            );
            break;
        }

        state.alpha *= 1.5;
    }

    // Final validation: check all alive cells are placed and region constraints hold.
    common::validate_all_placed(ctx)?;
    common::validate_region_constraints(ctx)?;

    info!("HeAP Placer: done.");
    Ok(())
}

