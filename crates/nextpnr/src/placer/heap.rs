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

use crate::context::Context;
use crate::netlist::CellId;
use crate::types::PlaceStrength;
use log::{debug, info};
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::initial_placement;
use super::PlacerError;

/// HeAP analytical placer.
pub struct PlacerHeap;

impl super::Placer for PlacerHeap {
    type Config = PlacerHeapCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), super::PlacerError> {
        place_heap(ctx, cfg)
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
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse linear system
// ---------------------------------------------------------------------------

/// Sparse linear system for analytical placement.
///
/// Represents the system A*x = b where A is a symmetric positive-definite
/// matrix stored as diagonal elements plus off-diagonal (i, j, weight) triples.
pub(crate) struct SparseSystem {
    /// Number of variables.
    pub n: usize,
    /// Diagonal elements of A.
    pub diag: Vec<f64>,
    /// Off-diagonal entries: (row, col, weight). Only upper triangle stored
    /// (row < col), but the matrix is treated as symmetric.
    pub off_diag: Vec<(usize, usize, f64)>,
    /// Right-hand side vector b.
    pub rhs: Vec<f64>,
}

impl SparseSystem {
    /// Create a new empty system of size n.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            diag: vec![0.0; n],
            off_diag: Vec::new(),
            rhs: vec![0.0; n],
        }
    }

    /// Add a connection between movable cells i and j with the given weight.
    ///
    /// This adds weight to A[i,i] and A[j,j], and -weight to A[i,j] and A[j,i].
    pub fn add_connection(&mut self, i: usize, j: usize, weight: f64) {
        debug_assert!(i < self.n && j < self.n);
        if i == j {
            return;
        }
        self.diag[i] += weight;
        self.diag[j] += weight;
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.off_diag.push((lo, hi, -weight));
    }

    /// Add an anchor force pulling cell i toward position pos with the given weight.
    ///
    /// Adds weight to A[i,i] and weight*pos to rhs[i].
    pub fn add_anchor(&mut self, i: usize, pos: f64, weight: f64) {
        debug_assert!(i < self.n);
        self.diag[i] += weight;
        self.rhs[i] += weight * pos;
    }

    /// Solve the system using conjugate gradient. Returns the number of
    /// iterations used.
    pub fn solve(&self, x: &mut [f64], tolerance: f64, max_iters: usize) -> usize {
        debug_assert_eq!(x.len(), self.n);
        conjugate_gradient(
            &self.diag,
            &self.off_diag,
            &self.rhs,
            x,
            tolerance,
            max_iters,
        )
    }
}

// ---------------------------------------------------------------------------
// Conjugate Gradient solver
// ---------------------------------------------------------------------------

/// Symmetric sparse matrix-vector product: result = A * x.
///
/// A is represented by its diagonal and a list of upper-triangle off-diagonal
/// entries (i, j, weight) where i < j. The matrix is symmetric, so each
/// off-diagonal entry contributes to both (i,j) and (j,i).
pub(crate) fn spmv(diag: &[f64], off_diag: &[(usize, usize, f64)], x: &[f64], result: &mut [f64]) {
    let n = diag.len();
    for i in 0..n {
        result[i] = diag[i] * x[i];
    }
    for &(i, j, w) in off_diag {
        result[i] += w * x[j];
        result[j] += w * x[i];
    }
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Conjugate Gradient solver for the symmetric positive-definite system A*x = b.
///
/// Returns the number of iterations performed.
pub(crate) fn conjugate_gradient(
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> usize {
    let n = diag.len();
    if n == 0 {
        return 0;
    }

    // r = b - A*x
    let mut ax = vec![0.0; n];
    spmv(diag, off_diag, x, &mut ax);
    let mut r: Vec<f64> = rhs
        .iter()
        .zip(ax.iter())
        .map(|(bi, axi)| bi - axi)
        .collect();

    // p = r
    let mut p = r.clone();

    let mut rs_old = dot(&r, &r);

    // If initial residual is already tiny, return immediately.
    let rhs_norm_sq = dot(rhs, rhs);
    let tol_sq = tol * tol * rhs_norm_sq.max(1e-30);

    if rs_old < tol_sq {
        return 0;
    }

    let mut ap = vec![0.0; n];

    for iter in 0..max_iters {
        // ap = A*p
        spmv(diag, off_diag, &p, &mut ap);

        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-30 {
            return iter + 1;
        }

        let alpha = rs_old / p_ap;

        // x = x + alpha * p
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // r = r - alpha * A*p
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        let rs_new = dot(&r, &r);

        if rs_new < tol_sq {
            return iter + 1;
        }

        let beta = rs_new / rs_old;

        // p = r + beta * p
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rs_old = rs_new;
    }

    max_iters
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
pub(crate) struct HeapState {
    pub cfg: PlacerHeapCfg,
    /// Movable cells (alive, not locked).
    pub movable_cells: Vec<CellId>,
    /// Map from CellIdx to index in movable_cells.
    pub cell_to_idx: FxHashMap<CellId, usize>,
    /// Current X positions (continuous).
    pub cell_x: Vec<f64>,
    /// Current Y positions (continuous).
    pub cell_y: Vec<f64>,
    /// Current spreading force multiplier.
    pub alpha: f64,
    /// Grid dimensions.
    pub grid_w: i32,
    pub grid_h: i32,
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

        for (ci, cell) in ctx.design.iter_alive_cells() {
            if !cell.bel_strength.is_locked() {
                let idx = movable_cells.len();
                cell_to_idx.insert(ci, idx);
                movable_cells.push(ci);
            }
        }

        let n = movable_cells.len();
        let grid_w = ctx.chipdb().width();
        let grid_h = ctx.chipdb().height();

        // Initialize cell positions to the center of the grid.
        let cx = (grid_w as f64 - 1.0) / 2.0;
        let cy = (grid_h as f64 - 1.0) / 2.0;
        let cell_x = vec![cx; n];
        let cell_y = vec![cy; n];

        Ok(Self {
            cfg: cfg.clone(),
            movable_cells,
            cell_to_idx,
            cell_x,
            cell_y,
            alpha: cfg.alpha,
            grid_w,
            grid_h,
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
            let drv_cell_idx = match net.driver.cell {
                Some(cell_idx) => cell_idx,
                None => continue,
            };
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
                let user_cell_idx = match user.cell {
                    Some(cell_idx) => cell_idx,
                    None => continue,
                };
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

        // Solve the two systems.
        let iters_x = sys_x.solve(
            &mut self.cell_x,
            self.cfg.solver_tolerance,
            self.cfg.max_solver_iters,
        );
        let iters_y = sys_y.solve(
            &mut self.cell_y,
            self.cfg.solver_tolerance,
            self.cfg.max_solver_iters,
        );

        debug!(
            "HeAP: analytical solve: CG iters x={}, y={}",
            iters_x, iters_y
        );

        // Clamp positions to grid bounds.
        let max_x = (self.grid_w - 1) as f64;
        let max_y = (self.grid_h - 1) as f64;
        for i in 0..n {
            self.cell_x[i] = self.cell_x[i].clamp(0.0, max_x);
            self.cell_y[i] = self.cell_y[i].clamp(0.0, max_y);
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

        // Count total BELs in the whole grid.
        let total_bels: usize = bel_grid.count_in_region(0, 0, self.grid_w - 1, self.grid_h - 1);

        // Build initial region covering the whole grid.
        let all_indices: Vec<usize> = (0..n).collect();
        let initial_region = Region {
            x0: 0,
            y0: 0,
            x1: self.grid_w - 1,
            y1: self.grid_h - 1,
            cells: all_indices,
            bel_count: total_bels,
        };

        // Recursively bisect.
        let mut leaf_regions: Vec<Region> = Vec::new();
        let mut stack: Vec<Region> = vec![initial_region];

        while let Some(region) = stack.pop() {
            if region.cells.is_empty() {
                continue;
            }

            // If cells fit in the region, this is a leaf.
            if region.cells.len() <= region.bel_count {
                leaf_regions.push(region);
                continue;
            }

            // Decide split direction: split along the longer dimension.
            let width = region.x1 - region.x0;
            let height = region.y1 - region.y0;

            if width <= 0 && height <= 0 {
                // Can't split further; treat as leaf.
                leaf_regions.push(region);
                continue;
            }

            let split_horizontal = width >= height;

            if split_horizontal {
                let mid = (region.x0 + region.x1) / 2;
                if mid == region.x0 && region.x1 > region.x0 {
                    // Cannot split meaningfully, treat as leaf.
                    leaf_regions.push(region);
                    continue;
                }

                // Partition cells.
                let mut left_cells = Vec::new();
                let mut right_cells = Vec::new();

                // Sort cells by x position for balanced splitting.
                let mut sorted_cells = region.cells.clone();
                sorted_cells.sort_by(|&a, &b| {
                    self.cell_x[a]
                        .partial_cmp(&self.cell_x[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Count BELs in each sub-region.
                let left_bels = bel_grid.count_in_region(region.x0, region.y0, mid, region.y1);
                let right_bels = bel_grid.count_in_region(mid + 1, region.y0, region.x1, region.y1);

                // Split cells proportionally to BEL counts.
                let total_bels_here = left_bels + right_bels;
                let left_capacity = if total_bels_here > 0 {
                    (sorted_cells.len() * left_bels) / total_bels_here
                } else {
                    sorted_cells.len() / 2
                };
                let left_capacity = left_capacity.max(0).min(sorted_cells.len());

                for (i, &cell_idx) in sorted_cells.iter().enumerate() {
                    if i < left_capacity {
                        left_cells.push(cell_idx);
                    } else {
                        right_cells.push(cell_idx);
                    }
                }

                // Move cells toward their assigned sub-region center.
                for &idx in &left_cells {
                    self.cell_x[idx] = self.cell_x[idx].clamp(region.x0 as f64, mid as f64);
                }
                for &idx in &right_cells {
                    self.cell_x[idx] = self.cell_x[idx].clamp((mid + 1) as f64, region.x1 as f64);
                }

                stack.push(Region {
                    x0: region.x0,
                    y0: region.y0,
                    x1: mid,
                    y1: region.y1,
                    cells: left_cells,
                    bel_count: left_bels,
                });
                stack.push(Region {
                    x0: mid + 1,
                    y0: region.y0,
                    x1: region.x1,
                    y1: region.y1,
                    cells: right_cells,
                    bel_count: right_bels,
                });
            } else {
                // Split vertically.
                let mid = (region.y0 + region.y1) / 2;
                if mid == region.y0 && region.y1 > region.y0 {
                    leaf_regions.push(region);
                    continue;
                }

                let mut bottom_cells = Vec::new();
                let mut top_cells = Vec::new();

                let mut sorted_cells = region.cells.clone();
                sorted_cells.sort_by(|&a, &b| {
                    self.cell_y[a]
                        .partial_cmp(&self.cell_y[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let bottom_bels = bel_grid.count_in_region(region.x0, region.y0, region.x1, mid);
                let top_bels = bel_grid.count_in_region(region.x0, mid + 1, region.x1, region.y1);

                let total_bels_here = bottom_bels + top_bels;
                let bottom_capacity = if total_bels_here > 0 {
                    (sorted_cells.len() * bottom_bels) / total_bels_here
                } else {
                    sorted_cells.len() / 2
                };
                let bottom_capacity = bottom_capacity.max(0).min(sorted_cells.len());

                for (i, &cell_idx) in sorted_cells.iter().enumerate() {
                    if i < bottom_capacity {
                        bottom_cells.push(cell_idx);
                    } else {
                        top_cells.push(cell_idx);
                    }
                }

                for &idx in &bottom_cells {
                    self.cell_y[idx] = self.cell_y[idx].clamp(region.y0 as f64, mid as f64);
                }
                for &idx in &top_cells {
                    self.cell_y[idx] = self.cell_y[idx].clamp((mid + 1) as f64, region.y1 as f64);
                }

                stack.push(Region {
                    x0: region.x0,
                    y0: region.y0,
                    x1: region.x1,
                    y1: mid,
                    cells: bottom_cells,
                    bel_count: bottom_bels,
                });
                stack.push(Region {
                    x0: region.x0,
                    y0: mid + 1,
                    x1: region.x1,
                    y1: region.y1,
                    cells: top_cells,
                    bel_count: top_bels,
                });
            }
        }

        // Compute quality: ratio of cells that fit into their leaf regions.
        let mut cells_fitting = 0usize;
        for region in &leaf_regions {
            cells_fitting += region.cells.len().min(region.bel_count);
        }
        let quality = if n > 0 {
            cells_fitting as f64 / n as f64
        } else {
            1.0
        };

        debug!("HeAP: spreading quality = {:.4}", quality);
        Ok(quality)
    }

    /// Legalize the placement: assign each movable cell to the nearest
    /// available BEL of matching bucket type.
    fn legalize(&mut self, ctx: &mut Context) -> Result<(), PlacerError> {
        let n = self.movable_cells.len();
        if n == 0 {
            return Ok(());
        }

        // First, unbind all movable cells.
        for &cell_idx in &self.movable_cells {
            let cell = ctx.cell(cell_idx);
            if let Some(bel) = cell.bel() {
                let bel = bel.id();
                ctx.unbind_bel(bel);
            }
        }

        // Sort movable cells by distance from center (place outer cells first
        // to give them priority for their preferred positions).
        let cx = (self.grid_w as f64 - 1.0) / 2.0;
        let cy = (self.grid_h as f64 - 1.0) / 2.0;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let da = (self.cell_x[a] - cx).powi(2) + (self.cell_y[a] - cy).powi(2);
            let db = (self.cell_x[b] - cx).powi(2) + (self.cell_y[b] - cy).powi(2);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        for &idx in &order {
            let cell_idx = self.movable_cells[idx];
            let target_x = self.cell_x[idx];
            let target_y = self.cell_y[idx];

            let (cell_type_name, cell_name) = {
                let cell = ctx.cell(cell_idx);
                (cell.cell_type().to_owned(), cell.name().to_owned())
            };

            // Find the nearest available BEL.
            let mut best_bel = None;
            let mut best_dist = f64::MAX;
            let mut has_bucket_bel = false;

            for bel in ctx.bels_for_bucket(&cell_type_name) {
                has_bucket_bel = true;
                if !bel.is_available() {
                    continue;
                }
                let loc = bel.loc();
                let dx = loc.x as f64 - target_x;
                let dy = loc.y as f64 - target_y;
                let dist = dx * dx + dy * dy;
                if dist < best_dist {
                    best_dist = dist;
                    best_bel = Some(bel.id());
                }
            }

            if !has_bucket_bel {
                return Err(PlacerError::NoBelsAvailable(cell_type_name));
            }

            match best_bel {
                Some(bel) => {
                    if !ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer) {
                        return Err(PlacerError::PlacementFailed(format!(
                            "Failed to bind cell {} to BEL {}",
                            cell_name,
                            bel,
                        )));
                    }
                    // Update analytical position to match the legal position.
                    let loc = ctx.bel(bel).loc();
                    self.cell_x[idx] = loc.x as f64;
                    self.cell_y[idx] = loc.y as f64;
                }
                None => {
                    return Err(PlacerError::NoBelsAvailable(format!(
                        "{} (no available BELs for cell {})",
                        cell_type_name,
                        cell_name,
                    )));
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count how many BELs fall within the given rectangular region.
#[cfg(test)]
pub(crate) fn count_bels_in_region(ctx: &Context, x0: i32, y0: i32, x1: i32, y1: i32) -> usize {
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

    for iter in 0..cfg.max_iterations {
        state.solve_analytical(ctx)?;
        let quality = state.spread(ctx)?;
        state.legalize(ctx)?;

        debug!(
            "HeAP Placer: iter={}, quality={:.4}, alpha={:.4}",
            iter, quality, state.alpha
        );

        if quality > cfg.spreading_threshold {
            info!(
                "HeAP Placer: converged at iteration {} (quality={:.4}).",
                iter, quality
            );
            break;
        }

        state.alpha *= 1.5;
    }

    // Final validation: check all alive cells are placed.
    for (cell_idx, cell) in ctx.design.iter_alive_cells() {
        if cell.bel.is_none() {
            return Err(PlacerError::PlacementFailed(format!(
                "Cell {} (index {}) is alive but has no BEL after placement",
                ctx.name_of(cell.name),
                cell_idx.slot()
            )));
        }
    }

    info!("HeAP Placer: done.");
    Ok(())
}

#[cfg(test)]
#[cfg(feature = "test-utils")]
mod tests {
    use super::*;
    use crate::chipdb::testutil::make_test_chipdb;
    use crate::context::Context;
    use crate::netlist::PortRef;
    use crate::types::PortType;

    fn make_context() -> Context {
        let chipdb = make_test_chipdb();
        Context::new(chipdb)
    }

    fn make_context_with_cells(n: usize) -> Context {
        assert!(n <= 4, "synthetic chipdb only has 4 BELs");
        let mut ctx = make_context();
        ctx.populate_bel_buckets();

        let cell_type = ctx.id("LUT4");
        let mut cell_names = Vec::new();

        for i in 0..n {
            let name = ctx.id(&format!("cell_{}", i));
            ctx.design.add_cell(name, cell_type);
            cell_names.push(name);
        }

        if n >= 2 {
            let net_name = ctx.id("net_0");
            let net_idx = ctx.design.add_net(net_name);
            let q_port = ctx.id("Q");
            let a_port = ctx.id("A");

            let cell0_idx = ctx.design.cell_by_name(cell_names[0]).unwrap();
            ctx.design.cell_edit(cell0_idx).add_port(q_port, PortType::Out);
            ctx.design.cell_edit(cell0_idx).set_port_net(q_port, Some(net_idx), None);

            ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
                cell: Some(cell0_idx), port: q_port, budget: 0,
            });

            for i in 1..n {
                let cell_idx = ctx.design.cell_by_name(cell_names[i]).unwrap();
                ctx.design.cell_edit(cell_idx).add_port(a_port, PortType::In);
                let user_idx = ctx.design.net_edit(net_idx).add_user_raw(PortRef {
                    cell: Some(cell_idx), port: a_port, budget: 0,
                });
                ctx.design.cell_edit(cell_idx).set_port_net(a_port, Some(net_idx), Some(user_idx));
            }
        }

        ctx
    }

    // SparseSystem tests

    #[test]
    fn sparse_system_new() {
        let sys = SparseSystem::new(3);
        assert_eq!(sys.n, 3);
        assert_eq!(sys.diag.len(), 3);
        assert_eq!(sys.rhs.len(), 3);
        assert!(sys.off_diag.is_empty());
        assert!(sys.diag.iter().all(|&d| d == 0.0));
        assert!(sys.rhs.iter().all(|&r| r == 0.0));
    }

    #[test]
    fn sparse_system_add_connection() {
        let mut sys = SparseSystem::new(3);
        sys.add_connection(0, 2, 5.0);
        assert_eq!(sys.diag[0], 5.0);
        assert_eq!(sys.diag[1], 0.0);
        assert_eq!(sys.diag[2], 5.0);
        assert_eq!(sys.off_diag.len(), 1);
        assert_eq!(sys.off_diag[0].0, 0);
        assert_eq!(sys.off_diag[0].1, 2);
        assert_eq!(sys.off_diag[0].2, -5.0);
    }

    #[test]
    fn sparse_system_add_connection_self_is_noop() {
        let mut sys = SparseSystem::new(2);
        sys.add_connection(1, 1, 3.0);
        assert_eq!(sys.diag[0], 0.0);
        assert_eq!(sys.diag[1], 0.0);
        assert!(sys.off_diag.is_empty());
    }

    #[test]
    fn sparse_system_add_anchor() {
        let mut sys = SparseSystem::new(2);
        sys.add_anchor(0, 3.0, 2.0);
        assert_eq!(sys.diag[0], 2.0);
        assert_eq!(sys.rhs[0], 6.0);
        assert_eq!(sys.diag[1], 0.0);
        assert_eq!(sys.rhs[1], 0.0);
    }

    #[test]
    fn sparse_system_solve_identity() {
        let mut sys = SparseSystem::new(3);
        sys.diag[0] = 1.0;
        sys.diag[1] = 1.0;
        sys.diag[2] = 1.0;
        sys.rhs[0] = 2.0;
        sys.rhs[1] = 5.0;
        sys.rhs[2] = -1.0;
        let mut x = vec![0.0; 3];
        let iters = sys.solve(&mut x, 1e-10, 100);
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - 5.0).abs() < 1e-6);
        assert!((x[2] - (-1.0)).abs() < 1e-6);
        assert!(iters <= 3);
    }

    #[test]
    fn sparse_system_solve_with_connections() {
        let mut sys = SparseSystem::new(2);
        sys.add_connection(0, 1, 1.0);
        sys.add_anchor(0, 0.0, 1.0);
        sys.add_anchor(1, 4.0, 1.0);
        let mut x = vec![0.0; 2];
        sys.solve(&mut x, 1e-10, 100);
        assert!((x[0] - 4.0 / 3.0).abs() < 1e-6);
        assert!((x[1] - 8.0 / 3.0).abs() < 1e-6);
    }

    // CG solver tests

    #[test]
    fn cg_identity_system() {
        let diag = vec![1.0, 1.0, 1.0];
        let off_diag: Vec<(usize, usize, f64)> = vec![];
        let rhs = vec![3.0, 7.0, -2.0];
        let mut x = vec![0.0, 0.0, 0.0];
        let iters = conjugate_gradient(&diag, &off_diag, &rhs, &mut x, 1e-10, 100);
        assert!((x[0] - 3.0).abs() < 1e-6);
        assert!((x[1] - 7.0).abs() < 1e-6);
        assert!((x[2] - (-2.0)).abs() < 1e-6);
        assert!(iters <= 3);
    }

    #[test]
    fn cg_diagonal_system() {
        let diag = vec![2.0, 3.0, 5.0];
        let off_diag: Vec<(usize, usize, f64)> = vec![];
        let rhs = vec![4.0, 9.0, 25.0];
        let mut x = vec![0.0, 0.0, 0.0];
        conjugate_gradient(&diag, &off_diag, &rhs, &mut x, 1e-10, 100);
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - 3.0).abs() < 1e-6);
        assert!((x[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cg_empty_system() {
        let diag: Vec<f64> = vec![];
        let off_diag: Vec<(usize, usize, f64)> = vec![];
        let rhs: Vec<f64> = vec![];
        let mut x: Vec<f64> = vec![];
        let iters = conjugate_gradient(&diag, &off_diag, &rhs, &mut x, 1e-10, 100);
        assert_eq!(iters, 0);
    }

    #[test]
    fn cg_single_variable() {
        let diag = vec![4.0];
        let off_diag: Vec<(usize, usize, f64)> = vec![];
        let rhs = vec![12.0];
        let mut x = vec![0.0];
        conjugate_gradient(&diag, &off_diag, &rhs, &mut x, 1e-10, 100);
        assert!((x[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn cg_with_off_diagonal() {
        let diag = vec![4.0, 4.0];
        let off_diag = vec![(0, 1, -1.0)];
        let rhs = vec![3.0, 3.0];
        let mut x = vec![0.0, 0.0];
        conjugate_gradient(&diag, &off_diag, &rhs, &mut x, 1e-10, 100);
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
    }

    // SPMV tests

    #[test]
    fn spmv_identity() {
        let diag = vec![1.0, 1.0, 1.0];
        let off_diag: Vec<(usize, usize, f64)> = vec![];
        let x = vec![3.0, 5.0, 7.0];
        let mut result = vec![0.0; 3];
        spmv(&diag, &off_diag, &x, &mut result);
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 7.0);
    }

    #[test]
    fn spmv_diagonal() {
        let diag = vec![2.0, 3.0, 4.0];
        let off_diag: Vec<(usize, usize, f64)> = vec![];
        let x = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 3];
        spmv(&diag, &off_diag, &x, &mut result);
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 6.0);
        assert_eq!(result[2], 12.0);
    }

    #[test]
    fn spmv_with_off_diagonal() {
        let diag = vec![2.0, 3.0];
        let off_diag = vec![(0, 1, -1.0)];
        let x = vec![1.0, 2.0];
        let mut result = vec![0.0; 2];
        spmv(&diag, &off_diag, &x, &mut result);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 5.0);
    }

    #[test]
    fn spmv_symmetric() {
        let diag = vec![4.0, 4.0, 4.0];
        let off_diag = vec![(0, 1, -1.0), (1, 2, -1.0)];
        let x = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 3];
        spmv(&diag, &off_diag, &x, &mut result);
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 4.0);
        assert_eq!(result[2], 10.0);
    }

    // Spreading tests

    #[test]
    fn spreading_no_cells() {
        let ctx = make_context();
        let cfg = PlacerHeapCfg::default();
        let mut state = HeapState::new(&ctx, &cfg).unwrap();
        let quality = state.spread(&ctx).unwrap();
        assert_eq!(quality, 1.0);
    }

    #[test]
    fn spreading_cells_fit() {
        let ctx = make_context_with_cells(4);
        let cfg = PlacerHeapCfg::default();
        let mut state = HeapState::new(&ctx, &cfg).unwrap();
        state.cell_x = vec![0.0, 1.0, 0.0, 1.0];
        state.cell_y = vec![0.0, 0.0, 1.0, 1.0];
        let quality = state.spread(&ctx).unwrap();
        assert!(quality >= 0.9, "quality should be high, got {}", quality);
    }

    #[test]
    fn spreading_clustered_cells() {
        let ctx = make_context_with_cells(3);
        let cfg = PlacerHeapCfg::default();
        let mut state = HeapState::new(&ctx, &cfg).unwrap();
        state.cell_x = vec![0.0, 0.0, 0.0];
        state.cell_y = vec![0.0, 0.0, 0.0];
        let quality = state.spread(&ctx).unwrap();
        assert!(quality > 0.0);
    }

    // count_bels_in_region tests

    #[test]
    fn count_bels_full_grid() {
        let ctx = make_context();
        assert_eq!(count_bels_in_region(&ctx, 0, 0, 1, 1), 4);
    }

    #[test]
    fn count_bels_single_tile() {
        let ctx = make_context();
        assert_eq!(count_bels_in_region(&ctx, 0, 0, 0, 0), 1);
    }

    #[test]
    fn count_bels_row() {
        let ctx = make_context();
        assert_eq!(count_bels_in_region(&ctx, 0, 0, 1, 0), 2);
    }

    #[test]
    fn count_bels_empty_region() {
        let ctx = make_context();
        assert_eq!(count_bels_in_region(&ctx, 5, 5, 10, 10), 0);
    }
}
