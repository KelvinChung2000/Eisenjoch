//! Shared helper functions used by multiple placer implementations.

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::{CellId, NetId};
use rustc_hash::{FxHashMap, FxHashSet};

/// Policy for whether GND/VCC/clock nets should contribute to placement cost.
///
/// - Constants (GND, VCC) are skipped by default because they don't route through
///   general fabric: every slice has local switch-matrix tieoffs, so including
///   them creates phantom anchors.
/// - Clocks are included by default because they still occupy topology
///   (dedicated clock tree), though the modeling is imperfect.
///
/// Env flags (shared across placers):
/// - `NPNR_INCLUDE_CONSTANTS=1`  → opt back in to GND/VCC in cost (debug / compat)
/// - `NPNR_EXCLUDE_CLOCKS=1`     → drop clock-like nets from cost
/// - `NPNR_OT_INCLUDE_CONSTANTS` and `NPNR_OT_EXCLUDE_GLOBALS` kept as legacy
///    aliases to avoid breaking old invocations.
#[derive(Clone, Copy, Debug)]
pub struct NetFilter {
    pub skip_constants: bool,
    pub skip_clocks: bool,
}

impl NetFilter {
    pub fn from_env() -> Self {
        let include_const = std::env::var("NPNR_INCLUDE_CONSTANTS").ok().as_deref() == Some("1")
            || std::env::var("NPNR_OT_INCLUDE_CONSTANTS").ok().as_deref() == Some("1");
        let skip_clocks = std::env::var("NPNR_EXCLUDE_CLOCKS").ok().as_deref() == Some("1")
            || std::env::var("NPNR_OT_EXCLUDE_CLOCKS").ok().as_deref() == Some("1")
            || std::env::var("NPNR_OT_EXCLUDE_GLOBALS").ok().as_deref() == Some("1");
        Self {
            skip_constants: !include_const,
            skip_clocks,
        }
    }

    /// Returns true if this net should be excluded from the placement cost
    /// and from HPWL reported for acceptance decisions.
    pub fn should_skip(&self, ctx: &Context, net_id: NetId) -> bool {
        let net_name = ctx.name_of(ctx.design.net(net_id).name);
        if self.skip_constants
            && (net_name == "$PACKER_GND_NET" || net_name == "$PACKER_VCC_NET")
        {
            return true;
        }
        if self.skip_clocks {
            let lower = net_name.to_ascii_lowercase();
            if lower.contains("clk") || lower.contains("clock") {
                return true;
            }
        }
        false
    }
}

#[inline]
fn scatter_bilinear_tile(
    map: &mut FxHashMap<(i32, i32), f64>,
    x: f64,
    y: f64,
    weight: f64,
    grid_w: usize,
    grid_h: usize,
) {
    if weight <= 0.0 || grid_w == 0 || grid_h == 0 {
        return;
    }

    let x = x.clamp(0.0, grid_w.saturating_sub(1) as f64);
    let y = y.clamp(0.0, grid_h.saturating_sub(1) as f64);

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(grid_w as i32 - 1);
    let y1 = (y0 + 1).min(grid_h as i32 - 1);
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    *map.entry((x0, y0)).or_insert(0.0) += weight * w00;
    *map.entry((x1, y0)).or_insert(0.0) += weight * w10;
    *map.entry((x0, y1)).or_insert(0.0) += weight * w01;
    *map.entry((x1, y1)).or_insert(0.0) += weight * w11;
}

/// Lock cells whose BELs only exist on boundary/IO tiles.
///
/// These cells (IOB, clock buffers, etc.) cannot benefit from the continuous
/// solve because their valid positions are sparse and fixed at the chip edge.
/// Locking them makes them fixed anchors in the Kirchhoff system, naturally
/// pulling connected logic toward the boundary via pressure gradients.
pub(crate) fn lock_boundary_cells(ctx: &mut Context) {
    // Collect all cell types present in the design.
    let mut cell_types: FxHashSet<crate::common::IdString> = FxHashSet::default();
    for (_id, cell) in ctx.design.iter_alive_cells() {
        cell_types.insert(cell.cell_type);
    }

    // For each cell type, check if its BELs are on CLB tiles (>= 4 BELs per tile).
    // Cell types with NO BELs on CLB tiles are boundary/IO types.
    let mut clb_cell_types: FxHashSet<crate::common::IdString> = FxHashSet::default();
    for &ct in &cell_types {
        let on_clb = ctx.bels_for_bucket(ct).any(|b| {
            let loc = b.loc();
            let tile = ctx.chipdb().tile_by_xy(loc.x, loc.y);
            ctx.chipdb().tile_type(tile).bels.len() >= 4
        });
        if on_clb {
            clb_cell_types.insert(ct);
        }
    }

    let mut locked_count = 0usize;
    let cell_ids: Vec<_> = ctx.design.iter_alive_cells().map(|(id, _)| id).collect();

    for cell_id in cell_ids {
        let cell = ctx.design.cell(cell_id);
        if cell.bel_strength.is_locked() {
            continue;
        }
        if clb_cell_types.contains(&cell.cell_type) {
            continue;
        }
        if cell.bel.is_some() {
            ctx.design.cell_mut(cell_id).bel_strength = PlaceStrength::Locked;
            locked_count += 1;
        }
    }

    if locked_count > 0 {
        eprintln!("Locked {} boundary/IO cells as fixed anchors", locked_count);
    }
}

use super::PlacerError;

/// Collect all live, placeable cell indices grouped by their cell type.
///
/// Returns a map from cell type IdString to the list of CellIdx values.
pub(crate) fn cells_by_type(ctx: &Context) -> FxHashMap<IdString, Vec<CellId>> {
    let mut map: FxHashMap<IdString, Vec<CellId>> = FxHashMap::default();
    for (cell_idx, cell) in ctx.design.iter_alive_cells() {
        map.entry(cell.cell_type).or_default().push(cell_idx);
    }
    map
}

/// Place all unplaced cells at random valid BELs.
///
/// Groups cells by type/bucket, collects available BELs of the matching bucket,
/// shuffles the BELs, and assigns cells sequentially.
/// Region-constrained cells are only placed on BELs within their region.
pub fn initial_placement(ctx: &mut Context) -> Result<(), PlacerError> {
    ctx.populate_bel_buckets();

    // Place region-constrained cells first, then unconstrained.
    let grouped = cells_by_type(ctx);

    for (&cell_type, cell_indices) in &grouped {
        let cell_type_name = ctx.name_of(cell_type).to_owned();

        // Separate unplaced cells into constrained and unconstrained.
        let mut constrained: Vec<(CellId, u32)> = Vec::new();
        let mut unconstrained: Vec<CellId> = Vec::new();
        for &ci in cell_indices {
            let cell = &ctx.design.cell(ci);
            if cell.bel.is_some() {
                continue; // already placed
            }
            if let Some(region_idx) = cell.region {
                constrained.push((ci, region_idx));
            } else {
                unconstrained.push(ci);
            }
        }

        // Place constrained cells first.
        for (ci, region_idx) in &constrained {
            let region_bels = ctx
                .bels_for_bucket_in_region(cell_type, *region_idx)
                .to_vec();
            let mut available: Vec<BelId> = region_bels
                .iter()
                .copied()
                .filter(|b| ctx.bel(*b).is_available())
                .collect();

            if available.is_empty() {
                let cell_name = ctx.cell(*ci).name_id();
                return Err(PlacerError::NoBelsAvailable(format!(
                    "{} in region (cell {})",
                    cell_type_name,
                    ctx.name_of(cell_name)
                )));
            }

            ctx.rng_mut().shuffle(&mut available);
            let bel = available[0];
            if !ctx.bind_bel(bel, *ci, PlaceStrength::Placer) {
                let cell_name = ctx.cell(*ci).name_id();
                return Err(PlacerError::InitialPlacementFailed(
                    ctx.name_of(cell_name).to_owned(),
                ));
            }
        }

        // Place unconstrained cells.
        if !unconstrained.is_empty() {
            let bucket_bels: Vec<_> = ctx.bels_for_bucket(cell_type).map(|bel| bel.id()).collect();
            if bucket_bels.is_empty() {
                return Err(PlacerError::NoBelsAvailable(cell_type_name));
            }

            let mut available: Vec<BelId> = bucket_bels
                .iter()
                .copied()
                .filter(|b| ctx.bel(*b).is_available())
                .collect();

            ctx.rng_mut().shuffle(&mut available);

            if unconstrained.len() > available.len() {
                return Err(PlacerError::NoBelsAvailable(format!(
                    "{} (need {} BELs but only {} available)",
                    cell_type_name,
                    unconstrained.len(),
                    available.len()
                )));
            }

            for (i, &cell_idx) in unconstrained.iter().enumerate() {
                let bel = available[i];
                if !ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer) {
                    let cell_name = ctx.cell(cell_idx).name_id();
                    return Err(PlacerError::InitialPlacementFailed(
                        ctx.name_of(cell_name).to_owned(),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Validate that all alive cells with region constraints are placed within their region.
pub(crate) fn validate_region_constraints(ctx: &Context) -> Result<(), PlacerError> {
    for (_cell_idx, cell) in ctx.design.iter_alive_cells() {
        if let (Some(region_idx), Some(bel)) = (cell.region, cell.bel) {
            let region = ctx.design.region(region_idx);
            let loc = ctx.bel(bel).loc();
            if !region.contains(loc.x, loc.y) {
                return Err(PlacerError::PlacementFailed(format!(
                    "Cell {} placed at ({},{}) violates region constraint",
                    ctx.name_of(cell.name),
                    loc.x,
                    loc.y,
                )));
            }
        }
    }
    Ok(())
}

/// Validate that all alive cells have been placed on a BEL.
pub(crate) fn validate_all_placed(ctx: &Context) -> Result<(), PlacerError> {
    for (cell_idx, cell) in ctx.design.iter_alive_cells() {
        if cell.bel.is_none() {
            return Err(PlacerError::PlacementFailed(format!(
                "Cell {} (index {}) is alive but has no BEL after placement",
                ctx.name_of(cell.name),
                cell_idx.slot()
            )));
        }
    }
    Ok(())
}

/// Collect movable (non-locked, cluster-root-only) cells for analytical placement.
///
/// Returns (cell_to_idx, idx_to_cell) where cell_to_idx maps CellId to solver index.
pub(crate) fn collect_movable_cells(ctx: &Context) -> (FxHashMap<CellId, usize>, Vec<CellId>) {
    let mut cell_to_idx = FxHashMap::default();
    let mut idx_to_cell = Vec::new();

    for (cell_idx, cell) in ctx.design.iter_alive_cells() {
        if cell.bel_strength.is_locked() {
            continue;
        }
        if let Some(root_id) = cell.cluster {
            if root_id != cell_idx {
                continue;
            }
        }
        let idx = idx_to_cell.len();
        cell_to_idx.insert(cell_idx, idx);
        idx_to_cell.push(cell_idx);
    }

    (cell_to_idx, idx_to_cell)
}

/// Initialize continuous positions from current BEL placements.
///
/// Cells without a BEL are placed at the grid center.
pub(crate) fn init_positions_from_bels(
    ctx: &Context,
    idx_to_cell: &[CellId],
    cell_x: &mut [f64],
    cell_y: &mut [f64],
) {
    let w = ctx.chipdb().width();
    let h = ctx.chipdb().height();

    for (i, &cell_id) in idx_to_cell.iter().enumerate() {
        let cell = ctx.design.cell(cell_id);
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            cell_x[i] = loc.x as f64;
            cell_y[i] = loc.y as f64;
        } else {
            cell_x[i] = w as f64 / 2.0;
            cell_y[i] = h as f64 / 2.0;
        }
    }
}

/// Initialize continuous positions by dropping all movable cells at the centroid
/// of fixed/IO anchor pins, with small deterministic jitter to break symmetry.
///
/// This creates a compact initial placement suitable for force-directed or
/// pressure-based placers that spread cells outward from a dense cluster.
/// Falls back to the grid center if no fixed cells exist.
pub(crate) fn init_positions_center_drop(
    ctx: &Context,
    idx_to_cell: &[CellId],
    cell_x: &mut [f64],
    cell_y: &mut [f64],
) {
    let w = ctx.chipdb().width();
    let h = ctx.chipdb().height();

    // Compute centroid of all locked/fixed cells.
    let mut cx_sum = 0.0f64;
    let mut cy_sum = 0.0f64;
    let mut n_fixed = 0usize;
    for (_cell_id, cell) in ctx.design.iter_alive_cells() {
        if !cell.bel_strength.is_locked() {
            continue;
        }
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            cx_sum += loc.x as f64;
            cy_sum += loc.y as f64;
            n_fixed += 1;
        }
    }

    let (center_x, center_y) = if n_fixed > 0 {
        (cx_sum / n_fixed as f64, cy_sum / n_fixed as f64)
    } else {
        (w as f64 / 2.0, h as f64 / 2.0)
    };

    // Place all movable cells at centroid with small deterministic jitter.
    let n = idx_to_cell.len();
    for i in 0..n {
        let hash_x = ((i as u64).wrapping_mul(2654435761)) as f64 / u64::MAX as f64;
        let hash_y = ((i as u64 + 1).wrapping_mul(2246822519)) as f64 / u64::MAX as f64;
        cell_x[i] = center_x + (hash_x - 0.5) * 2.0;
        cell_y[i] = center_y + (hash_y - 0.5) * 2.0;
    }

    // Clamp to grid bounds.
    let max_x = (w - 1) as f64;
    let max_y = (h - 1) as f64;
    clamp_positions(cell_x, cell_y, max_x, max_y);

    eprintln!(
        "  center-drop init: centroid=({:.1},{:.1}), n_fixed={}, n_movable={}",
        center_x, center_y, n_fixed, n,
    );
}

/// Compute WA wirelength gradient for all nets, accumulating into grad_x/grad_y.
///
/// Same pattern as `add_wirelength_gradient` but uses WA instead of LSE.
/// Accepts optional net weights for timing-driven placement.
pub(crate) fn add_wa_wirelength_gradient(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    wl_coeff: f64,
    grad_x: &mut [f64],
    grad_y: &mut [f64],
    net_weights: Option<&FxHashMap<crate::netlist::NetId, f64>>,
) {
    use crate::solver::wa;

    let mut pin_xs = Vec::new();
    let mut pin_ys = Vec::new();
    let mut pin_indices = Vec::new();
    let mut net_grad_x = Vec::new();
    let mut net_grad_y = Vec::new();

    for (net_id, net) in ctx.design.iter_alive_nets() {
        pin_xs.clear();
        pin_ys.clear();
        pin_indices.clear();

        if let Some(driver_pin) = net.driver() {
            collect_pin_position_xy(
                ctx,
                cell_to_idx,
                cell_x,
                cell_y,
                driver_pin.cell,
                &mut pin_xs,
                &mut pin_ys,
                &mut pin_indices,
            );
        }

        for user in net.users().iter() {
            collect_pin_position_xy(
                ctx,
                cell_to_idx,
                cell_x,
                cell_y,
                user.cell,
                &mut pin_xs,
                &mut pin_ys,
                &mut pin_indices,
            );
        }

        if pin_xs.len() < 2 {
            continue;
        }

        let net_weight = net_weights
            .and_then(|w| w.get(&net_id))
            .copied()
            .unwrap_or(1.0);

        net_grad_x.clear();
        net_grad_x.resize(pin_xs.len(), 0.0);
        net_grad_y.clear();
        net_grad_y.resize(pin_ys.len(), 0.0);
        wa::wa_axis_grad(&pin_xs, wl_coeff, &mut net_grad_x);
        wa::wa_axis_grad(&pin_ys, wl_coeff, &mut net_grad_y);

        for (k, &solver_idx) in pin_indices.iter().enumerate() {
            if solver_idx != usize::MAX {
                grad_x[solver_idx] += net_weight * net_grad_x[k];
                grad_y[solver_idx] += net_weight * net_grad_y[k];
            }
        }
    }
}

/// Collect position of a single pin for WA gradient computation (separate x/y arrays).
fn collect_pin_position_xy(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    cell_id: CellId,
    xs: &mut Vec<f64>,
    ys: &mut Vec<f64>,
    indices: &mut Vec<usize>,
) {
    if let Some(&idx) = cell_to_idx.get(&cell_id) {
        xs.push(cell_x[idx]);
        ys.push(cell_y[idx]);
        indices.push(idx);
    } else {
        let cell = ctx.design.cell(cell_id);
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            xs.push(loc.x as f64);
            ys.push(loc.y as f64);
            indices.push(usize::MAX);
        }
    }
}

/// Compute per-cell pin weights for the WA preconditioner.
///
/// For each movable cell (by solver index), accumulates the sum of `1 / net_degree`
/// over every net the cell connects to. This measures how "connected" a cell is:
/// cells on many low-fanout nets get higher weights than cells on a few high-fanout nets.
pub(crate) fn compute_pin_weights(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    n: usize,
) -> Vec<f64> {
    let mut weights = vec![0.0; n];
    for (_, net) in ctx.design.iter_alive_nets() {
        let driver = net.driver();
        let users = net.users();
        let degree = driver.is_some() as usize + users.len();
        if degree < 2 {
            continue;
        }
        let w = 1.0 / degree as f64;
        if let Some(dp) = driver {
            if let Some(&idx) = cell_to_idx.get(&dp.cell) {
                weights[idx] += w;
            }
        }
        for user in users {
            if let Some(&idx) = cell_to_idx.get(&user.cell) {
                weights[idx] += w;
            }
        }
    }
    weights
}

/// Clamp positions to grid bounds.
pub(crate) fn clamp_positions(cell_x: &mut [f64], cell_y: &mut [f64], max_x: f64, max_y: f64) {
    for i in 0..cell_x.len() {
        cell_x[i] = cell_x[i].clamp(0.0, max_x);
        cell_y[i] = cell_y[i].clamp(0.0, max_y);
    }
}

/// Compute the L2 norm of a 2D gradient vector (grad_x, grad_y).
pub(crate) fn gradient_norm(grad_x: &[f64], grad_y: &[f64]) -> f64 {
    grad_x
        .iter()
        .chain(grad_y.iter())
        .map(|g| g * g)
        .sum::<f64>()
        .sqrt()
}

#[allow(dead_code)]
/// Minimum step size for Lipschitz-based step size estimation.
const LIPSCHITZ_STEP_MIN: f64 = 1e-2;
#[allow(dead_code)]
/// Maximum step size for Lipschitz-based step size estimation.
const LIPSCHITZ_STEP_MAX: f64 = 0.2;

/// Shared state for the FISTA/Nesterov optimization loop.
///
/// Encapsulates Lipschitz step size estimation, previous gradient tracking,
/// and best-position snapshot for divergence recovery.
pub(crate) struct NesterovLoopState {
    /// Previous gradient (for Lipschitz step estimation).
    pub prev_grad_x: Vec<f64>,
    /// Previous gradient (for Lipschitz step estimation).
    pub prev_grad_y: Vec<f64>,
    /// Best metric seen during legalization.
    pub best_metric: f64,
    /// Iteration at which best_metric was last updated.
    pub best_iter: usize,
    /// Cell x positions at the best metric.
    pub best_positions_x: Vec<f64>,
    /// Cell y positions at the best metric.
    pub best_positions_y: Vec<f64>,
}

impl NesterovLoopState {
    /// Create a new loop state for `n` movable cells with initial positions.
    pub fn new(initial_x: &[f64], initial_y: &[f64]) -> Self {
        Self {
            prev_grad_x: vec![0.0; initial_x.len()],
            prev_grad_y: vec![0.0; initial_y.len()],
            best_metric: f64::INFINITY,
            best_iter: 0,
            best_positions_x: initial_x.to_vec(),
            best_positions_y: initial_y.to_vec(),
        }
    }

    /// Update Lipschitz-based step sizes from consecutive gradients.
    #[allow(dead_code)]
    pub fn update_step_sizes(
        &mut self,
        nesterov_x: &mut crate::solver::NesterovSolver,
        nesterov_y: &mut crate::solver::NesterovSolver,
        grad_x: &[f64],
        grad_y: &[f64],
    ) {
        let lip_x = nesterov_x.lipschitz_step_size(&self.prev_grad_x, grad_x);
        let lip_y = nesterov_y.lipschitz_step_size(&self.prev_grad_y, grad_y);
        nesterov_x.set_step_size(lip_x.clamp(LIPSCHITZ_STEP_MIN, LIPSCHITZ_STEP_MAX));
        nesterov_y.set_step_size(lip_y.clamp(LIPSCHITZ_STEP_MIN, LIPSCHITZ_STEP_MAX));
    }

    /// Save current gradients for the next iteration's Lipschitz estimate.
    pub fn save_gradients(&mut self, grad_x: &[f64], grad_y: &[f64]) {
        self.prev_grad_x.copy_from_slice(grad_x);
        self.prev_grad_y.copy_from_slice(grad_y);
    }

    /// Record a legalization result. Returns true if this is a new best.
    pub fn record_metric(
        &mut self,
        metric: f64,
        cell_x: &[f64],
        cell_y: &[f64],
        iter: usize,
    ) -> bool {
        if metric < self.best_metric {
            self.best_metric = metric;
            self.best_iter = iter;
            self.best_positions_x.copy_from_slice(cell_x);
            self.best_positions_y.copy_from_slice(cell_y);
            true
        } else {
            false
        }
    }
}

/// Lock all placed cells NOT in the given set, run a closure, then restore strengths.
///
/// Used by incremental placement: cells outside the target set are temporarily
/// locked as `Fixed` so the placer treats them as immovable.
pub(crate) fn with_locked_others<F>(
    ctx: &mut Context,
    target_cells: &FxHashSet<CellId>,
    f: F,
) -> Result<(), PlacerError>
where
    F: FnOnce(&mut Context) -> Result<(), PlacerError>,
{
    let mut restore_list: Vec<(CellId, PlaceStrength)> = Vec::new();
    for (ci, cell) in ctx.design.iter_alive_cells() {
        if !target_cells.contains(&ci) && cell.bel.is_some() && !cell.bel_strength.is_locked() {
            restore_list.push((ci, cell.bel_strength));
        }
    }

    for &(ci, _) in &restore_list {
        let bel = ctx.design.cell(ci).bel;
        ctx.design.cell_edit(ci).set_bel(bel, PlaceStrength::Fixed);
    }

    let result = f(ctx);

    for (ci, original_strength) in restore_list {
        let bel = ctx.design.cell(ci).bel;
        ctx.design.cell_edit(ci).set_bel(bel, original_strength);
    }

    result
}

// ---------------------------------------------------------------------------
// Type-aware tile placement
// ---------------------------------------------------------------------------

/// Pre-computed per-cell-type valid tile positions and BEL capacities.
///
/// Shared across placers. Enables type-aware snapping (cells move only to tiles
/// with compatible BELs) and per-type overflow detection for density control.
pub struct TypeAwarePlacement {
    /// For each cell type (resolved bucket): sorted list of valid x-coordinates.
    pub valid_xs: FxHashMap<IdString, Vec<f64>>,
    /// For each cell type: sorted list of valid y-coordinates per x-column.
    pub valid_ys: FxHashMap<IdString, FxHashMap<i32, Vec<f64>>>,
    /// Per-tile capacity for each cell type: (vx, vy) → n_compatible_bels.
    pub tile_capacity: FxHashMap<IdString, FxHashMap<(i32, i32), u32>>,
    /// Per-tile pin capacity for each cell type: (vx, vy) → sum of compatible BEL pins.
    pub tile_pin_capacity: FxHashMap<IdString, FxHashMap<(i32, i32), u32>>,
    /// Total per-tile pin capacity across all active placement buckets.
    pub total_tile_pin_capacity: FxHashMap<(i32, i32), u32>,
}

impl TypeAwarePlacement {
    /// Build from the chip database. `x0`/`y0` is the grid origin offset
    /// (physical tile coords → virtual grid coords: `vx = loc.x - x0`).
    pub fn build(ctx: &Context, x0: i32, y0: i32) -> Self {
        let mut cell_types_present: FxHashSet<IdString> = FxHashSet::default();
        for (_cell_id, cell) in ctx.design.iter_alive_cells() {
            if cell.bel_strength.is_locked() {
                continue;
            }
            let bucket = ctx.resolve_bucket(cell.cell_type);
            cell_types_present.insert(bucket);
        }

        let mut valid_xs: FxHashMap<IdString, Vec<f64>> = FxHashMap::default();
        let mut valid_ys: FxHashMap<IdString, FxHashMap<i32, Vec<f64>>> = FxHashMap::default();
        let mut tile_capacity: FxHashMap<IdString, FxHashMap<(i32, i32), u32>> =
            FxHashMap::default();
        let mut tile_pin_capacity: FxHashMap<IdString, FxHashMap<(i32, i32), u32>> =
            FxHashMap::default();
        let mut total_tile_pin_capacity: FxHashMap<(i32, i32), u32> = FxHashMap::default();

        for &ct in &cell_types_present {
            let mut xs_set: FxHashSet<i32> = FxHashSet::default();
            let mut ys_per_x: FxHashMap<i32, FxHashSet<i32>> = FxHashMap::default();
            let cap_map = tile_capacity.entry(ct).or_default();
            let pin_cap_map = tile_pin_capacity.entry(ct).or_default();

            for bel in ctx.bels_for_bucket(ct) {
                let loc = bel.loc();
                let vx = loc.x - x0;
                let vy = loc.y - y0;
                xs_set.insert(vx);
                ys_per_x.entry(vx).or_default().insert(vy);
                *cap_map.entry((vx, vy)).or_insert(0) += 1;
                let bel_pin_count = ctx.chipdb().bel_info(bel.id()).pins.get().len() as u32;
                *pin_cap_map.entry((vx, vy)).or_insert(0) += bel_pin_count;
                *total_tile_pin_capacity.entry((vx, vy)).or_insert(0) += bel_pin_count;
            }

            let mut xs: Vec<f64> = xs_set.into_iter().map(|x| x as f64).collect();
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            valid_xs.insert(ct, xs);

            let mut ys_map: FxHashMap<i32, Vec<f64>> = FxHashMap::default();
            for (x, ys_set) in ys_per_x {
                let mut ys: Vec<f64> = ys_set.into_iter().map(|y| y as f64).collect();
                ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ys_map.insert(x, ys);
            }
            valid_ys.insert(ct, ys_map);
        }

        let n_types = cell_types_present.len();
        for ct in &cell_types_present {
            let n_pos = valid_xs.get(ct).map(|v| v.len()).unwrap_or(0);
            let n_bels: u32 = tile_capacity.get(ct).map(|m| m.values().sum()).unwrap_or(0);
            eprintln!(
                "  type {}: {} valid columns, {} total BELs",
                ctx.name_of(*ct),
                n_pos,
                n_bels,
            );
        }
        eprintln!("TypeAwarePlacement: {} cell types", n_types);

        Self {
            valid_xs,
            valid_ys,
            tile_capacity,
            tile_pin_capacity,
            total_tile_pin_capacity,
        }
    }

    /// Snap x to nearest valid column for this cell type.
    pub fn snap_x(&self, bucket: IdString, x: f64) -> f64 {
        let xs = match self.valid_xs.get(&bucket) {
            Some(xs) if !xs.is_empty() => xs,
            _ => return x,
        };
        let idx = xs.partition_point(|&c| c < x);
        if idx == 0 {
            xs[0]
        } else if idx >= xs.len() {
            *xs.last().unwrap()
        } else {
            let left = xs[idx - 1];
            let right = xs[idx];
            if (x - left) <= (right - x) {
                left
            } else {
                right
            }
        }
    }

    /// Snap y to nearest valid row for this cell type at the given x column.
    pub fn snap_y(&self, bucket: IdString, snapped_x: i32, y: f64) -> f64 {
        let ys = match self.valid_ys.get(&bucket) {
            Some(ys_map) => match ys_map.get(&snapped_x) {
                Some(ys) if !ys.is_empty() => ys,
                _ => return y,
            },
            _ => return y,
        };
        let idx = ys.partition_point(|&c| c < y);
        if idx == 0 {
            ys[0]
        } else if idx >= ys.len() {
            *ys.last().unwrap()
        } else {
            let left = ys[idx - 1];
            let right = ys[idx];
            if (y - left) <= (right - y) {
                left
            } else {
                right
            }
        }
    }

    /// Compute per-type tile overflow statistics.
    ///
    /// For each cell type, counts cells at each tile and divides by compatible
    /// BEL capacity. Returns:
    /// - `max_overflow_ratio`: worst tile occupancy / capacity ratio
    /// - `n_tiles_over_capacity`: count of overflowing tiles
    /// - `overflow_excess`: sum of overflow ratios above capacity
    pub fn compute_overflow(
        &self,
        cell_buckets: &[IdString],
        cell_weights: &[f64],
        cell_x: &[f64],
        cell_y: &[f64],
        grid_w: usize,
        grid_h: usize,
    ) -> (f64, usize, f64) {
        let mut max_overflow = 0.0f64;
        let mut n_over = 0usize;
        let mut overflow_excess = 0.0f64;

        // Count pin demand per type per tile using bilinear scatter so the
        // occupancy field moves continuously with pin positions.
        let mut type_tile_count: FxHashMap<IdString, FxHashMap<(i32, i32), f64>> =
            FxHashMap::default();
        for (i, &bucket) in cell_buckets.iter().enumerate() {
            let weight = cell_weights.get(i).copied().unwrap_or(0.0);
            scatter_bilinear_tile(
                type_tile_count.entry(bucket).or_default(),
                cell_x[i],
                cell_y[i],
                weight,
                grid_w,
                grid_h,
            );
        }

        for (bucket, tile_counts) in &type_tile_count {
            let cap_map = match self.tile_pin_capacity.get(bucket) {
                Some(m) => m,
                None => continue,
            };
            for (&(tx, ty), &count) in tile_counts {
                let cap = cap_map.get(&(tx, ty)).copied().unwrap_or(0);
                if cap == 0 {
                    // Zero-capacity tile (wrong type): skip. Cells here will
                    // be snapped to valid tiles during legalization. Counting
                    // them as overflow produces misleading metrics.
                    continue;
                }
                let ratio = count / cap as f64;
                max_overflow = max_overflow.max(ratio);
                if count > cap as f64 {
                    n_over += 1;
                    overflow_excess += ratio - 1.0;
                }
            }
        }

        (max_overflow, n_over, overflow_excess)
    }

    /// Compute unified pin-demand occupancy ratio per tile:
    /// demand / available pin capacity.
    pub fn compute_pin_utilization_map(
        &self,
        cell_weights: &[f64],
        cell_x: &[f64],
        cell_y: &[f64],
        grid_w: usize,
        grid_h: usize,
    ) -> FxHashMap<(i32, i32), f64> {
        let mut demand_per_tile: FxHashMap<(i32, i32), f64> = FxHashMap::default();
        for i in 0..cell_weights.len() {
            scatter_bilinear_tile(
                &mut demand_per_tile,
                cell_x[i],
                cell_y[i],
                cell_weights[i],
                grid_w,
                grid_h,
            );
        }

        let mut util = FxHashMap::default();
        for (tile, demand) in demand_per_tile {
            let cap = self
                .total_tile_pin_capacity
                .get(&tile)
                .copied()
                .unwrap_or(0) as f64;
            let ratio = if cap > 0.0 { demand / cap } else { demand };
            util.insert(tile, ratio);
        }
        util
    }

    /// Compute the coarse-grid unified overflow hotspot field.
    ///
    /// For each coarse tile, this returns the maximum physical-tile
    /// bucket-specific pin-demand / pin-capacity ratio inside that coarse tile.
    /// This preserves local hotspots instead of averaging them away.
    pub fn compute_overflow_map_coarse(
        &self,
        cell_buckets: &[IdString],
        cell_weights: &[f64],
        cell_x: &[f64],
        cell_y: &[f64],
        coarsen: usize,
        grid_w: usize,
        grid_h: usize,
    ) -> FxHashMap<(i32, i32), f64> {
        let coarsen = coarsen.max(1);
        let coarse_w = grid_w.div_ceil(coarsen) as i32;
        let coarse_h = grid_h.div_ceil(coarsen) as i32;

        let mut demand_per_bucket_tile: FxHashMap<IdString, FxHashMap<(i32, i32), f64>> =
            FxHashMap::default();
        for (i, &bucket) in cell_buckets.iter().enumerate() {
            scatter_bilinear_tile(
                demand_per_bucket_tile.entry(bucket).or_default(),
                cell_x[i],
                cell_y[i],
                cell_weights[i],
                grid_w,
                grid_h,
            );
        }

        let mut util = FxHashMap::default();
        let mut coarse_accum: FxHashMap<(i32, i32), (f64, f64)> = FxHashMap::default();
        for (bucket, tile_demands) in demand_per_bucket_tile {
            let Some(cap_map) = self.tile_pin_capacity.get(&bucket) else {
                continue;
            };
            for ((tx, ty), demand) in tile_demands {
                let cap = cap_map.get(&(tx, ty)).copied().unwrap_or(0) as f64;
                let ratio = if cap > 0.0 { demand / cap } else { demand };
                let cx = (tx / coarsen as i32).clamp(0, coarse_w - 1);
                let cy = (ty / coarsen as i32).clamp(0, coarse_h - 1);
                let entry = coarse_accum.entry((cx, cy)).or_insert((0.0, 0.0));
                entry.0 += ratio * ratio;
                entry.1 += 1.0;
            }
        }

        for (coord, (sum_sq, count)) in coarse_accum {
            util.insert(coord, (sum_sq / count.max(1.0)).sqrt());
        }

        util
    }

    /// Normalize coarse occupancy so coarse levels activate congestion on a
    /// comparable scale to finer levels.
    ///
    /// The continuous scatter + coarse aggregation path dilutes L0 occupancy by
    /// roughly the linear coarsening factor. Re-inflate by `coarsen` so a
    /// meaningfully full coarse region produces a comparable congestion signal.
    pub fn normalize_overflow_map_coarse(
        field: &FxHashMap<(i32, i32), f64>,
        coarsen: usize,
    ) -> FxHashMap<(i32, i32), f64> {
        let scale = coarsen.max(1) as f64;
        field.iter().map(|(&k, &v)| (k, v * scale)).collect()
    }

    /// Direct destructive energy from the coarse overflow field.
    ///
    /// This uses the same unified pin-demand / pin-capacity ratio as the
    /// overflow metric. We charge occupancy directly, rather than only the
    /// amount above 1.0, so the destructive term remains visible before bins
    /// are formally overfull.
    pub fn overflow_energy_from_map(field: &FxHashMap<(i32, i32), f64>) -> f64 {
        field.values().map(|&u| u * u).sum()
    }
}

// ---------------------------------------------------------------------------
// Cell-validity mask: O(1) "is cell `ci` allowed at `(gx, gy)`?" lookup
// ---------------------------------------------------------------------------

/// Bit-packed per-cell validity overlay on top of `TypeAwarePlacement`.
///
/// Every placer needs to check whether a given cell can legally land at a
/// specific grid position before evaluating its cost. This struct answers that
/// in O(1). It is a static pre-computation derived from `TypeAwarePlacement`:
/// `is_valid(ci, gx, gy)` is true iff `(gx, gy)` is a valid position for the
/// bucket of cell `cell_ids[ci]`.
///
/// Memory cost: `n_cells * width * height` bits, e.g. ~4 MB for 300 cells on a
/// 350×350 grid.
pub struct CellValidityMask {
    width: i32,
    height: i32,
    n_cells: usize,
    // bits[idx] bit `b`: (cell_idx * W*H + gy*W + gx) where idx = bit >> 6,
    // b = bit & 63.
    bits: Vec<u64>,
}

impl CellValidityMask {
    /// Build from an already-constructed `TypeAwarePlacement`. For each cell i,
    /// resolve its bucket, then mark every (gx, gy) in
    /// `valid_xs[bucket] × valid_ys[bucket][gx]` as allowed.
    ///
    /// `cell_ids[i]` must correspond to the cell referred to by index `i` in
    /// the placer's cell arrays (e.g. `idx_to_cell`). Fixed cells should still
    /// appear in `cell_ids`; the mask will simply encode their single allowed
    /// position via `TypeAwarePlacement` (or mark nothing if they are not in
    /// the type-aware tables — they are never queried).
    pub fn build(
        ctx: &Context,
        cell_ids: &[CellId],
        type_aware: &TypeAwarePlacement,
        width: i32,
        height: i32,
    ) -> Self {
        let n_cells = cell_ids.len();
        let bits_per_cell = (width as usize) * (height as usize);
        let total_bits = n_cells * bits_per_cell;
        let n_words = (total_bits + 63) / 64;
        let mut bits = vec![0u64; n_words];
        let stride = bits_per_cell;

        let mut n_unmapped = 0usize;
        let mut per_cell_counts = Vec::with_capacity(n_cells);
        for (ci, &cell_id) in cell_ids.iter().enumerate() {
            let bucket = ctx.resolve_bucket(ctx.design.cell(cell_id).cell_type);
            let xs = match type_aware.valid_xs.get(&bucket) {
                Some(xs) if !xs.is_empty() => xs,
                _ => {
                    n_unmapped += 1;
                    per_cell_counts.push(0usize);
                    continue; // no valid positions (e.g. fixed/non-placeable); leave all bits 0
                }
            };
            let ys_map = match type_aware.valid_ys.get(&bucket) {
                Some(m) => m,
                None => continue,
            };

            let base = (ci * stride) as u64;
            let mut set_count = 0usize;
            for &xf in xs {
                let gx = xf.round() as i32;
                if gx < 0 || gx >= width {
                    continue;
                }
                let Some(ys) = ys_map.get(&gx) else { continue };
                for &yf in ys {
                    let gy = yf.round() as i32;
                    if gy < 0 || gy >= height {
                        continue;
                    }
                    let bit = base + (gy as u64) * (width as u64) + (gx as u64);
                    let word = (bit >> 6) as usize;
                    let shift = (bit & 63) as u32;
                    bits[word] |= 1u64 << shift;
                    set_count += 1;
                }
            }
            per_cell_counts.push(set_count);
        }

        let min_set = per_cell_counts.iter().copied().min().unwrap_or(0);
        let max_set = per_cell_counts.iter().copied().max().unwrap_or(0);
        let avg_set: f64 = per_cell_counts.iter().copied().sum::<usize>() as f64 / n_cells.max(1) as f64;
        eprintln!(
            "CellValidityMask: {} cells, {}x{} grid, per-cell valid positions: min={} avg={:.1} max={}, unmapped cells={}",
            n_cells, width, height, min_set, avg_set, max_set, n_unmapped,
        );

        Self {
            width,
            height,
            n_cells,
            bits,
        }
    }

    /// O(1) check. Returns false for out-of-bounds or invalid-for-cell positions.
    #[inline(always)]
    pub fn is_valid(&self, cell_idx: usize, gx: i32, gy: i32) -> bool {
        if cell_idx >= self.n_cells {
            return false;
        }
        if gx < 0 || gy < 0 || gx >= self.width || gy >= self.height {
            return false;
        }
        let stride = (self.width as u64) * (self.height as u64);
        let bit = (cell_idx as u64) * stride
            + (gy as u64) * (self.width as u64)
            + (gx as u64);
        let word = self.bits[(bit >> 6) as usize];
        ((word >> (bit & 63)) & 1) == 1
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
}
