//! Shared helper functions used by multiple placer implementations.

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::CellId;
use rustc_hash::{FxHashMap, FxHashSet};

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
            let region_bels = ctx.bels_for_bucket_in_region(cell_type, *region_idx).to_vec();
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
            let bucket_bels: Vec<_> =
                ctx.bels_for_bucket(cell_type).map(|bel| bel.id()).collect();
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

/// Compute LSE wirelength gradient for all nets, accumulating into grad_x/grad_y.
///
/// Uses the `cell_to_idx` map to determine which pins are movable solver variables.
/// Fixed pins use their current BEL position.
pub(crate) fn add_wirelength_gradient(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    gamma: f64,
    grad_x: &mut [f64],
    grad_y: &mut [f64],
) {
    use super::solver::lse;

    for (_, net) in ctx.design.iter_alive_nets() {
        let mut pin_positions: Vec<(f64, f64)> = Vec::new();
        let mut pin_indices: Vec<usize> = Vec::new();

        if let Some(driver_pin) = net.driver() {
            collect_pin_position(
                ctx, cell_to_idx, cell_x, cell_y,
                driver_pin.cell, &mut pin_positions, &mut pin_indices,
            );
        }

        for user in net.users().iter() {
            collect_pin_position(
                ctx, cell_to_idx, cell_x, cell_y,
                user.cell, &mut pin_positions, &mut pin_indices,
            );
        }

        if pin_positions.len() < 2 {
            continue;
        }

        let mut net_grad = vec![(0.0, 0.0); pin_positions.len()];
        lse::lse_gradient(&pin_positions, gamma, &mut net_grad);

        for (k, &solver_idx) in pin_indices.iter().enumerate() {
            if solver_idx != usize::MAX {
                grad_x[solver_idx] += net_grad[k].0;
                grad_y[solver_idx] += net_grad[k].1;
            }
        }
    }
}

/// Collect position of a single pin for gradient computation.
fn collect_pin_position(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    cell_id: CellId,
    positions: &mut Vec<(f64, f64)>,
    indices: &mut Vec<usize>,
) {
    if let Some(&idx) = cell_to_idx.get(&cell_id) {
        positions.push((cell_x[idx], cell_y[idx]));
        indices.push(idx);
    } else {
        let cell = ctx.design.cell(cell_id);
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            positions.push((loc.x as f64, loc.y as f64));
            indices.push(usize::MAX);
        }
    }
}

/// Unbind all movable cells and their cluster children.
pub(crate) fn unbind_movable_cells(ctx: &mut Context, idx_to_cell: &[CellId]) {
    for &cell_id in idx_to_cell {
        let cell = ctx.design.cell(cell_id);
        if let Some(bel) = cell.bel {
            if !cell.bel_strength.is_locked() {
                ctx.unbind_bel(bel);
            }
        }
        if let Some(cluster) = ctx.design.clusters.get(&cell_id) {
            let children: Vec<_> = cluster.constr_children.clone();
            for child_id in children {
                let child = ctx.design.cell(child_id);
                if let Some(bel) = child.bel {
                    if !child.bel_strength.is_locked() {
                        ctx.unbind_bel(bel);
                    }
                }
            }
        }
    }
}

/// Place cluster children relative to the root BEL location.
///
/// Tries exact constraint position first, then any available BEL of matching type.
pub(crate) fn place_cluster_children(
    ctx: &mut Context,
    cell_id: CellId,
    root_bel: BelId,
) -> Result<(), PlacerError> {
    let cluster = match ctx.design.clusters.get(&cell_id) {
        Some(c) => c,
        None => return Ok(()),
    };
    let children: Vec<_> = cluster.constr_children.clone();
    let root_loc = ctx.bel(root_bel).loc();

    for child_id in children {
        let child = ctx.design.cell(child_id);
        let child_type = child.cell_type;
        let child_x = root_loc.x + child.constr_x;
        let child_y = root_loc.y + child.constr_y;

        let mut placed = false;

        let exact_candidates: Vec<_> = ctx
            .bels_for_bucket(child_type)
            .filter(|b| b.is_available() && b.loc().x == child_x && b.loc().y == child_y)
            .map(|b| b.id())
            .collect();
        for bel_id in exact_candidates {
            if ctx.bind_bel(bel_id, child_id, PlaceStrength::Placer) {
                placed = true;
                break;
            }
        }

        if !placed {
            let fallback_candidates: Vec<_> = ctx
                .bels_for_bucket(child_type)
                .filter(|b| b.is_available())
                .map(|b| b.id())
                .collect();
            for bel_id in fallback_candidates {
                if ctx.bind_bel(bel_id, child_id, PlaceStrength::Placer) {
                    placed = true;
                    break;
                }
            }
        }

        if !placed {
            return Err(PlacerError::PlacementFailed(format!(
                "Failed to place cluster child {}",
                ctx.name_of(ctx.design.cell(child_id).name)
            )));
        }
    }

    Ok(())
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

/// Apply WA preconditioner: `grad[i] /= max(1.0, pin_weight[i] + alpha * penalty_weight)`.
///
/// This normalizes gradients so highly-connected cells receive smaller update steps,
/// preventing them from oscillating and causing placement divergence. The `alpha`
/// parameter controls how much the density penalty term contributes to preconditioning.
///
/// In our FPGA model all cells occupy 1 BEL (area = 1), so the cell area term from
/// the original ePlace formulation is folded into `penalty_weight` directly.
pub(crate) fn apply_preconditioner(
    grad_x: &mut [f64],
    grad_y: &mut [f64],
    pin_weights: &[f64],
    alpha: f64,
    penalty_weight: f64,
) {
    for i in 0..grad_x.len() {
        let precond = (pin_weights[i] + alpha * penalty_weight).max(1.0);
        grad_x[i] /= precond;
        grad_y[i] /= precond;
    }
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

/// Minimum step size for Lipschitz-based step size estimation.
const LIPSCHITZ_STEP_MIN: f64 = 1e-4;
/// Maximum step size for Lipschitz-based step size estimation.
const LIPSCHITZ_STEP_MAX: f64 = 1.0;
/// Maximum preconditioner alpha value.
const PRECOND_ALPHA_MAX: f64 = 1024.0;
/// Preconditioner alpha doubling interval (iterations).
const PRECOND_DOUBLE_INTERVAL: usize = 20;

/// Shared state for the FISTA/Nesterov optimization loop.
///
/// Encapsulates the pattern common to both ElectroPlace and Hydraulic placers:
/// Lipschitz step size estimation, WA preconditioner alpha scaling, previous
/// gradient tracking, and best-position snapshot for divergence recovery.
pub(crate) struct NesterovLoopState {
    /// Previous gradient (for Lipschitz step estimation).
    pub prev_grad_x: Vec<f64>,
    /// Previous gradient (for Lipschitz step estimation).
    pub prev_grad_y: Vec<f64>,
    /// WA preconditioner scaling factor (doubles periodically).
    pub precond_alpha: f64,
    /// Best HPWL seen during legalization.
    pub best_hpwl: f64,
    /// Cell x positions at the best HPWL.
    pub best_positions_x: Vec<f64>,
    /// Cell y positions at the best HPWL.
    pub best_positions_y: Vec<f64>,
}

impl NesterovLoopState {
    /// Create a new loop state for `n` movable cells with initial positions.
    pub fn new(initial_x: &[f64], initial_y: &[f64]) -> Self {
        let n = initial_x.len();
        Self {
            prev_grad_x: vec![0.0; n],
            prev_grad_y: vec![0.0; n],
            precond_alpha: 1.0,
            best_hpwl: f64::INFINITY,
            best_positions_x: initial_x.to_vec(),
            best_positions_y: initial_y.to_vec(),
        }
    }

    /// Update Lipschitz-based step sizes from consecutive gradients.
    ///
    /// Should be called after iteration 0 (needs a previous gradient).
    pub fn update_step_sizes(
        &mut self,
        nesterov_x: &mut super::solver::NesterovSolver,
        nesterov_y: &mut super::solver::NesterovSolver,
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

    /// Double preconditioner alpha if conditions are met (low overflow, periodic).
    pub fn maybe_increase_precond_alpha(&mut self, overflow_metric: f64, threshold: f64, iter: usize) {
        if overflow_metric < threshold && iter > 0 && iter % PRECOND_DOUBLE_INTERVAL == 0 {
            self.precond_alpha = (self.precond_alpha * 2.0).min(PRECOND_ALPHA_MAX);
        }
    }

    /// Record a legalization result. Returns true if this is a new best.
    pub fn record_hpwl(&mut self, hpwl: f64, cell_x: &[f64], cell_y: &[f64]) -> bool {
        if hpwl < self.best_hpwl {
            self.best_hpwl = hpwl;
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
