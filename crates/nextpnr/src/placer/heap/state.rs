use crate::context::Context;
use crate::netlist::CellId;
use rustc_hash::FxHashMap;

use super::config::PlacerHeapCfg;
use crate::placer::PlacerError;

/// Internal state for the HeAP algorithm.
pub struct HeapState {
    pub(super) cfg: PlacerHeapCfg,
    /// Movable cells (alive, not locked).
    pub(super) movable_cells: Vec<CellId>,
    /// Map from CellIdx to index in movable_cells.
    pub(super) cell_to_idx: FxHashMap<CellId, usize>,
    /// Current X positions (continuous).
    #[cfg(not(feature = "test-utils"))]
    pub(super) cell_x: Vec<f64>,
    #[cfg(feature = "test-utils")]
    pub cell_x: Vec<f64>,
    /// Current Y positions (continuous).
    #[cfg(not(feature = "test-utils"))]
    pub(super) cell_y: Vec<f64>,
    #[cfg(feature = "test-utils")]
    pub cell_y: Vec<f64>,
    /// Region constraint for each movable cell (parallel to movable_cells).
    pub(super) cell_region: Vec<Option<u32>>,
    /// Current spreading force multiplier.
    pub(super) alpha: f64,
    /// Grid dimensions.
    pub(super) grid_w: i32,
    pub(super) grid_h: i32,
    /// Congestion-aware displacement targets (target_x, target_y, force_weight).
    pub(super) congestion_targets: Option<Vec<(f64, f64, f64)>>,
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

    /// Sync analytical positions from the current BEL placements.
    ///
    /// Call this after `PlacerPipeline::prepare_discrete` has placed all cells.
    pub(super) fn sync_positions_from_bels(&mut self, ctx: &Context) {
        for (idx, &cell_idx) in self.movable_cells.iter().enumerate() {
            let cell = ctx.cell(cell_idx);
            if let Some(bel) = cell.bel() {
                let loc = bel.loc();
                self.cell_x[idx] = loc.x as f64;
                self.cell_y[idx] = loc.y as f64;
            }
        }
    }
}
