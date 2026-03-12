//! Hydraulic placer state: cell positions, network, solver state.

use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::CellId;

use super::config::HydraulicPlacerCfg;
use super::network::PipeNetwork;
use crate::placer::solver::NesterovSolver;

pub struct HydraulicState {
    pub cell_x: Vec<f64>,
    pub cell_y: Vec<f64>,
    pub cell_to_idx: FxHashMap<CellId, usize>,
    pub idx_to_cell: Vec<CellId>,
    pub network: PipeNetwork,
    pub nesterov_x: NesterovSolver,
    pub nesterov_y: NesterovSolver,
}

impl HydraulicState {
    pub fn new(ctx: &Context, cfg: &HydraulicPlacerCfg) -> Self {
        let (cell_to_idx, idx_to_cell) = crate::placer::common::collect_movable_cells(ctx);
        let n = idx_to_cell.len();

        let mut cell_x = vec![0.0; n];
        let mut cell_y = vec![0.0; n];
        crate::placer::common::init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);

        let network = PipeNetwork::from_context(ctx);

        let mut nesterov_x = NesterovSolver::new(n, cfg.nesterov_step_size);
        let mut nesterov_y = NesterovSolver::new(n, cfg.nesterov_step_size);
        nesterov_x.set_positions(&cell_x);
        nesterov_y.set_positions(&cell_y);

        Self {
            cell_x,
            cell_y,
            cell_to_idx,
            idx_to_cell,
            network,
            nesterov_x,
            nesterov_y,
        }
    }

    pub fn num_cells(&self) -> usize {
        self.idx_to_cell.len()
    }

    fn cell_tile(&self, i: usize) -> (i32, i32) {
        let tx = (self.cell_x[i].round() as i32).clamp(0, self.network.width - 1);
        let ty = (self.cell_y[i].round() as i32).clamp(0, self.network.height - 1);
        (tx, ty)
    }

    fn pin_tile(&self, ctx: &Context, cell_id: CellId) -> (i32, i32) {
        if let Some(&idx) = self.cell_to_idx.get(&cell_id) {
            self.cell_tile(idx)
        } else {
            let cell = ctx.design.cell(cell_id);
            if let Some(bel) = cell.bel {
                let loc = ctx.bel(bel).loc();
                (loc.x.clamp(0, self.network.width - 1), loc.y.clamp(0, self.network.height - 1))
            } else {
                (self.network.width / 2, self.network.height / 2)
            }
        }
    }

    /// Build demand vector from net connectivity (Kirchhoff model).
    ///
    /// Each net driver injects +1 unit of fluid at its tile junction.
    /// Each net sink extracts 1/fanout at its tile junction.
    /// This ensures demand is balanced (sum = 0) for each net.
    pub fn compute_net_demands(&self, ctx: &Context) -> Vec<f64> {
        let n_j = self.network.num_junctions();
        let mut demand = vec![0.0; n_j];

        for (_, net) in ctx.design.iter_alive_nets() {
            let driver = net.driver();
            let users = net.users();
            if users.is_empty() {
                continue;
            }
            let Some(dp) = driver else { continue };

            let (dx, dy) = self.pin_tile(ctx, dp.cell);
            let driver_jidx = self.network.junction_index(dx, dy);
            demand[driver_jidx] += 1.0;

            let sink_weight = 1.0 / users.len() as f64;
            for user in users {
                if !user.is_valid() {
                    continue;
                }
                let (sx, sy) = self.pin_tile(ctx, user.cell);
                let sink_jidx = self.network.junction_index(sx, sy);
                demand[sink_jidx] -= sink_weight;
            }
        }

        demand
    }

    /// Compute Kirchhoff pin weights for gradient normalization.
    ///
    /// For each movable cell, sums the pin connectivity:
    /// driver pins contribute +1.0 per net, sink pins contribute 1.0/fanout per net.
    pub fn compute_kirchhoff_pin_weights(&self, ctx: &Context) -> Vec<f64> {
        let n = self.num_cells();
        let mut weights = vec![0.0; n];

        for (_, net) in ctx.design.iter_alive_nets() {
            let driver = net.driver();
            let users = net.users();
            if users.is_empty() {
                continue;
            }
            let Some(dp) = driver else { continue };

            if let Some(&idx) = self.cell_to_idx.get(&dp.cell) {
                weights[idx] += 1.0;
            }

            let sink_weight = 1.0 / users.len() as f64;
            for user in users {
                if !user.is_valid() {
                    continue;
                }
                if let Some(&idx) = self.cell_to_idx.get(&user.cell) {
                    weights[idx] += sink_weight;
                }
            }
        }

        weights
    }

    /// Unified pressure gradient: the force on each cell from the solved pressure field.
    ///
    /// Force = -grad(P) at each cell's tile, computed via central finite differences.
    /// This IS the wirelength + congestion force in one physical quantity.
    pub fn compute_pressure_gradient(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.num_cells();
        let mut grad_x = vec![0.0; n];
        let mut grad_y = vec![0.0; n];

        let w = self.network.width;
        let h = self.network.height;

        for i in 0..n {
            let (tx, ty) = self.cell_tile(i);
            let center = self.pressure_at(tx, ty);

            let east = if tx + 1 < w { self.pressure_at(tx + 1, ty) } else { center };
            let west = if tx > 0 { self.pressure_at(tx - 1, ty) } else { center };
            let north = if ty + 1 < h { self.pressure_at(tx, ty + 1) } else { center };
            let south = if ty > 0 { self.pressure_at(tx, ty - 1) } else { center };

            // Central differences: dP/dx ~ (P_east - P_west) / 2
            grad_x[i] = (east - west) / 2.0;
            grad_y[i] = (north - south) / 2.0;
        }

        (grad_x, grad_y)
    }

    #[inline]
    fn pressure_at(&self, x: i32, y: i32) -> f64 {
        self.network.junctions[self.network.junction_index(x, y)].pressure
    }

    pub fn sync_to_nesterov(&mut self) {
        self.nesterov_x.set_positions(&self.cell_x);
        self.nesterov_y.set_positions(&self.cell_y);
    }

    pub fn sync_from_nesterov(&mut self) {
        self.cell_x.copy_from_slice(self.nesterov_x.positions());
        self.cell_y.copy_from_slice(self.nesterov_y.positions());
    }

    pub fn clamp_positions(&mut self) {
        let max_x = (self.network.width - 1) as f64;
        let max_y = (self.network.height - 1) as f64;
        crate::placer::common::clamp_positions(&mut self.cell_x, &mut self.cell_y, max_x, max_y);
    }
}
