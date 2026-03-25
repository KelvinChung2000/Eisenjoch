//! Pressure gradient and anchor gradient computation.

use crate::context::Context;

use super::density::gaussian_blur_2d;
use super::super::network::Port;
use super::OptTransState;

const ALL_PORTS: [Port; 4] = [Port::North, Port::East, Port::South, Port::West];

impl OptTransState {
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

    /// Steiner anchor gradient: pulls each pin toward the net's routing center.
    ///
    /// For each net, computes the centroid of all pin positions (movable + fixed).
    /// Adds a gradient pulling each movable pin toward the centroid, weighted
    /// by 1/fanout to prevent large nets from dominating.
    ///
    /// This approximates the Steiner junction -- the optimal routing meeting point.
    /// Keeps nets spatially compact without fighting the transport energy.
    pub fn compute_anchor_gradient(
        &self,
        ctx: &Context,
        anchor_weight: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.num_cells();
        let mut grad_x = vec![0.0; n];
        let mut grad_y = vec![0.0; n];

        for (_net_id, net) in ctx.design.iter_alive_nets() {
            let Some(dp) = net.driver() else { continue };
            let users = net.users();
            if users.is_empty() {
                continue;
            }

            // Collect all pin positions for this net.
            let mut pin_xs = Vec::new();
            let mut pin_ys = Vec::new();
            let mut pin_indices = Vec::new(); // solver index, or usize::MAX for fixed

            let (dx, dy) = self.pin_pos(ctx, dp.cell);
            pin_xs.push(dx);
            pin_ys.push(dy);
            pin_indices.push(self.cell_to_idx.get(&dp.cell).copied().unwrap_or(usize::MAX));

            for user in users {
                if !user.is_valid() {
                    continue;
                }
                let (ux, uy) = self.pin_pos(ctx, user.cell);
                pin_xs.push(ux);
                pin_ys.push(uy);
                pin_indices.push(self.cell_to_idx.get(&user.cell).copied().unwrap_or(usize::MAX));
            }

            if pin_xs.len() < 2 {
                continue;
            }

            // Net center: centroid of all pins (movable + fixed).
            let fanout = pin_xs.len() as f64;
            let cx = pin_xs.iter().sum::<f64>() / fanout;
            let cy = pin_ys.iter().sum::<f64>() / fanout;

            // Pull each movable pin toward the center.
            // Weight = anchor_weight / fanout (large nets get weaker per-pin pull).
            let w = anchor_weight / fanout;
            for (k, &idx) in pin_indices.iter().enumerate() {
                if idx != usize::MAX {
                    grad_x[idx] += w * (pin_xs[k] - cx);
                    grad_y[idx] += w * (pin_ys[k] - cy);
                }
            }
        }

        (grad_x, grad_y)
    }
}
