//! Timing oracle trait and implementation for `TimingAnalyser`.
//!
//! Decouples timing feedback from placement internals so that different
//! placement algorithms (OptTrans, ElectroPlace, etc.) can share a single
//! timing interface.

use crate::context::Context;
use crate::netlist::{CellId, NetId};

/// Trait for timing feedback used by placement algorithms.
///
/// Decouples timing computation from placement internals, allowing
/// different timing strategies to be swapped in.
pub trait TimingOracle {
    /// Criticality of a net (0.0 = not critical, 1.0 = most critical).
    fn net_criticality(&self, net: NetId) -> f64;

    /// Criticality of a cell (max of its connected net criticalities).
    fn cell_criticality(&self, cell: CellId) -> f64;

    /// Recompute timing after placement changes.
    fn update(&mut self, ctx: &Context);
}

// ---------------------------------------------------------------------------
// Implementation for TimingAnalyser
// ---------------------------------------------------------------------------

use super::analyser::TimingAnalyser;

impl TimingOracle for TimingAnalyser {
    fn net_criticality(&self, net: NetId) -> f64 {
        // Delegate to the internal map (f32) and widen to f64.
        self.net_criticality.get(&net).copied().unwrap_or(0.0) as f64
    }

    fn cell_criticality(&self, cell: CellId) -> f64 {
        // No timing violations means no criticality.
        if self.worst_slack >= 0 {
            return 0.0;
        }

        let neg_ws = -self.worst_slack as f64;
        let mut max_crit: f64 = 0.0;

        // Scan all ports belonging to this cell and compute per-port
        // criticality from arrival/required times (same formula as
        // `port_criticality` in queries.rs).
        for (pin, _) in &self.port_data {
            if pin.cell != cell {
                continue;
            }
            if let (Some(&arr), Some(&req)) =
                (self.arrival_times.get(pin), self.required_times.get(pin))
            {
                let slack = req - arr;
                let crit =
                    (1.0 - ((slack - self.worst_slack) as f64 / neg_ws)).clamp(0.0, 1.0);
                if crit > max_crit {
                    max_crit = crit;
                }
            }
        }

        max_crit
    }

    fn update(&mut self, ctx: &Context) {
        if self.is_valid {
            // Ports and domains already set up — just re-run propagation.
            self.run(ctx);
        } else {
            // First time or invalidated — full setup.
            self.setup_and_run(ctx);
        }
    }
}
