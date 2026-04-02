//! Unified resistance model for the Beckmann optimal transport placer.
//!
//! Resistance sigma encodes the active transport physics:
//! - Congestion: increases with pipe competition (`net_count`)
//! - Timing: critical nets resist stretching

use super::network::Pipe;

/// Computes effective resistance for a pipe given congestion, interference,
/// and timing state.
pub struct ResistanceModel {
    /// Gain applied to the passive net-count congestion response.
    pub congestion_scale: f64,
    /// Net-count is raised to this power before scaling resistance.
    pub congestion_power: f64,
    /// Weight for timing criticality term.
    pub timing_weight: f64,
}

impl ResistanceModel {
    /// Compute effective resistance for a single pipe.
    ///
    /// R_eff = R_base * R_cong * R_timing where:
    /// - R_cong = 1 + scale * net_count^power  (pipe competition congestion)
    /// - R_timing = 1 + w_t * criticality  (timing resistance)
    #[inline(always)]
    pub fn effective_resistance(&self, pipe: &Pipe, timing_criticality: f64) -> f64 {
        let r_base = pipe.base_resistance;

        let count = pipe.net_count.max(1) as f64;
        let r_cong = if self.congestion_scale <= 0.0 {
            1.0
        } else {
            1.0 + self.congestion_scale * count.powf(self.congestion_power.max(1.0))
        };

        // Timing: critical nets resist stretching
        let r_timing = 1.0 + self.timing_weight * timing_criticality;

        r_base * r_cong * r_timing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::PipeType;

    fn test_pipe(flow: f64, capacity: f64, net_count: u32) -> Pipe {
        Pipe {
            from: 0,
            to: 1,
            base_resistance: 1.0,
            capacity,
            flow,
            net_count,
            raw_cell_density: 0.0,
            cell_density: 0.0,
            eff_conductance: 1.0,
            pipe_type: PipeType::IntraTile,
        }
    }

    #[test]
    fn zero_flow_gives_base_resistance() {
        let model = ResistanceModel {
            congestion_scale: 0.01,
            congestion_power: 2.0,
            timing_weight: 0.0,
        };
        let pipe = test_pipe(0.0, 10.0, 1);
        let r = model.effective_resistance(&pipe, 0.0);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn congestion_increases_resistance() {
        let model = ResistanceModel {
            congestion_scale: 0.01,
            congestion_power: 2.0,
            timing_weight: 0.0,
        };
        let mut pipe_low = test_pipe(1.0, 10.0, 1);
        let mut pipe_high = test_pipe(9.0, 10.0, 1);
        pipe_low.net_count = 1;
        pipe_high.net_count = 4;
        assert!(
            model.effective_resistance(&pipe_high, 0.0)
                > model.effective_resistance(&pipe_low, 0.0)
        );
    }

    #[test]
    fn net_count_no_longer_changes_resistance_directly() {
        let model = ResistanceModel {
            congestion_scale: 0.01,
            congestion_power: 2.0,
            timing_weight: 0.0,
        };
        let pipe_1net = test_pipe(5.0, 10.0, 1);
        let pipe_5nets = test_pipe(5.0, 10.0, 5);
        assert_eq!(
            model.effective_resistance(&pipe_5nets, 0.0),
            model.effective_resistance(&pipe_1net, 0.0)
        );
    }
}
