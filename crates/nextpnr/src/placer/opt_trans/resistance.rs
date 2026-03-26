//! Unified resistance model for the Beckmann optimal transport placer.
//!
//! Resistance sigma encodes ALL physics:
//! - Congestion (Beckmann): increases with flow/capacity ratio
//! - Interference: more nets sharing a pipe = more resistance
//! - Timing: critical nets resist stretching

use super::network::Pipe;

/// Computes effective resistance for a pipe given congestion, interference,
/// and timing state.
pub struct ResistanceModel {
    /// Exponent alpha for congestion: (|J|/C)^alpha.
    pub congestion_exponent: f64,
    /// Weight for flow interference term.
    pub interference_weight: f64,
    /// Weight for timing criticality term.
    pub timing_weight: f64,
}

impl ResistanceModel {
    /// Compute effective resistance for a single pipe.
    ///
    /// R_eff = R_base * R_cong * R_interf * R_timing where:
    /// - R_cong = 1 + (|flow|/capacity)^alpha  (Beckmann congestion)
    /// - R_interf = 1 + w_i * (n_nets - 1) * util^2  (flow interference)
    /// - R_timing = 1 + w_t * criticality  (timing resistance)
    pub fn effective_resistance(&self, pipe: &Pipe, timing_criticality: f64) -> f64 {
        let r_base = pipe.base_resistance;

        // Congestion: increases with utilization ratio
        let util = (pipe.flow.abs() / pipe.capacity.max(1.0)).min(10.0);
        let r_cong = 1.0 + util.powf(self.congestion_exponent);

        // Interference: more nets sharing a pipe = more resistance
        let r_interf = 1.0
            + self.interference_weight
                * (pipe.net_count.max(1) - 1) as f64
                * util
                * util;

        // Timing: critical nets resist stretching
        let r_timing = 1.0 + self.timing_weight * timing_criticality;

        r_base * r_cong * r_interf * r_timing
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
            pipe_type: PipeType::IntraTile,
        }
    }

    #[test]
    fn zero_flow_gives_base_resistance() {
        let model = ResistanceModel {
            congestion_exponent: 2.0,
            interference_weight: 1.0,
            timing_weight: 0.0,
        };
        let pipe = test_pipe(0.0, 10.0, 1);
        let r = model.effective_resistance(&pipe, 0.0);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn congestion_increases_resistance() {
        let model = ResistanceModel {
            congestion_exponent: 2.0,
            interference_weight: 0.0,
            timing_weight: 0.0,
        };
        let pipe_low = test_pipe(1.0, 10.0, 1);
        let pipe_high = test_pipe(9.0, 10.0, 1);
        assert!(model.effective_resistance(&pipe_high, 0.0) > model.effective_resistance(&pipe_low, 0.0));
    }

    #[test]
    fn interference_increases_resistance() {
        let model = ResistanceModel {
            congestion_exponent: 2.0,
            interference_weight: 1.0,
            timing_weight: 0.0,
        };
        let pipe_1net = test_pipe(5.0, 10.0, 1);
        let pipe_5nets = test_pipe(5.0, 10.0, 5);
        assert!(model.effective_resistance(&pipe_5nets, 0.0) > model.effective_resistance(&pipe_1net, 0.0));
    }
}
