//! BPR congestion model for pipe costs.
//!
//! The effective resistance of a pipe follows:
//!     R_eff = base * (1 + alpha * (usage / capacity)^beta)
//! with a bounded utilization ratio. The cap keeps a few severely overfull
//! template edges from numerically dominating the whole placement objective.

use super::network::Pipe;

pub(crate) const BPR_ALPHA: f64 = 0.05;
pub(crate) const BPR_BETA: f64 = 4.0;

#[derive(Clone, Copy, Default)]
pub struct ResistanceModel;

impl ResistanceModel {
    #[inline(always)]
    pub fn effective_resistance(&self, pipe: &Pipe) -> f64 {
        let base = pipe.base_resistance;
        let cap = pipe.capacity;
        if cap <= 0.0 {
            return base;
        }
        let usage = pipe.net_count.max(0.0);
        let ratio = usage / cap;
        base * (1.0 + BPR_ALPHA * ratio.powf(BPR_BETA))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::{Direction, PipeType};

    fn test_pipe(capacity: f64, net_count: f64) -> Pipe {
        Pipe {
            from: 0,
            to: 1,
            base_resistance: 1.0,
            capacity,
            flow: 0.0,
            net_count,
            raw_cell_density: 0.0,
            cell_density: 0.0,
            eff_conductance: 1.0,
            pipe_type: PipeType::InterTile(Direction::East),
        }
    }

    #[test]
    fn zero_usage_returns_base_resistance() {
        let model = ResistanceModel;
        let pipe = test_pipe(10.0, 0.0);
        let r = model.effective_resistance(&pipe);
        // base * cap / (cap - 0) = base = 1.0
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn half_capacity_is_mild() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&test_pipe(10.0, 5.0));
        let expected = 1.0 * (1.0 + BPR_ALPHA * 0.5f64.powf(BPR_BETA));
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn full_capacity_is_alpha_over_base() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&test_pipe(10.0, 10.0));
        assert!((r - (1.0 + BPR_ALPHA)).abs() < 1e-12);
    }

    #[test]
    fn overload_grows_polynomially() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&test_pipe(10.0, 20.0));
        let expected = 1.0 * (1.0 + BPR_ALPHA * 2.0f64.powf(BPR_BETA));
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn overload_grows_unbounded() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&test_pipe(1.0, 100.0));
        let expected = 1.0 * (1.0 + BPR_ALPHA * 100.0f64.powf(BPR_BETA));
        assert!((r - expected).abs() < 1e-6);
    }

    #[test]
    fn zero_capacity_returns_base() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&test_pipe(0.0, 5.0));
        assert!((r - 1.0).abs() < 1e-12);
    }
}
