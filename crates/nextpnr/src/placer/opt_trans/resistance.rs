//! Parameterless log-barrier congestion model.
//!
//! The effective resistance of a pipe grows as its usage approaches capacity:
//!     R_eff = base * cap / max(EPS, cap - usage)
//! At zero usage `R_eff = base`. As usage approaches `cap`, resistance grows
//! smoothly and diverges, so the path solver routes around saturated pipes.
//! `EPS` is a numerical clamp that activates at ~99.9% saturation; it is not
//! a tuning knob. The only input is `pipe.net_count`, which carries the hard
//! per-pipe usage count produced by the previous iteration's Dijkstra (set
//! by `refresh_net_counts_from_usage` in `algorithm.rs`).

use super::network::Pipe;

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
        const EPS: f64 = 1e-3;
        let usage = pipe.net_count as f64;
        let headroom = (cap - usage).max(EPS);
        base * cap / headroom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::{Direction, PipeType};

    fn test_pipe(capacity: f64, net_count: u32) -> Pipe {
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
        let pipe = test_pipe(10.0, 0);
        let r = model.effective_resistance(&pipe);
        // base * cap / (cap - 0) = base = 1.0
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn resistance_grows_with_usage() {
        let model = ResistanceModel;
        let r_low = model.effective_resistance(&test_pipe(10.0, 2));
        let r_high = model.effective_resistance(&test_pipe(10.0, 8));
        // 1 * 10 / (10-2) = 1.25 vs 1 * 10 / (10-8) = 5.0
        assert!(r_high > r_low);
        assert!((r_low - 1.25).abs() < 1e-12);
        assert!((r_high - 5.0).abs() < 1e-12);
    }

    #[test]
    fn saturated_pipe_clamps_to_finite_value() {
        let model = ResistanceModel;
        // usage >= cap: headroom clamps to EPS = 1e-3
        let r = model.effective_resistance(&test_pipe(10.0, 20));
        assert!(r.is_finite());
        // base * cap / EPS = 1 * 10 / 1e-3 = 1e4
        assert!((r - 1.0e4).abs() < 1e-6);
    }

    #[test]
    fn zero_capacity_returns_base() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&test_pipe(0.0, 5));
        assert!((r - 1.0).abs() < 1e-12);
    }
}
