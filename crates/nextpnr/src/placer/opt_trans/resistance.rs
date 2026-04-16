//! BPR congestion model for pipe costs.
//!
//! The effective resistance of a pipe follows:
//!     R_eff = base * (1 + alpha * (usage / eff_capacity)^beta)
//!     eff_capacity = capacity * borrow_slack(span)
//!
//! Under the one-wire-one-entry span model (Model A), each physical wire
//! contributes capacity only to its max-reach span bucket. A hex wire (L6)
//! registers at span=6 even though it physically has taps at every
//! intermediate tile and can serve span<=6 nets. To model this short-span
//! flexibility without double-counting wires across spans, we inflate the
//! effective capacity of short-span pipes by a small factor — the placer can
//! mildly overflow short-span pipes at a BPR-scaled penalty, which represents
//! the option to borrow longer wires at the cost of route-length waste.
//!
//! `borrow_slack(span)` linearly decreases from +25% at span=1 to 0 at
//! span>=SHORT_SPAN_BORROW_LIMIT (chipdb's longest "short-reach" wire class,
//! hex=6 for xc7). Beyond that, longer wires rarely serve shorter nets.

use super::network::{Pipe, PipeType};

pub(crate) const BPR_ALPHA: f64 = 0.05;
pub(crate) const BPR_BETA: f64 = 4.0;

/// Maximum span (in composite units) at which longer-wire borrowing is
/// credible. xc7 hex wires reach 6 composites vertically; any shorter span
/// can in principle be served by consuming part of a hex wire.
pub(crate) const SHORT_SPAN_BORROW_LIMIT: i32 = 6;

/// Per-step capacity inflation granted to each span below the borrow limit.
/// span=1 → +0.25 (25%), span=2 → +0.20, …, span=5 → +0.05, span>=6 → 0.
pub(crate) const BORROW_SLACK_PER_STEP: f64 = 0.05;

#[inline(always)]
fn pipe_span(pipe_type: &PipeType) -> i32 {
    match pipe_type {
        PipeType::InterTile(_) => 1,
        PipeType::LongRange { dx, dy } => dx.abs() + dy.abs(),
    }
}

/// Effective-capacity inflation factor accounting for longer-wire borrowing
/// on short-span pipes.
#[inline(always)]
pub(crate) fn borrow_slack(span: i32) -> f64 {
    if span >= SHORT_SPAN_BORROW_LIMIT {
        1.0
    } else {
        1.0 + BORROW_SLACK_PER_STEP * (SHORT_SPAN_BORROW_LIMIT - span) as f64
    }
}

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
        let slack = borrow_slack(pipe_span(&pipe.pipe_type));
        let eff_cap = cap * slack;
        let usage = pipe.net_count.max(0.0);
        let ratio = usage / eff_cap;
        base * (1.0 + BPR_ALPHA * ratio.powf(BPR_BETA))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::{Direction, PipeType};

    fn pipe_with_span(dx: i32, dy: i32, capacity: f64, net_count: f64) -> Pipe {
        let pipe_type = if dx.abs() + dy.abs() <= 1 {
            PipeType::InterTile(Direction::East)
        } else {
            PipeType::LongRange { dx, dy }
        };
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
            pipe_type,
        }
    }

    /// Pipe with no borrowing slack (span >= SHORT_SPAN_BORROW_LIMIT).
    fn long_pipe(capacity: f64, net_count: f64) -> Pipe {
        pipe_with_span(0, SHORT_SPAN_BORROW_LIMIT, capacity, net_count)
    }

    #[test]
    fn zero_usage_returns_base_resistance() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&long_pipe(10.0, 0.0));
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn half_capacity_is_mild() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&long_pipe(10.0, 5.0));
        let expected = 1.0 * (1.0 + BPR_ALPHA * 0.5f64.powf(BPR_BETA));
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn full_capacity_is_alpha_over_base() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&long_pipe(10.0, 10.0));
        assert!((r - (1.0 + BPR_ALPHA)).abs() < 1e-12);
    }

    #[test]
    fn overload_grows_polynomially() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&long_pipe(10.0, 20.0));
        let expected = 1.0 * (1.0 + BPR_ALPHA * 2.0f64.powf(BPR_BETA));
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn overload_grows_unbounded() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&long_pipe(1.0, 100.0));
        let expected = 1.0 * (1.0 + BPR_ALPHA * 100.0f64.powf(BPR_BETA));
        assert!((r - expected).abs() < 1e-6);
    }

    #[test]
    fn zero_capacity_returns_base() {
        let model = ResistanceModel;
        let r = model.effective_resistance(&long_pipe(0.0, 5.0));
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn borrow_slack_follows_span_schedule() {
        // span=1 → +25%, span=2 → +20%, …, span=5 → +5%, span>=6 → 0.
        assert!((borrow_slack(1) - 1.25).abs() < 1e-12);
        assert!((borrow_slack(2) - 1.20).abs() < 1e-12);
        assert!((borrow_slack(3) - 1.15).abs() < 1e-12);
        assert!((borrow_slack(4) - 1.10).abs() < 1e-12);
        assert!((borrow_slack(5) - 1.05).abs() < 1e-12);
        assert!((borrow_slack(6) - 1.0).abs() < 1e-12);
        assert!((borrow_slack(18) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn short_span_at_nominal_capacity_has_no_penalty_with_slack() {
        // Span-1 pipe at usage=capacity. Without slack: R = base * (1 + alpha).
        // With slack=1.25: ratio=1/1.25=0.8, R = base * (1 + alpha * 0.8^4).
        let model = ResistanceModel;
        let r = model.effective_resistance(&pipe_with_span(1, 0, 10.0, 10.0));
        let expected = 1.0 + BPR_ALPHA * (1.0 / 1.25f64).powf(BPR_BETA);
        assert!((r - expected).abs() < 1e-12, "got {}, expected {}", r, expected);
        // And that's strictly less than the no-slack penalty at ratio=1:
        assert!(r < 1.0 + BPR_ALPHA);
    }

    #[test]
    fn mild_short_span_overflow_scales_gently() {
        // span-1 pipe at 25% overflow (usage=1.25*cap). Effective ratio is
        // exactly 1.0, so R = base * (1 + alpha) — the same penalty a long
        // pipe gets at its nominal capacity.
        let model = ResistanceModel;
        let r = model.effective_resistance(&pipe_with_span(1, 0, 10.0, 12.5));
        let expected = 1.0 + BPR_ALPHA;
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn long_span_gets_no_slack() {
        // span-6 pipe at usage=capacity should match the pre-slack behavior.
        let model = ResistanceModel;
        let r = model.effective_resistance(&pipe_with_span(0, 6, 10.0, 10.0));
        assert!((r - (1.0 + BPR_ALPHA)).abs() < 1e-12);
    }
}
