use ruststa::engine::types::ClockId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CritNorm {
    #[default]
    RequiredTime,
    WnsRelaxed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairBudget {
    pub launch: ClockId,
    pub capture: ClockId,
    pub d_req_ns: f64,
    pub wns_ns: f64,
}

impl CritNorm {
    pub fn pair_criticality(self, slack_ns: f64, budget: PairBudget) -> f64 {
        let relaxation_ns = match self {
            CritNorm::RequiredTime => 0.0,
            CritNorm::WnsRelaxed => budget.wns_ns.min(0.0),
        };
        let d_req_ns = budget.d_req_ns - relaxation_ns;
        if d_req_ns <= 0.0 {
            return 1.0;
        }
        (1.0 - (slack_ns - relaxation_ns) / d_req_ns).clamp(0.0, 1.0)
    }
}
