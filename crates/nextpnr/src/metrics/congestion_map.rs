use crate::chipdb::WireId;
use rustc_hash::FxHashMap;

/// Shared wire-level congestion tracking for routing algorithms.
///
/// Tracks wire usage counts and accumulated penalties.
pub struct CongestionMap {
    wire_usage: FxHashMap<WireId, u32>,
    wire_penalty: FxHashMap<WireId, f64>,
    base_penalty: f64,
}

impl CongestionMap {
    pub fn new(base_penalty: f64) -> Self {
        Self {
            wire_usage: FxHashMap::default(),
            wire_penalty: FxHashMap::default(),
            base_penalty,
        }
    }

    pub fn record_wire_use(&mut self, wire: WireId) {
        *self.wire_usage.entry(wire).or_insert(0) += 1;
    }

    pub fn release_wire(&mut self, wire: WireId) {
        if let Some(count) = self.wire_usage.get_mut(&wire) {
            *count = count.saturating_sub(1);
        }
    }

    pub fn usage(&self, wire: WireId) -> u32 {
        self.wire_usage.get(&wire).copied().unwrap_or(0)
    }

    pub fn penalty(&self, wire: WireId) -> f64 {
        self.wire_penalty.get(&wire).copied().unwrap_or(self.base_penalty)
    }

    /// Update penalties based on current overuse.
    pub fn update_penalties(&mut self, factor: f64) {
        for (&wire, &usage) in &self.wire_usage {
            if usage > 1 {
                let penalty = self.wire_penalty.entry(wire).or_insert(self.base_penalty);
                *penalty += factor * (usage as f64 - 1.0);
            }
        }
    }

    pub fn clear(&mut self) {
        self.wire_usage.clear();
        self.wire_penalty.clear();
    }

    pub fn overused_wire_count(&self) -> usize {
        self.wire_usage.values().filter(|&&u| u > 1).count()
    }
}
