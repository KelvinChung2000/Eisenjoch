//! SDC-style timing constraint storage.

use crate::common::IdString;
use crate::netlist::CellPin;
use crate::timing::{DelayPair, DelayT};

use super::domain::ClockDomainPair;
use rustc_hash::FxHashMap;

/// Definition of a clock from SDC `create_clock`.
#[derive(Clone, Debug)]
pub struct ClockDef {
    /// Clock name (user-defined or auto-derived from net).
    pub name: IdString,
    /// Clock period in picoseconds.
    pub period: DelayT,
    /// Waveform edges (rise_time, fall_time) in ps, relative to period start.
    pub waveform: (DelayT, DelayT),
    /// Source port, if specified.
    pub source_port: Option<IdString>,
}

/// Input or output delay constraint from SDC.
#[derive(Clone, Debug)]
pub struct IoDelay {
    /// Reference clock name.
    pub clock: IdString,
    /// Delay value in picoseconds.
    pub delay: DelayT,
    /// Ports this constraint applies to.
    pub ports: Vec<IdString>,
    /// Whether this is a max delay (true) or min delay (false).
    pub is_max: bool,
}

/// A false path exception from SDC `set_false_path`.
#[derive(Clone, Debug)]
pub struct FalsePath {
    pub from: Vec<CellPin>,
    pub to: Vec<CellPin>,
    pub through: Vec<CellPin>,
}

/// A multicycle path exception from SDC `set_multicycle_path`.
#[derive(Clone, Debug)]
pub struct MulticyclePath {
    pub from_clock: IdString,
    pub to_clock: IdString,
    pub setup_cycles: u32,
    pub hold_cycles: u32,
}

/// Clock group type from SDC `set_clock_groups`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockGroupType {
    Asynchronous,
    Exclusive,
    PhysicallyExclusive,
}

/// A clock group definition.
#[derive(Clone, Debug)]
pub struct ClockGroup {
    pub group_type: ClockGroupType,
    /// Groups of clock names. Clocks within the same sub-group are related;
    /// clocks in different sub-groups are unrelated per the group_type.
    pub groups: Vec<Vec<IdString>>,
}

/// Container for all SDC-style timing constraints.
#[derive(Clone, Debug, Default)]
pub struct SdcConstraints {
    pub clocks: Vec<ClockDef>,
    pub input_delays: Vec<IoDelay>,
    pub output_delays: Vec<IoDelay>,
    pub false_paths: Vec<FalsePath>,
    pub multicycle_paths: Vec<MulticyclePath>,
    pub max_delays: Vec<(DelayT, CellPin, CellPin)>,
    pub min_delays: Vec<(DelayT, CellPin, CellPin)>,
    pub clock_groups: Vec<ClockGroup>,
    pub clock_uncertainty: FxHashMap<ClockDomainPair, DelayPair>,
}

impl SdcConstraints {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a clock definition.
    pub fn add_clock(&mut self, def: ClockDef) {
        self.clocks.push(def);
    }

    /// Check if a false path exception matches a given (from, to) pair.
    pub fn is_false_path(&self, from: CellPin, to: CellPin) -> bool {
        self.false_paths.iter().any(|fp| {
            let from_match = fp.from.is_empty() || fp.from.contains(&from);
            let to_match = fp.to.is_empty() || fp.to.contains(&to);
            from_match && to_match
        })
    }

    /// Get multicycle path multiplier for a domain pair, if any.
    pub fn multicycle_setup(&self, launch_clk: IdString, capture_clk: IdString) -> Option<u32> {
        self.multicycle_paths
            .iter()
            .find(|mp| mp.from_clock == launch_clk && mp.to_clock == capture_clk)
            .map(|mp| mp.setup_cycles)
    }
}
