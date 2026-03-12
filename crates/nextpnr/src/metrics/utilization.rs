//! Resource utilization report for the FPGA design.

use crate::context::Context;
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;
use std::fmt;

/// A single row in the utilization report.
#[derive(Debug, Clone)]
pub struct ResourceRow {
    pub resource: String,
    pub used: usize,
    pub available: usize,
}

impl ResourceRow {
    pub fn percent(&self) -> f64 {
        if self.available == 0 {
            0.0
        } else {
            100.0 * self.used as f64 / self.available as f64
        }
    }
}

/// Full utilization report.
#[derive(Debug, Clone)]
pub struct UtilizationReport {
    pub rows: Vec<ResourceRow>,
    pub total_cells: usize,
    pub total_nets: usize,
    pub placed_cells: usize,
}

impl fmt::Display for UtilizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Resource Utilization")?;
        writeln!(f, "{:<20} {:>8} {:>10} {:>8}", "Resource", "Used", "Available", "Util%")?;
        writeln!(f, "{}", "-".repeat(48))?;
        for row in &self.rows {
            writeln!(f, "{:<20} {:>8} {:>10} {:>7.1}%", row.resource, row.used, row.available, row.percent())?;
        }
        writeln!(f, "{}", "-".repeat(48))?;
        writeln!(f, "Design: {} cells, {} nets, {}/{} placed", self.total_cells, self.total_nets, self.placed_cells, self.total_cells)
    }
}

/// Generate a resource utilization report.
///
/// Counts available BELs by type from the chipdb, and used cells by type
/// from the design, producing a per-resource-type summary.
pub fn utilization_report(ctx: &Context) -> UtilizationReport {
    let mut available: FxHashMap<&str, usize> = FxHashMap::default();
    for bel in ctx.bels() {
        *available.entry(bel.bel_type()).or_insert(0) += 1;
    }

    let mut used: FxHashMap<&str, usize> = FxHashMap::default();
    let mut total_cells = 0;
    let mut placed_cells = 0;
    for cell in ctx.cells() {
        total_cells += 1;
        *used.entry(cell.cell_type()).or_insert(0) += 1;
        if cell.bel_id().is_some() {
            placed_cells += 1;
        }
    }

    let total_nets = ctx.design.num_nets();

    let all_types: BTreeSet<&str> = available.keys().chain(used.keys()).copied().collect();
    let rows = all_types
        .into_iter()
        .map(|t| ResourceRow {
            resource: t.to_string(),
            used: used.get(t).copied().unwrap_or(0),
            available: available.get(t).copied().unwrap_or(0),
        })
        .collect();

    UtilizationReport {
        rows,
        total_cells,
        total_nets,
        placed_cells,
    }
}
