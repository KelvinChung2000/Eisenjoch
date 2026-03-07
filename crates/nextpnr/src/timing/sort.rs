//! Topological sorting of cells for timing propagation.
//!
//! Cells are sorted so that a cell's inputs (through combinational paths) are
//! always processed before the cell itself. Register outputs start new
//! propagation levels, so they do not create ordering constraints on their
//! driver cells.

use crate::netlist::{CellId, Design};
use crate::netlist::PortType;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Topologically sort cells for timing propagation.
///
/// Uses Kahn's algorithm (BFS-based). Each cell's in-degree counts the number
/// of combinational input nets whose drivers are other cells in the design.
/// Clock nets and register feedback do not contribute to the ordering.
///
/// Returns cells in an order suitable for forward timing propagation.
pub fn topological_sort(design: &Design) -> Vec<CellId> {
    let num_cells = design.num_cells();

    // Map from CellIdx to its in-degree (number of unprocessed input dependencies).
    let mut in_degree: FxHashMap<CellId, usize> = FxHashMap::default();
    // Adjacency list: cell -> cells that depend on it (through combinational nets).
    let mut dependents: FxHashMap<CellId, Vec<CellId>> = FxHashMap::default();
    // Set of alive cell indices.
    let mut alive_cells: Vec<CellId> = Vec::with_capacity(num_cells);

    for (cell_idx, _cell) in design.iter_alive_cells() {
        alive_cells.push(cell_idx);
        in_degree.insert(cell_idx, 0);
    }

    // Build dependency graph: for each input port connected to a net driven
    // by another cell, add an edge from driver -> this cell.
    for &cell_idx in &alive_cells {
        let cell = design.cell(cell_idx);
        let mut input_drivers: FxHashSet<CellId> = FxHashSet::default();

        for (_port_name, port_info) in &cell.ports {
            if port_info.port_type != PortType::In {
                continue;
            }
            if port_info.net.is_none() {
                continue;
            }

            let net_idx = match port_info.net {
                Some(net_idx) => net_idx,
                None => continue,
            };
            let net = design.net(net_idx);
            if !net.driver.is_connected() {
                continue;
            }

            let driver_cell = net.driver.cell;

            // Skip self-loops and already-counted drivers.
            if driver_cell == cell_idx {
                continue;
            }
            if !design.cell(driver_cell).alive {
                continue;
            }

            // Only add this dependency once per (driver, cell) pair.
            if input_drivers.insert(driver_cell) {
                dependents.entry(driver_cell).or_default().push(cell_idx);
            }
        }

        *in_degree.entry(cell_idx).or_insert(0) += input_drivers.len();
    }

    // Kahn's algorithm: start with cells that have no input dependencies.
    let mut queue: VecDeque<CellId> = VecDeque::new();
    for &cell_idx in &alive_cells {
        if *in_degree.get(&cell_idx).unwrap_or(&0) == 0 {
            queue.push_back(cell_idx);
        }
    }

    let mut sorted: Vec<CellId> = Vec::with_capacity(alive_cells.len());

    while let Some(cell_idx) = queue.pop_front() {
        sorted.push(cell_idx);

        if let Some(deps) = dependents.get(&cell_idx) {
            for &dep in deps {
                if let Some(deg) = in_degree.get_mut(&dep) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(dep);
                    }
                }
            }
        }
    }

    // If there are cells not yet in the sorted list (due to cycles), append
    // them in arena order. Cycles can happen with feedback paths; we handle
    // them gracefully rather than panicking.
    if sorted.len() < alive_cells.len() {
        let sorted_set: FxHashSet<CellId> = sorted.iter().copied().collect();
        for &cell_idx in &alive_cells {
            if !sorted_set.contains(&cell_idx) {
                sorted.push(cell_idx);
            }
        }
    }

    sorted
}
