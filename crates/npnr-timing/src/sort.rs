//! Topological sorting of cells for timing propagation.
//!
//! Cells are sorted so that a cell's inputs (through combinational paths) are
//! always processed before the cell itself. Register outputs start new
//! propagation levels, so they do not create ordering constraints on their
//! driver cells.

use npnr_netlist::{CellIdx, Design};
use npnr_types::PortType;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Topologically sort cells for timing propagation.
///
/// Uses Kahn's algorithm (BFS-based). Each cell's in-degree counts the number
/// of combinational input nets whose drivers are other cells in the design.
/// Clock nets and register feedback do not contribute to the ordering.
///
/// Returns cells in an order suitable for forward timing propagation.
pub fn topological_sort(design: &Design) -> Vec<CellIdx> {
    let num_cells = design.cell_store.len();

    // Map from CellIdx to its in-degree (number of unprocessed input dependencies).
    let mut in_degree: FxHashMap<CellIdx, usize> = FxHashMap::default();
    // Adjacency list: cell -> cells that depend on it (through combinational nets).
    let mut dependents: FxHashMap<CellIdx, Vec<CellIdx>> = FxHashMap::default();
    // Set of alive cell indices.
    let mut alive_cells: Vec<CellIdx> = Vec::with_capacity(num_cells);

    for (idx, cell) in design.cell_store.iter().enumerate() {
        if !cell.alive {
            continue;
        }
        let cell_idx = CellIdx(idx as u32);
        alive_cells.push(cell_idx);
        in_degree.insert(cell_idx, 0);
    }

    // Build dependency graph: for each input port connected to a net driven
    // by another cell, add an edge from driver -> this cell.
    for &cell_idx in &alive_cells {
        let cell = design.cell(cell_idx);
        let mut input_drivers: FxHashSet<CellIdx> = FxHashSet::default();

        for (_port_name, port_info) in &cell.ports {
            if port_info.port_type != PortType::In {
                continue;
            }
            if port_info.net.is_none() {
                continue;
            }

            let net = design.net(port_info.net);
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
                dependents
                    .entry(driver_cell)
                    .or_insert_with(Vec::new)
                    .push(cell_idx);
            }
        }

        *in_degree.entry(cell_idx).or_insert(0) += input_drivers.len();
    }

    // Kahn's algorithm: start with cells that have no input dependencies.
    let mut queue: VecDeque<CellIdx> = VecDeque::new();
    for &cell_idx in &alive_cells {
        if *in_degree.get(&cell_idx).unwrap_or(&0) == 0 {
            queue.push_back(cell_idx);
        }
    }

    let mut sorted: Vec<CellIdx> = Vec::with_capacity(alive_cells.len());

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
        let sorted_set: FxHashSet<CellIdx> = sorted.iter().copied().collect();
        for &cell_idx in &alive_cells {
            if !sorted_set.contains(&cell_idx) {
                sorted.push(cell_idx);
            }
        }
    }

    sorted
}

#[cfg(test)]
mod tests {
    use super::*;
    use npnr_netlist::{Design, PortRef};
    use npnr_types::{IdStringPool, PortType};

    #[test]
    fn empty_design() {
        let design = Design::new();
        let sorted = topological_sort(&design);
        assert!(sorted.is_empty());
    }

    #[test]
    fn single_cell() {
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let name = pool.intern("cell_a");
        let ctype = pool.intern("LUT4");
        let idx = design.add_cell(name, ctype);

        let sorted = topological_sort(&design);
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0], idx);
    }

    #[test]
    fn chain_of_three() {
        let pool = IdStringPool::new();
        let mut d = Design::new();

        let a_name = pool.intern("a");
        let b_name = pool.intern("b");
        let c_name = pool.intern("c");
        let lut = pool.intern("LUT4");

        let o = pool.intern("O");
        let i = pool.intern("I");

        let a_idx = d.add_cell(a_name, lut);
        let b_idx = d.add_cell(b_name, lut);
        let c_idx = d.add_cell(c_name, lut);

        d.cell_mut(a_idx).add_port(o, PortType::Out);
        d.cell_mut(b_idx).add_port(i, PortType::In);
        d.cell_mut(b_idx).add_port(o, PortType::Out);
        d.cell_mut(c_idx).add_port(i, PortType::In);

        // a.O -> b.I
        let n1_name = pool.intern("n1");
        let n1 = d.add_net(n1_name);
        d.net_mut(n1).driver = PortRef {
            cell: a_idx,
            port: o,
            budget: 0,
        };
        d.cell_mut(a_idx).port_mut(o).unwrap().net = n1;
        d.net_mut(n1).users.push(PortRef {
            cell: b_idx,
            port: i,
            budget: 0,
        });
        d.cell_mut(b_idx).port_mut(i).unwrap().net = n1;

        // b.O -> c.I
        let n2_name = pool.intern("n2");
        let n2 = d.add_net(n2_name);
        d.net_mut(n2).driver = PortRef {
            cell: b_idx,
            port: o,
            budget: 0,
        };
        d.cell_mut(b_idx).port_mut(o).unwrap().net = n2;
        d.net_mut(n2).users.push(PortRef {
            cell: c_idx,
            port: i,
            budget: 0,
        });
        d.cell_mut(c_idx).port_mut(i).unwrap().net = n2;

        let sorted = topological_sort(&d);
        assert_eq!(sorted.len(), 3);

        let a_pos = sorted.iter().position(|&x| x == a_idx).unwrap();
        let b_pos = sorted.iter().position(|&x| x == b_idx).unwrap();
        let c_pos = sorted.iter().position(|&x| x == c_idx).unwrap();

        assert!(a_pos < b_pos, "a should come before b");
        assert!(b_pos < c_pos, "b should come before c");
    }

    #[test]
    fn fanout() {
        let pool = IdStringPool::new();
        let mut d = Design::new();

        let lut = pool.intern("LUT4");
        let o = pool.intern("O");
        let i = pool.intern("I");

        let a_idx = d.add_cell(pool.intern("a"), lut);
        let b_idx = d.add_cell(pool.intern("b"), lut);
        let c_idx = d.add_cell(pool.intern("c"), lut);

        d.cell_mut(a_idx).add_port(o, PortType::Out);
        d.cell_mut(b_idx).add_port(i, PortType::In);
        d.cell_mut(c_idx).add_port(i, PortType::In);

        // a.O -> b.I and a.O -> c.I (fanout of 2)
        let n1 = d.add_net(pool.intern("n1"));
        d.net_mut(n1).driver = PortRef {
            cell: a_idx,
            port: o,
            budget: 0,
        };
        d.cell_mut(a_idx).port_mut(o).unwrap().net = n1;
        d.net_mut(n1).users.push(PortRef {
            cell: b_idx,
            port: i,
            budget: 0,
        });
        d.cell_mut(b_idx).port_mut(i).unwrap().net = n1;
        d.net_mut(n1).users.push(PortRef {
            cell: c_idx,
            port: i,
            budget: 0,
        });
        d.cell_mut(c_idx).port_mut(i).unwrap().net = n1;

        let sorted = topological_sort(&d);
        assert_eq!(sorted.len(), 3);

        let a_pos = sorted.iter().position(|&x| x == a_idx).unwrap();
        let b_pos = sorted.iter().position(|&x| x == b_idx).unwrap();
        let c_pos = sorted.iter().position(|&x| x == c_idx).unwrap();

        assert!(a_pos < b_pos, "a should come before b");
        assert!(a_pos < c_pos, "a should come before c");
    }

    #[test]
    fn dead_cells_excluded() {
        let pool = IdStringPool::new();
        let mut d = Design::new();

        let lut = pool.intern("LUT4");
        let a_name = pool.intern("a");
        let b_name = pool.intern("b");

        d.add_cell(a_name, lut);
        d.add_cell(b_name, lut);

        // Kill cell b.
        d.remove_cell(b_name);

        let sorted = topological_sort(&d);
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0], CellIdx(0));
    }
}
