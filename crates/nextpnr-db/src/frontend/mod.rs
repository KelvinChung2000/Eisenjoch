//! Yosys JSON netlist parser for the nextpnr-rust FPGA place-and-route tool.
//!
//! This module reads the JSON output produced by Yosys (`write_json`) and
//! populates a [`Design`](crate::netlist::Design) with cells, nets, ports, parameters,
//! and attributes.

mod helpers;
mod parser;
mod routed;

pub use helpers::{parse_bit_value, parse_port_direction, parse_property, port_bit_name, BitValue};
pub use parser::{parse_json, IO_CELL_PREFIX};
pub use routed::{
    parse_routed_json, region_rects, AttributeFault, RoutedAttribute, RoutedJsonError,
    REGION_SETTING_PREFIX, SDC_SETTING,
};
