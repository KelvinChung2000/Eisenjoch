//! Netlist data structures for the nextpnr-rust FPGA place-and-route tool.

mod cell;
mod cluster;
mod design;
mod editors;
mod hierarchy;
mod indices;
mod net;
mod port_kind;
mod ports;
mod property;

pub use cell::CellInfo;
pub use cluster::Cluster;
pub use design::Design;
pub use editors::{CellEditor, NetEditor};
pub use hierarchy::{HierarchicalCell, HierarchicalNet};
pub use indices::{CellId, FlatIndex, NetId, TimingIndex};
pub use net::{NetInfo, PipMap};
pub use port_kind::PortType;
pub(crate) use ports::PortData;
pub use ports::CellPin;
pub use property::Property;
