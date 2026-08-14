//! Central context struct and architecture API for the nextpnr-rust FPGA
//! place-and-route tool.
//!
//! The [`Context`] ties together the read-only chip database ([`ChipDb`]) with the
//! mutable design netlist ([`Design`]), and maintains the placement and routing
//! state maps that track which hardware resources (bels, wires, pips) are bound
//! to which design elements (cells, nets).
//!
//! All placer, router, and timing code operates through the `Context`.

mod arch_api;
mod buckets;
mod core;
mod definition;
mod occupancy;
mod rng;
mod storage;
mod timing;
mod views;

pub use arch_api::BoundingBox;
pub use crate::metrics::{ResourceRow, UtilizationReport};
pub use definition::{ArchValidityCheck, Context};
pub use rng::DeterministicRng;
pub use views::{
    Bel, BelPin, BelPinView, Cell, CellPinView, ChipView, IdStringView, Net, Pip, PlacementView,
    TileView, Wire,
};
