//! Core types for the nextpnr-rust FPGA place-and-route tool.
//!
//! This crate provides the fundamental types used throughout all other nextpnr
//! crates, including packed identifiers for chip elements, string interning,
//! delay representations, grid locations, and various enumerations.

mod delay;
mod enums;
mod id_string;
mod ids;
mod loc;
mod property;

// Re-export all public types at the crate root for convenience.
pub use delay::{DelayPair, DelayQuad, DelayT};
pub use enums::{ClockEdge, PlaceStrength, PortType, TimingPortClass};
pub use id_string::{IdString, IdStringPool};
pub use ids::{BelId, PipId, WireId};
pub use loc::Loc;
pub use property::Property;
