//! Shared low-level types used across multiple nextpnr subsystems.

mod binding;
mod id_string;

pub use binding::PlaceStrength;
pub use id_string::{IdString, IdStringPool, IntoIdString};
