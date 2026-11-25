//! Packed identifier types for BELs, wires, and PIPs.
//!
//! Each identifier packs a (tile: i32, index: i32) pair into a single u64
//! for fast hashing and comparison. The high 32 bits store the tile index
//! and the low 32 bits store the element index within that tile.

use std::fmt;

#[inline]
const fn pack(tile: i32, index: i32) -> u64 {
    ((tile as u32 as u64) << 32) | (index as u32 as u64)
}

#[inline]
const fn unpack_tile(packed: u64) -> i32 {
    (packed >> 32) as u32 as i32
}

#[inline]
const fn unpack_index(packed: u64) -> i32 {
    packed as u32 as i32
}

macro_rules! define_packed_id {
    (
        $(#[$meta:meta])*
        $name:ident
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Sentinel value representing an invalid/unset identifier.
            /// Uses tile = -1 (0xFFFFFFFF) and index = 0.
            pub const INVALID: Self = Self(pack(-1, 0));

            /// Create a new identifier from tile and index.
            #[inline]
            pub const fn new(tile: i32, index: i32) -> Self {
                Self(pack(tile, index))
            }

            /// Get the tile index (high 32 bits).
            #[inline]
            pub const fn tile(self) -> i32 {
                unpack_tile(self.0)
            }

            /// Get the element index within the tile (low 32 bits).
            #[inline]
            pub const fn index(self) -> i32 {
                unpack_index(self.0)
            }

            /// Returns true if this identifier is valid (tile != -1).
            #[inline]
            pub const fn is_valid(self) -> bool {
                self.tile() != -1
            }

            /// Get the raw packed u64 value.
            #[inline]
            pub const fn raw(self) -> u64 {
                self.0
            }

            /// Create from a raw packed u64 value.
            #[inline]
            pub const fn from_raw(raw: u64) -> Self {
                Self(raw)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.is_valid() {
                    write!(f, "{}(tile={}, index={})", stringify!($name), self.tile(), self.index())
                } else {
                    write!(f, "{}(INVALID)", stringify!($name))
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.is_valid() {
                    write!(f, "{}:{}", self.tile(), self.index())
                } else {
                    write!(f, "<invalid>")
                }
            }
        }
    };
}

define_packed_id! {
    /// A BEL (Basic Element of Logic) identifier.
    ///
    /// Packs tile and index into a single u64 for efficient storage and hashing.
    BelId
}

define_packed_id! {
    /// A wire identifier.
    ///
    /// Packs tile and index into a single u64 for efficient storage and hashing.
    WireId
}

define_packed_id! {
    /// A PIP (Programmable Interconnect Point) identifier.
    ///
    /// Packs tile and index into a single u64 for efficient storage and hashing.
    PipId
}
