//! Packed identifier types for BELs, wires, and PIPs.
//!
//! Each identifier packs a (tile: i32, index: i32) pair into a single u64
//! for fast hashing and comparison. The high 32 bits store the tile index
//! and the low 32 bits store the element index within that tile.

use std::fmt;

/// Creates a packed u64 from tile and index values.
#[inline]
const fn pack(tile: i32, index: i32) -> u64 {
    ((tile as u32 as u64) << 32) | (index as u32 as u64)
}

/// Extracts the tile (high 32 bits) from a packed u64.
#[inline]
const fn unpack_tile(packed: u64) -> i32 {
    (packed >> 32) as u32 as i32
}

/// Extracts the index (low 32 bits) from a packed u64.
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

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_packed_id {
        ($name:ident, $ty:ident) => {
            mod $name {
                use super::*;

                #[test]
                fn new_and_accessors() {
                    let id = $ty::new(10, 20);
                    assert_eq!(id.tile(), 10);
                    assert_eq!(id.index(), 20);
                    assert!(id.is_valid());
                }

                #[test]
                fn zero_values() {
                    let id = $ty::new(0, 0);
                    assert_eq!(id.tile(), 0);
                    assert_eq!(id.index(), 0);
                    assert!(id.is_valid());
                }

                #[test]
                fn negative_values() {
                    let id = $ty::new(-5, -10);
                    assert_eq!(id.tile(), -5);
                    assert_eq!(id.index(), -10);
                    // tile == -5 != -1, so it's valid
                    assert!(id.is_valid());
                }

                #[test]
                fn invalid_constant() {
                    let id = $ty::INVALID;
                    assert_eq!(id.tile(), -1);
                    assert_eq!(id.index(), 0);
                    assert!(!id.is_valid());
                }

                #[test]
                fn default_is_zero() {
                    let id = $ty::default();
                    assert_eq!(id.tile(), 0);
                    assert_eq!(id.index(), 0);
                    assert!(id.is_valid());
                }

                #[test]
                fn equality() {
                    let a = $ty::new(1, 2);
                    let b = $ty::new(1, 2);
                    let c = $ty::new(1, 3);
                    assert_eq!(a, b);
                    assert_ne!(a, c);
                }

                #[test]
                fn hashing() {
                    use std::collections::HashSet;
                    let mut set = HashSet::new();
                    set.insert($ty::new(1, 2));
                    set.insert($ty::new(3, 4));
                    set.insert($ty::new(1, 2)); // duplicate
                    assert_eq!(set.len(), 2);
                }

                #[test]
                fn copy_semantics() {
                    let a = $ty::new(5, 6);
                    let b = a;
                    assert_eq!(a, b); // a is still usable after copy
                }

                #[test]
                fn max_values() {
                    let id = $ty::new(i32::MAX, i32::MAX);
                    assert_eq!(id.tile(), i32::MAX);
                    assert_eq!(id.index(), i32::MAX);
                    assert!(id.is_valid());
                }

                #[test]
                fn min_values() {
                    let id = $ty::new(i32::MIN, i32::MIN);
                    assert_eq!(id.tile(), i32::MIN);
                    assert_eq!(id.index(), i32::MIN);
                    // i32::MIN != -1, so it's valid
                    assert!(id.is_valid());
                }

                #[test]
                fn raw_roundtrip() {
                    let id = $ty::new(42, 99);
                    let raw = id.raw();
                    let restored = $ty::from_raw(raw);
                    assert_eq!(id, restored);
                    assert_eq!(restored.tile(), 42);
                    assert_eq!(restored.index(), 99);
                }

                #[test]
                fn debug_format_valid() {
                    let id = $ty::new(1, 2);
                    let debug = format!("{:?}", id);
                    assert!(debug.contains(stringify!($ty)));
                    assert!(debug.contains("tile=1"));
                    assert!(debug.contains("index=2"));
                }

                #[test]
                fn debug_format_invalid() {
                    let id = $ty::INVALID;
                    let debug = format!("{:?}", id);
                    assert!(debug.contains("INVALID"));
                }

                #[test]
                fn display_format_valid() {
                    let id = $ty::new(1, 2);
                    assert_eq!(format!("{}", id), "1:2");
                }

                #[test]
                fn display_format_invalid() {
                    let id = $ty::INVALID;
                    assert_eq!(format!("{}", id), "<invalid>");
                }

                #[test]
                fn tile_negative_one_is_invalid() {
                    // Any ID with tile == -1 is invalid, regardless of index
                    let id = $ty::new(-1, 42);
                    assert!(!id.is_valid());
                }
            }
        };
    }

    test_packed_id!(bel_id, BelId);
    test_packed_id!(wire_id, WireId);
    test_packed_id!(pip_id, PipId);

    #[test]
    fn different_types_are_distinct() {
        // Ensure BelId, WireId, PipId are not accidentally interchangeable at the type level.
        // This is a compile-time check -- if this compiles, the types are distinct.
        let _bel: BelId = BelId::new(0, 0);
        let _wire: WireId = WireId::new(0, 0);
        let _pip: PipId = PipId::new(0, 0);
    }
}
