//! The [`Property`] type for representing cell parameters and attributes.
//!
//! In Yosys JSON netlists, properties can be strings, integers, or bit vectors.

use std::fmt;

/// A property value that can be attached to cells and nets.
///
/// Mirrors the Yosys property representation: strings, integers, or bit vectors.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Property {
    /// A string value.
    String(String),
    /// An integer value (64-bit signed).
    Int(i64),
    /// A bit vector, stored as a string of '0', '1', 'x', 'z' characters.
    /// The first character is the MSB.
    BitVector(String),
}

impl Property {
    /// Create a string property.
    pub fn string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    /// Create an integer property.
    pub const fn int(v: i64) -> Self {
        Self::Int(v)
    }

    /// Create a bit vector property from a string of '0'/'1'/'x'/'z' characters.
    pub fn bit_vector(bits: impl Into<String>) -> Self {
        Self::BitVector(bits.into())
    }

    /// Try to interpret this property as an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::String(s) => s.parse().ok(),
            Self::BitVector(bits) => {
                if bits.is_empty() {
                    return Some(0);
                }
                if bits.chars().all(|c| c == '0' || c == '1') {
                    Some(i64::from_str_radix(bits, 2).unwrap_or(0))
                } else {
                    None
                }
            }
        }
    }

    /// Try to interpret this property as a string.
    pub fn as_str(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Int(v) => v.to_string(),
            Self::BitVector(bits) => bits.clone(),
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    pub fn is_bit_vector(&self) -> bool {
        matches!(self, Self::BitVector(_))
    }
}

impl Default for Property {
    fn default() -> Self {
        Self::String(String::new())
    }
}

impl fmt::Display for Property {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => write!(f, "{}", s),
            Self::Int(v) => write!(f, "{}", v),
            Self::BitVector(bits) => write!(f, "{}", bits),
        }
    }
}

impl From<&str> for Property {
    fn from(s: &str) -> Self {
        Self::String(s.to_owned())
    }
}

impl From<String> for Property {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<i64> for Property {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<i32> for Property {
    fn from(v: i32) -> Self {
        Self::Int(v as i64)
    }
}

impl From<bool> for Property {
    fn from(v: bool) -> Self {
        Self::Int(if v { 1 } else { 0 })
    }
}
