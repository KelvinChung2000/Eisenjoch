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
    String(std::string::String),
    /// An integer value (64-bit signed).
    Int(i64),
    /// A bit vector, stored as a string of '0', '1', 'x', 'z' characters.
    /// The first character is the MSB.
    BitVector(std::string::String),
}

impl Property {
    /// Create a string property.
    pub fn string(s: impl Into<std::string::String>) -> Self {
        Self::String(s.into())
    }

    /// Create an integer property.
    pub const fn int(v: i64) -> Self {
        Self::Int(v)
    }

    /// Create a bit vector property from a string of '0'/'1'/'x'/'z' characters.
    pub fn bit_vector(bits: impl Into<std::string::String>) -> Self {
        Self::BitVector(bits.into())
    }

    /// Try to interpret this property as an integer.
    ///
    /// - `Int` values are returned directly.
    /// - `String` values are parsed as decimal integers.
    /// - `BitVector` values are interpreted as binary (only if all bits are '0' or '1').
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
    ///
    /// - `String` values are returned directly.
    /// - `Int` values are formatted as decimal strings.
    /// - `BitVector` values are returned as-is.
    pub fn as_str(&self) -> std::string::String {
        match self {
            Self::String(s) => s.clone(),
            Self::Int(v) => v.to_string(),
            Self::BitVector(bits) => bits.clone(),
        }
    }

    /// Returns true if this is a string property.
    pub fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }

    /// Returns true if this is an integer property.
    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    /// Returns true if this is a bit vector property.
    pub fn is_bit_vector(&self) -> bool {
        matches!(self, Self::BitVector(_))
    }
}

impl Default for Property {
    fn default() -> Self {
        Self::String(std::string::String::new())
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

impl From<std::string::String> for Property {
    fn from(s: std::string::String) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_property() {
        let p = Property::string("hello");
        assert!(p.is_string());
        assert!(!p.is_int());
        assert!(!p.is_bit_vector());
        assert_eq!(p.as_str(), "hello");
    }

    #[test]
    fn int_property() {
        let p = Property::int(42);
        assert!(p.is_int());
        assert!(!p.is_string());
        assert!(!p.is_bit_vector());
        assert_eq!(p.as_int(), Some(42));
    }

    #[test]
    fn bit_vector_property() {
        let p = Property::bit_vector("1010");
        assert!(p.is_bit_vector());
        assert!(!p.is_string());
        assert!(!p.is_int());
        assert_eq!(p.as_int(), Some(0b1010));
    }

    #[test]
    fn bit_vector_with_unknown() {
        let p = Property::bit_vector("10x1");
        assert!(p.is_bit_vector());
        assert_eq!(p.as_int(), None); // cannot convert to int with 'x'
    }

    #[test]
    fn bit_vector_empty() {
        let p = Property::bit_vector("");
        assert_eq!(p.as_int(), Some(0));
    }

    #[test]
    fn string_as_int() {
        let p = Property::string("123");
        assert_eq!(p.as_int(), Some(123));
    }

    #[test]
    fn string_not_int() {
        let p = Property::string("not_a_number");
        assert_eq!(p.as_int(), None);
    }

    #[test]
    fn int_as_str() {
        let p = Property::int(42);
        assert_eq!(p.as_str(), "42");
    }

    #[test]
    fn default_is_empty_string() {
        let p = Property::default();
        assert!(p.is_string());
        assert_eq!(p.as_str(), "");
    }

    #[test]
    fn display_string() {
        let p = Property::string("test");
        assert_eq!(format!("{}", p), "test");
    }

    #[test]
    fn display_int() {
        let p = Property::int(-5);
        assert_eq!(format!("{}", p), "-5");
    }

    #[test]
    fn display_bit_vector() {
        let p = Property::bit_vector("1100");
        assert_eq!(format!("{}", p), "1100");
    }

    #[test]
    fn from_str_ref() {
        let p: Property = "hello".into();
        assert_eq!(p, Property::String("hello".to_owned()));
    }

    #[test]
    fn from_string() {
        let p: Property = String::from("hello").into();
        assert_eq!(p, Property::String("hello".to_owned()));
    }

    #[test]
    fn from_i64() {
        let p: Property = 42i64.into();
        assert_eq!(p, Property::Int(42));
    }

    #[test]
    fn from_i32() {
        let p: Property = 42i32.into();
        assert_eq!(p, Property::Int(42));
    }

    #[test]
    fn from_bool_true() {
        let p: Property = true.into();
        assert_eq!(p, Property::Int(1));
    }

    #[test]
    fn from_bool_false() {
        let p: Property = false.into();
        assert_eq!(p, Property::Int(0));
    }

    #[test]
    fn equality() {
        assert_eq!(Property::int(1), Property::int(1));
        assert_ne!(Property::int(1), Property::int(2));
        assert_ne!(Property::int(1), Property::string("1"));
    }

    #[test]
    fn clone() {
        let p = Property::string("test");
        let q = p.clone();
        assert_eq!(p, q);
    }

    #[test]
    fn negative_int() {
        let p = Property::int(-100);
        assert_eq!(p.as_int(), Some(-100));
        assert_eq!(p.as_str(), "-100");
    }

    #[test]
    fn bit_vector_all_zeros() {
        let p = Property::bit_vector("0000");
        assert_eq!(p.as_int(), Some(0));
    }

    #[test]
    fn bit_vector_all_ones() {
        let p = Property::bit_vector("1111");
        assert_eq!(p.as_int(), Some(0b1111));
    }

    #[test]
    fn hashing() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Property::int(1));
        set.insert(Property::int(2));
        set.insert(Property::int(1));
        assert_eq!(set.len(), 2);
    }
}
