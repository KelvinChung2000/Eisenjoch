use std::fmt;

/// Strength of a cell's placement constraint.
///
/// Higher values indicate stronger constraints that are harder to override.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(u8)]
pub enum PlaceStrength {
    /// No placement constraint.
    #[default]
    None = 0,
    /// Weak preference (e.g., from initial placement hints).
    Weak = 1,
    /// Strong preference.
    Strong = 2,
    /// Placed by the placer algorithm.
    Placer = 3,
    /// Fixed by the user or constraints file (cannot be moved by placer).
    Fixed = 4,
    /// Locked (stronger than fixed, e.g., from bitstream).
    Locked = 5,
    /// User-specified absolute constraint.
    User = 6,
}

impl PlaceStrength {
    /// Returns true if this placement strength prevents the placer from moving the cell.
    #[inline]
    pub fn is_locked(self) -> bool {
        matches!(self, Self::Fixed | Self::Locked | Self::User)
    }
}

impl fmt::Display for PlaceStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "NONE"),
            Self::Weak => write!(f, "WEAK"),
            Self::Strong => write!(f, "STRONG"),
            Self::Placer => write!(f, "PLACER"),
            Self::Fixed => write!(f, "FIXED"),
            Self::Locked => write!(f, "LOCKED"),
            Self::User => write!(f, "USER"),
        }
    }
}
