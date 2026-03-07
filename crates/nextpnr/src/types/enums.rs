//! Enumeration types used throughout nextpnr.

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

/// Direction of a port on a cell.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(u8)]
pub enum PortType {
    /// Input port.
    #[default]
    In = 0,
    /// Output port.
    Out = 1,
    /// Bidirectional port.
    InOut = 2,
}

impl fmt::Display for PortType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::In => write!(f, "IN"),
            Self::Out => write!(f, "OUT"),
            Self::InOut => write!(f, "INOUT"),
        }
    }
}

/// Classification of a timing port for static timing analysis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(u8)]
pub enum TimingPortClass {
    /// Combinational path through this port.
    #[default]
    Combinational = 0,
    /// Register input (data/enable/reset).
    RegisterInput = 1,
    /// Register output (Q).
    RegisterOutput = 2,
    /// Clock input to a register.
    ClockInput = 3,
    /// Generated clock output.
    GenClock = 4,
    /// Port should be ignored for timing analysis.
    Ignore = 5,
}

impl fmt::Display for TimingPortClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Combinational => write!(f, "COMBINATIONAL"),
            Self::RegisterInput => write!(f, "REGISTER_INPUT"),
            Self::RegisterOutput => write!(f, "REGISTER_OUTPUT"),
            Self::ClockInput => write!(f, "CLOCK_INPUT"),
            Self::GenClock => write!(f, "GEN_CLOCK"),
            Self::Ignore => write!(f, "IGNORE"),
        }
    }
}

impl TimingPortClass {
    /// Returns true if this port class is a register endpoint (input or output).
    #[inline]
    pub fn is_register(self) -> bool {
        matches!(self, Self::RegisterInput | Self::RegisterOutput)
    }

    /// Returns true if this port class is clock-related.
    #[inline]
    pub fn is_clock(self) -> bool {
        matches!(self, Self::ClockInput | Self::GenClock)
    }
}

/// Edge of a clock signal.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(u8)]
pub enum ClockEdge {
    /// Rising edge (positive edge).
    #[default]
    Rising = 0,
    /// Falling edge (negative edge).
    Falling = 1,
}

impl ClockEdge {
    /// Returns the opposite edge.
    #[inline]
    pub fn opposite(self) -> Self {
        match self {
            Self::Rising => Self::Falling,
            Self::Falling => Self::Rising,
        }
    }
}

impl fmt::Display for ClockEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rising => write!(f, "RISING"),
            Self::Falling => write!(f, "FALLING"),
        }
    }
}
