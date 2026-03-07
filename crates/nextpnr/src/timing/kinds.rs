//! Shared timing enums and classifications.

use std::fmt;

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
