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

#[cfg(test)]
mod tests {
    use super::*;

    // === PlaceStrength tests ===

    #[test]
    fn place_strength_default() {
        assert_eq!(PlaceStrength::default(), PlaceStrength::None);
    }

    #[test]
    fn place_strength_is_locked() {
        assert!(!PlaceStrength::None.is_locked());
        assert!(!PlaceStrength::Weak.is_locked());
        assert!(!PlaceStrength::Strong.is_locked());
        assert!(!PlaceStrength::Placer.is_locked());
        assert!(PlaceStrength::Fixed.is_locked());
        assert!(PlaceStrength::Locked.is_locked());
        assert!(PlaceStrength::User.is_locked());
    }

    #[test]
    fn place_strength_equality() {
        assert_eq!(PlaceStrength::None, PlaceStrength::None);
        assert_ne!(PlaceStrength::None, PlaceStrength::Weak);
    }

    #[test]
    fn place_strength_copy() {
        let a = PlaceStrength::Fixed;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn place_strength_display() {
        assert_eq!(format!("{}", PlaceStrength::None), "NONE");
        assert_eq!(format!("{}", PlaceStrength::Weak), "WEAK");
        assert_eq!(format!("{}", PlaceStrength::Strong), "STRONG");
        assert_eq!(format!("{}", PlaceStrength::Placer), "PLACER");
        assert_eq!(format!("{}", PlaceStrength::Fixed), "FIXED");
        assert_eq!(format!("{}", PlaceStrength::Locked), "LOCKED");
        assert_eq!(format!("{}", PlaceStrength::User), "USER");
    }

    #[test]
    fn place_strength_debug() {
        assert_eq!(format!("{:?}", PlaceStrength::Fixed), "Fixed");
    }

    #[test]
    fn place_strength_hashing() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PlaceStrength::None);
        set.insert(PlaceStrength::Fixed);
        set.insert(PlaceStrength::None);
        assert_eq!(set.len(), 2);
    }

    // === PortType tests ===

    #[test]
    fn port_type_default() {
        assert_eq!(PortType::default(), PortType::In);
    }

    #[test]
    fn port_type_equality() {
        assert_eq!(PortType::In, PortType::In);
        assert_ne!(PortType::In, PortType::Out);
        assert_ne!(PortType::Out, PortType::InOut);
    }

    #[test]
    fn port_type_display() {
        assert_eq!(format!("{}", PortType::In), "IN");
        assert_eq!(format!("{}", PortType::Out), "OUT");
        assert_eq!(format!("{}", PortType::InOut), "INOUT");
    }

    #[test]
    fn port_type_debug() {
        assert_eq!(format!("{:?}", PortType::In), "In");
    }

    #[test]
    fn port_type_copy() {
        let a = PortType::Out;
        let b = a;
        assert_eq!(a, b);
    }

    // === TimingPortClass tests ===

    #[test]
    fn timing_port_class_default() {
        assert_eq!(TimingPortClass::default(), TimingPortClass::Combinational);
    }

    #[test]
    fn timing_port_class_is_register() {
        assert!(!TimingPortClass::Combinational.is_register());
        assert!(TimingPortClass::RegisterInput.is_register());
        assert!(TimingPortClass::RegisterOutput.is_register());
        assert!(!TimingPortClass::ClockInput.is_register());
        assert!(!TimingPortClass::GenClock.is_register());
        assert!(!TimingPortClass::Ignore.is_register());
    }

    #[test]
    fn timing_port_class_is_clock() {
        assert!(!TimingPortClass::Combinational.is_clock());
        assert!(!TimingPortClass::RegisterInput.is_clock());
        assert!(!TimingPortClass::RegisterOutput.is_clock());
        assert!(TimingPortClass::ClockInput.is_clock());
        assert!(TimingPortClass::GenClock.is_clock());
        assert!(!TimingPortClass::Ignore.is_clock());
    }

    #[test]
    fn timing_port_class_display() {
        assert_eq!(format!("{}", TimingPortClass::Combinational), "COMBINATIONAL");
        assert_eq!(format!("{}", TimingPortClass::RegisterInput), "REGISTER_INPUT");
        assert_eq!(format!("{}", TimingPortClass::RegisterOutput), "REGISTER_OUTPUT");
        assert_eq!(format!("{}", TimingPortClass::ClockInput), "CLOCK_INPUT");
        assert_eq!(format!("{}", TimingPortClass::GenClock), "GEN_CLOCK");
        assert_eq!(format!("{}", TimingPortClass::Ignore), "IGNORE");
    }

    #[test]
    fn timing_port_class_equality() {
        assert_eq!(TimingPortClass::Combinational, TimingPortClass::Combinational);
        assert_ne!(TimingPortClass::Combinational, TimingPortClass::Ignore);
    }

    #[test]
    fn timing_port_class_hashing() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TimingPortClass::Combinational);
        set.insert(TimingPortClass::RegisterInput);
        set.insert(TimingPortClass::Combinational);
        assert_eq!(set.len(), 2);
    }

    // === ClockEdge tests ===

    #[test]
    fn clock_edge_default() {
        assert_eq!(ClockEdge::default(), ClockEdge::Rising);
    }

    #[test]
    fn clock_edge_opposite() {
        assert_eq!(ClockEdge::Rising.opposite(), ClockEdge::Falling);
        assert_eq!(ClockEdge::Falling.opposite(), ClockEdge::Rising);
    }

    #[test]
    fn clock_edge_double_opposite() {
        assert_eq!(ClockEdge::Rising.opposite().opposite(), ClockEdge::Rising);
        assert_eq!(ClockEdge::Falling.opposite().opposite(), ClockEdge::Falling);
    }

    #[test]
    fn clock_edge_display() {
        assert_eq!(format!("{}", ClockEdge::Rising), "RISING");
        assert_eq!(format!("{}", ClockEdge::Falling), "FALLING");
    }

    #[test]
    fn clock_edge_equality() {
        assert_eq!(ClockEdge::Rising, ClockEdge::Rising);
        assert_ne!(ClockEdge::Rising, ClockEdge::Falling);
    }

    #[test]
    fn clock_edge_copy() {
        let a = ClockEdge::Rising;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn clock_edge_hashing() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ClockEdge::Rising);
        set.insert(ClockEdge::Falling);
        set.insert(ClockEdge::Rising);
        assert_eq!(set.len(), 2);
    }
}
