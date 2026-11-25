use std::fmt;

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
