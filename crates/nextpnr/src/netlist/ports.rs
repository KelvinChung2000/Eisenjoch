use crate::types::{DelayT, IdString, PortType};

use super::{CellId, NetId};

/// A (cell, port) pair identifying a specific pin on a cell.
///
/// Netlist-level equivalent of `BelPin`. Used as a key in the timing engine
/// for arrival/required times, domain assignments, and path tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellPin {
    pub cell: CellId,
    pub port: IdString,
}

impl CellPin {
    #[inline]
    pub const fn new(cell: CellId, port: IdString) -> Self {
        Self { cell, port }
    }
}

#[derive(Clone, Debug)]
pub struct PortRef {
    pub cell: Option<CellId>,
    pub port: IdString,
    pub budget: DelayT,
}

impl PortRef {
    pub fn connected(cell: CellId, port: IdString, budget: DelayT) -> Self {
        Self {
            cell: Some(cell),
            port,
            budget,
        }
    }

    pub fn unconnected() -> Self {
        Self {
            cell: None,
            port: IdString::EMPTY,
            budget: 0,
        }
    }

    #[inline]
    pub fn is_connected(&self) -> bool {
        self.cell.is_some()
    }
}

#[derive(Clone, Debug)]
pub struct PortInfo {
    pub(crate) name: IdString,
    pub(crate) port_type: PortType,
    pub(crate) net: Option<NetId>,
    pub(crate) user_idx: Option<u32>,
}

impl PortInfo {
    pub fn new(name: IdString, port_type: PortType) -> Self {
        Self {
            name,
            port_type,
            net: None,
            user_idx: None,
        }
    }

    #[inline]
    pub fn is_connected(&self) -> bool {
        self.net.is_some()
    }

    #[inline]
    pub fn net(&self) -> Option<NetId> {
        self.net
    }

    #[inline]
    pub fn port_type(&self) -> PortType {
        self.port_type
    }

    #[inline]
    pub fn name(&self) -> IdString {
        self.name
    }

    #[inline]
    pub fn user_idx(&self) -> Option<u32> {
        self.user_idx
    }

    #[inline]
    pub fn set_net(&mut self, net: Option<NetId>) {
        self.net = net;
    }

    #[inline]
    pub fn set_user_idx(&mut self, user_idx: Option<u32>) {
        self.user_idx = user_idx;
    }

    #[inline]
    pub fn set_name(&mut self, name: IdString) {
        self.name = name;
    }
}
