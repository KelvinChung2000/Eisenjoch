use crate::types::{DelayT, IdString, PortType};

use super::{CellId, NetId};

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
    pub name: IdString,
    pub port_type: PortType,
    pub net: Option<NetId>,
    pub user_idx: Option<u32>,
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
}
