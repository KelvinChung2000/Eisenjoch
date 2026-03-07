use crate::common::IdString;
use crate::timing::DelayT;

use super::{CellId, NetId, PortType};

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
    pub const INVALID: Self = Self {
        cell: CellId::NONE,
        port: IdString::EMPTY,
    };

    #[inline]
    pub const fn new(cell: CellId, port: IdString) -> Self {
        Self { cell, port }
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.cell.is_some() && !self.port.is_empty()
    }

    #[inline]
    pub const fn is_connected(self) -> bool {
        self.is_valid()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PortData {
    pub(crate) port_type: PortType,
    pub(crate) net: Option<NetId>,
    pub(crate) user_idx: Option<u32>,
    pub(crate) budget: DelayT,
}

impl PortData {
    pub(crate) fn new(port_type: PortType) -> Self {
        Self {
            port_type,
            net: None,
            user_idx: None,
            budget: 0,
        }
    }

    #[inline]
    pub(crate) fn is_connected(&self) -> bool {
        self.net.is_some()
    }

    #[inline]
    pub(crate) fn net(&self) -> Option<NetId> {
        self.net
    }

    #[inline]
    pub(crate) fn port_type(&self) -> PortType {
        self.port_type
    }

    #[inline]
    pub(crate) fn user_idx(&self) -> Option<u32> {
        self.user_idx
    }

    #[inline]
    pub(crate) fn budget(&self) -> DelayT {
        self.budget
    }

    #[inline]
    pub(crate) fn set_net(&mut self, net: Option<NetId>) {
        self.net = net;
    }

    #[inline]
    pub(crate) fn set_user_idx(&mut self, user_idx: Option<u32>) {
        self.user_idx = user_idx;
    }

    #[inline]
    pub(crate) fn set_budget(&mut self, budget: DelayT) {
        self.budget = budget;
    }
}
