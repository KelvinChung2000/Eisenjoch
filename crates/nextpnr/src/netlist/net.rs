use crate::types::{DelayT, IdString, PipId, PlaceStrength, Property, WireId};
use rustc_hash::FxHashMap;

use super::PortRef;

#[derive(Clone, Debug)]
pub struct PipMap {
    pub pip: Option<PipId>,
    pub strength: PlaceStrength,
}

pub struct NetInfo {
    pub name: IdString,
    pub driver: PortRef,
    pub users: Vec<PortRef>,
    pub attrs: FxHashMap<IdString, Property>,
    pub wires: FxHashMap<WireId, PipMap>,
    pub clock_constraint: DelayT,
    pub region: Option<u32>,
    pub alive: bool,
}

impl NetInfo {
    pub fn new(name: IdString) -> Self {
        Self {
            name,
            driver: PortRef::unconnected(),
            users: Vec::new(),
            attrs: FxHashMap::default(),
            wires: FxHashMap::default(),
            clock_constraint: 0,
            region: None,
            alive: true,
        }
    }

    #[inline]
    pub fn has_driver(&self) -> bool {
        self.driver.is_connected()
    }

    #[inline]
    pub fn num_users(&self) -> usize {
        self.users.len()
    }
}
