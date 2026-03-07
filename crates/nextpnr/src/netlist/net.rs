use crate::types::{DelayT, IdString, PipId, PlaceStrength, Property, WireId};
use rustc_hash::FxHashMap;

use super::CellPin;

#[derive(Clone, Debug)]
pub struct PipMap {
    pub pip: Option<PipId>,
    pub strength: PlaceStrength,
}

pub struct NetInfo {
    pub name: IdString,
    pub(crate) driver: CellPin,
    pub(crate) users: Vec<CellPin>,
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
            driver: CellPin::INVALID,
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
        self.driver.is_valid()
    }

    #[inline]
    pub fn num_users(&self) -> usize {
        self.users.len()
    }

    #[inline]
    pub fn driver(&self) -> Option<CellPin> {
        self.driver.is_valid().then_some(self.driver)
    }

    #[inline]
    pub fn users(&self) -> &[CellPin] {
        &self.users
    }

    #[inline]
    pub fn set_driver_raw(&mut self, driver: CellPin) {
        self.driver = driver;
    }

    #[inline]
    pub fn add_user_raw(&mut self, user: CellPin) {
        self.users.push(user);
    }
}
