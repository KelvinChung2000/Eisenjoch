use crate::chipdb::{BelId, PipId, WireId};
use crate::common::PlaceStrength;
use crate::netlist::{CellId, NetId};
use log::warn;
use std::fmt;

use super::Context;

impl Context {
    #[inline]
    pub(crate) fn bel_slot(&self, bel: BelId) -> Option<&Option<CellId>> {
        let tile = usize::try_from(bel.tile()).ok()?;
        let index = usize::try_from(bel.index()).ok()?;
        self.bel_to_cell.get(tile, index)
    }

    #[inline]
    pub(crate) fn bel_slot_mut(&mut self, bel: BelId) -> Option<&mut Option<CellId>> {
        let tile = usize::try_from(bel.tile()).ok()?;
        let index = usize::try_from(bel.index()).ok()?;
        self.bel_to_cell.get_mut(tile, index)
    }

    /// The occupancy key for `wire`: its node's canonical wire. One node is
    /// one binding however many tiles it crosses, so occupancy grows with the
    /// routed nodes rather than with their length.
    #[inline]
    fn wire_key(&self, wire: WireId) -> Option<(u32, u32)> {
        let wire = self.chipdb.canonical_wire(wire);
        Some((
            u32::try_from(wire.tile()).ok()?,
            u32::try_from(wire.index()).ok()?,
        ))
    }

    /// Return the current binding of `wire`'s node, or `None` if unbound or
    /// the id is out of range. Any member of the node answers the same.
    #[inline]
    pub fn wire_binding(&self, wire: WireId) -> Option<(NetId, PlaceStrength)> {
        self.wire_to_net.get(&self.wire_key(wire)?).copied()
    }

    /// Number of bound wire entries. Memory for wire occupancy is this times
    /// one map entry.
    pub fn bound_wire_count(&self) -> usize {
        self.wire_to_net.len()
    }

    /// Return the current binding of a pip, or `None` if unbound or out of
    /// range.
    #[inline]
    pub(crate) fn pip_binding(&self, pip: PipId) -> Option<(NetId, PlaceStrength)> {
        let tile = u32::try_from(pip.tile()).ok()?;
        let index = u32::try_from(pip.index()).ok()?;
        self.pip_to_net.get(&(tile, index)).copied()
    }

    /// Bind a cell to a BEL.
    pub fn bind_bel(
        &mut self,
        bel: impl Into<BelId>,
        cell_idx: impl Into<CellId>,
        strength: PlaceStrength,
    ) -> bool {
        let bel = bel.into();
        let cell_idx = cell_idx.into();
        if self.bel_slot(bel).and_then(|slot| *slot).is_some() {
            warn!("bind_bel: bel {} already occupied", bel);
            return false;
        }
        let Some(slot) = self.bel_slot_mut(bel) else {
            warn!("bind_bel: bel {} out of range", bel);
            return false;
        };

        *slot = Some(cell_idx);

        let cell = self.design.cell_mut(cell_idx);
        cell.bel = Some(bel);
        cell.bel_strength = strength;

        true
    }

    /// Unbind a cell from its BEL.
    pub fn unbind_bel(&mut self, bel: impl Into<BelId>) {
        let bel = bel.into();
        if let Some(slot) = self.bel_slot_mut(bel) {
            if let Some(cell_idx) = slot.take() {
                let cell = self.design.cell_mut(cell_idx);
                cell.bel = None;
                cell.bel_strength = PlaceStrength::None;
            }
        }
    }

    /// Bind `cell` to `bel` after checking that the BEL exists, matches the
    /// cell's type and is free, that the cell is unbound, and that the tile is
    /// valid with the cell in place. Every error leaves the Context unchanged.
    pub fn try_bind_bel(
        &mut self,
        cell: CellId,
        bel: BelId,
        strength: PlaceStrength,
    ) -> Result<(), BelBindError> {
        let Some(&occupant) = self.bel_slot(bel) else {
            return Err(BelBindError::BelNotFound {
                cell: self.cell_name(cell),
                bel,
            });
        };
        self.check_bel_type(cell, bel)?;
        if let Some(bound_to) = self.design.cell(cell).bel {
            return Err(BelBindError::CellAlreadyBound {
                cell: self.cell_name(cell),
                bel: self.bel_location(bel),
                bound_to: self.bel_location(bound_to),
            });
        }
        if let Some(occupant) = occupant {
            return Err(BelBindError::BelOccupied {
                cell: self.cell_name(cell),
                bel: self.bel_location(bel),
                occupant: self.cell_name(occupant),
            });
        }
        let previous_strength = self.design.cell(cell).bel_strength;
        assert!(self.bind_bel(bel, cell, strength));
        if !self.is_bel_location_valid(bel) {
            self.unbind_bel(bel);
            self.design.cell_mut(cell).bel_strength = previous_strength;
            return Err(self.invalid_location(cell, bel));
        }
        Ok(())
    }

    /// Exchange the BELs of two placed cells, each keeping its own strength.
    /// Fails if either cell is unplaced, a cell's type does not match the
    /// other's BEL, or either tile is invalid after the exchange. Every error
    /// leaves the Context unchanged.
    pub fn try_swap_cells(&mut self, first: CellId, second: CellId) -> Result<(), BelBindError> {
        let first_bel = self.placed_bel(first)?;
        let second_bel = self.placed_bel(second)?;
        self.check_bel_type(first, second_bel)?;
        self.check_bel_type(second, first_bel)?;
        self.place_pair(first, second_bel, second, first_bel);
        for (cell, bel) in [(first, second_bel), (second, first_bel)] {
            if !self.is_bel_location_valid(bel) {
                self.place_pair(first, first_bel, second, second_bel);
                return Err(self.invalid_location(cell, bel));
            }
        }
        Ok(())
    }

    fn place_pair(&mut self, first: CellId, first_bel: BelId, second: CellId, second_bel: BelId) {
        for (cell, bel) in [(first, first_bel), (second, second_bel)] {
            *self
                .bel_slot_mut(bel)
                .expect("a placed cell's BEL is in range") = Some(cell);
            self.design.cell_mut(cell).bel = Some(bel);
        }
    }

    fn placed_bel(&self, cell: CellId) -> Result<BelId, BelBindError> {
        self.design
            .cell(cell)
            .bel
            .ok_or_else(|| BelBindError::CellNotPlaced {
                cell: self.cell_name(cell),
            })
    }

    fn check_bel_type(&self, cell: CellId, bel: BelId) -> Result<(), BelBindError> {
        let cell_type = self.name_of(self.design.cell(cell).cell_type);
        let bel_type = self.chipdb.bel_type(bel);
        if cell_type == bel_type {
            return Ok(());
        }
        Err(BelBindError::TypeMismatch {
            cell: self.cell_name(cell),
            bel: self.bel_location(bel),
            cell_type: cell_type.into(),
            bel_type: bel_type.into(),
        })
    }

    fn invalid_location(&self, cell: CellId, bel: BelId) -> BelBindError {
        BelBindError::InvalidLocation {
            cell: self.cell_name(cell),
            bel: self.bel_location(bel),
        }
    }

    fn cell_name(&self, cell: CellId) -> Box<str> {
        self.name_of(self.design.cell(cell).name).into()
    }

    fn bel_location(&self, bel: BelId) -> BelLocation {
        BelLocation {
            tile: self.chipdb.tile_name(bel.tile()).into(),
            bel: self.chipdb.bel_name(bel).into(),
        }
    }

    /// Bind `wire`'s node to a net.
    pub fn bind_wire(
        &mut self,
        wire: impl Into<WireId>,
        net_idx: impl Into<NetId>,
        strength: PlaceStrength,
    ) {
        if let Some(key) = self.wire_key(wire.into()) {
            self.wire_to_net.insert(key, (net_idx.into(), strength));
        }
    }

    /// Unbind `wire`'s node. No-op if not bound.
    pub fn unbind_wire(&mut self, wire: impl Into<WireId>) {
        if let Some(key) = self.wire_key(wire.into()) {
            self.wire_to_net.remove(&key);
        }
    }

    /// Bind `wire`'s node to `net`, or return `Err(wire)` if a different net
    /// already holds it. Rebinding the same net is idempotent.
    pub fn try_bind_wire_node(
        &mut self,
        wire: WireId,
        net_idx: NetId,
        strength: PlaceStrength,
    ) -> Result<(), WireId> {
        if let Some((owner, _)) = self.wire_binding(wire) {
            if owner != net_idx {
                return Err(wire);
            }
        }
        self.bind_wire(wire, net_idx, strength);
        Ok(())
    }

    /// Bind a PIP to a net.
    pub fn bind_pip(
        &mut self,
        pip: impl Into<PipId>,
        net_idx: impl Into<NetId>,
        strength: PlaceStrength,
    ) {
        let pip = pip.into();
        let net_idx = net_idx.into();
        let Ok(tile) = u32::try_from(pip.tile()) else {
            return;
        };
        let Ok(index) = u32::try_from(pip.index()) else {
            return;
        };
        self.pip_to_net.insert((tile, index), (net_idx, strength));
    }

    /// Unbind a PIP. O(1) hash-map removal; no-op if not bound.
    pub fn unbind_pip(&mut self, pip: impl Into<PipId>) {
        let pip = pip.into();
        let Ok(tile) = u32::try_from(pip.tile()) else {
            return;
        };
        let Ok(index) = u32::try_from(pip.index()) else {
            return;
        };
        self.pip_to_net.remove(&(tile, index));
    }
}

/// A BEL named by its tile and its name within the tile, displayed as `<tile>.<bel>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BelLocation {
    pub tile: Box<str>,
    pub bel: Box<str>,
}

impl fmt::Display for BelLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.tile, self.bel)
    }
}

/// Why [`Context::try_bind_bel`] or [`Context::try_swap_cells`] refused an edit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BelBindError {
    #[error("cell {cell} cannot bind to BEL {bel}, which is not in the chipdb. Pick a BEL the chipdb lists.")]
    BelNotFound { cell: Box<str>, bel: BelId },
    #[error("cell {cell} of type {cell_type} cannot bind to BEL {bel} of type {bel_type}. Pick a BEL of type {cell_type}.")]
    TypeMismatch {
        cell: Box<str>,
        bel: BelLocation,
        cell_type: Box<str>,
        bel_type: Box<str>,
    },
    #[error(
        "cell {cell} cannot bind to BEL {bel} while it is bound to {bound_to}. Unbind it first."
    )]
    CellAlreadyBound {
        cell: Box<str>,
        bel: BelLocation,
        bound_to: BelLocation,
    },
    #[error("cell {cell} cannot bind to BEL {bel}, which holds cell {occupant}. Unbind {occupant} or swap the two cells.")]
    BelOccupied {
        cell: Box<str>,
        bel: BelLocation,
        occupant: Box<str>,
    },
    #[error(
        "cell {cell} at BEL {bel} leaves its tile in an invalid configuration. Pick another BEL."
    )]
    InvalidLocation { cell: Box<str>, bel: BelLocation },
    #[error("cell {cell} is not placed, so it has no BEL to swap. Bind it first.")]
    CellNotPlaced { cell: Box<str> },
}
