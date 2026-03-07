use crate::chipdb::{BelId, PipId, WireId};
use crate::common::PlaceStrength;
use crate::netlist::{CellId, NetId};
use log::warn;

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

    #[inline]
    pub(crate) fn wire_slot(&self, wire: WireId) -> Option<&Option<(NetId, PlaceStrength)>> {
        let tile = usize::try_from(wire.tile()).ok()?;
        let index = usize::try_from(wire.index()).ok()?;
        self.wire_to_net.get(tile, index)
    }

    #[inline]
    pub(crate) fn wire_slot_mut(
        &mut self,
        wire: WireId,
    ) -> Option<&mut Option<(NetId, PlaceStrength)>> {
        let tile = usize::try_from(wire.tile()).ok()?;
        let index = usize::try_from(wire.index()).ok()?;
        self.wire_to_net.get_mut(tile, index)
    }

    #[inline]
    pub(crate) fn pip_slot(&self, pip: PipId) -> Option<&Option<(NetId, PlaceStrength)>> {
        let tile = usize::try_from(pip.tile()).ok()?;
        let index = usize::try_from(pip.index()).ok()?;
        self.pip_to_net.get(tile, index)
    }

    #[inline]
    pub(crate) fn pip_slot_mut(
        &mut self,
        pip: PipId,
    ) -> Option<&mut Option<(NetId, PlaceStrength)>> {
        let tile = usize::try_from(pip.tile()).ok()?;
        let index = usize::try_from(pip.index()).ok()?;
        self.pip_to_net.get_mut(tile, index)
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

    /// Bind a wire to a net.
    pub fn bind_wire(
        &mut self,
        wire: impl Into<WireId>,
        net_idx: impl Into<NetId>,
        strength: PlaceStrength,
    ) {
        let wire = wire.into();
        let net_idx = net_idx.into();
        if let Some(slot) = self.wire_slot_mut(wire) {
            *slot = Some((net_idx, strength));
        }
    }

    /// Unbind a wire.
    pub fn unbind_wire(&mut self, wire: impl Into<WireId>) {
        let wire = wire.into();
        if let Some(slot) = self.wire_slot_mut(wire) {
            *slot = None;
        }
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
        if let Some(slot) = self.pip_slot_mut(pip) {
            *slot = Some((net_idx, strength));
        }
    }

    /// Unbind a PIP.
    pub fn unbind_pip(&mut self, pip: impl Into<PipId>) {
        let pip = pip.into();
        if let Some(slot) = self.pip_slot_mut(pip) {
            *slot = None;
        }
    }
}
