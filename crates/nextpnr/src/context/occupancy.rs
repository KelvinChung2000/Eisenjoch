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

    /// The occupancy key for `wire`: its node's canonical wire. One node is
    /// one binding however many tiles it crosses, so occupancy grows with the
    /// routed nodes rather than with their length.
    #[inline]
    fn wire_key(&self, wire: WireId) -> Option<(u32, u32)> {
        let wire = self.chipdb.canonical_wire(wire);
        Some((u32::try_from(wire.tile()).ok()?, u32::try_from(wire.index()).ok()?))
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
