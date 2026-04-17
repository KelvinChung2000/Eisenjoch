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

    /// Return the current binding of a wire, or `None` if unbound or the id is
    /// out of range. Sparse lookup: `O(1)` against the hash map.
    #[inline]
    pub(crate) fn wire_binding(&self, wire: WireId) -> Option<(NetId, PlaceStrength)> {
        let tile = u32::try_from(wire.tile()).ok()?;
        let index = u32::try_from(wire.index()).ok()?;
        self.wire_to_net.get(&(tile, index)).copied()
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

    /// Bind a wire to a net.
    pub fn bind_wire(
        &mut self,
        wire: impl Into<WireId>,
        net_idx: impl Into<NetId>,
        strength: PlaceStrength,
    ) {
        let wire = wire.into();
        let net_idx = net_idx.into();
        let Ok(tile) = u32::try_from(wire.tile()) else { return };
        let Ok(index) = u32::try_from(wire.index()) else { return };
        self.wire_to_net.insert((tile, index), (net_idx, strength));
    }

    /// Unbind a wire. O(1) hash-map removal; no-op if not bound.
    pub fn unbind_wire(&mut self, wire: impl Into<WireId>) {
        let wire = wire.into();
        let Ok(tile) = u32::try_from(wire.tile()) else { return };
        let Ok(index) = u32::try_from(wire.index()) else { return };
        self.wire_to_net.remove(&(tile, index));
    }

    /// Bind `wire` and all node-equivalent wires to `net`. Returns `Err` if
    /// any wire in the node is already bound to a different net. Either all
    /// bindings succeed or none do — no partial application.
    ///
    /// Binding the same net to a wire that already holds that binding is
    /// treated as idempotent (no error, no change).
    pub fn try_bind_wire_node(
        &mut self,
        wire: WireId,
        net_idx: NetId,
        strength: PlaceStrength,
    ) -> Result<(), WireId> {
        // Collect wire + all node-equivalent wires.
        let mut wires: Vec<WireId> = Vec::with_capacity(8);
        wires.push(wire);
        self.chipdb().node_wires_cb(wire, |nw| {
            wires.push(nw);
        });

        // Phase 1: verify no wire is bound to a different net.
        for &w in &wires {
            if let Some((owner, _)) = self.wire_binding(w) {
                if owner != net_idx {
                    return Err(w);
                }
            }
        }

        // Phase 2: apply all bindings.
        for &w in &wires {
            let Ok(tile) = u32::try_from(w.tile()) else { continue };
            let Ok(index) = u32::try_from(w.index()) else { continue };
            self.wire_to_net.insert((tile, index), (net_idx, strength));
        }

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
        let Ok(tile) = u32::try_from(pip.tile()) else { return };
        let Ok(index) = u32::try_from(pip.index()) else { return };
        self.pip_to_net.insert((tile, index), (net_idx, strength));
    }

    /// Unbind a PIP. O(1) hash-map removal; no-op if not bound.
    pub fn unbind_pip(&mut self, pip: impl Into<PipId>) {
        let pip = pip.into();
        let Ok(tile) = u32::try_from(pip.tile()) else { return };
        let Ok(index) = u32::try_from(pip.index()) else { return };
        self.pip_to_net.remove(&(tile, index));
    }
}
