use crate::chipdb::ChipDb;
use crate::common::{IdString, IdStringPool, IntoIdString};
use crate::netlist::{CellId, CellPin, Design, NetId};
use crate::netlist::Property;
use rustc_hash::FxHashMap;

use super::views::{Bel, BelPin, BelPinView, Cell, CellPinView, IdStringView, Net, Pip, Wire};
use super::Context;

impl Context {
    /// Intern a string, returning its IdString handle.
    #[inline]
    pub fn id(&self, s: impl IntoIdString) -> IdString {
        s.into_id(&self.id_pool)
    }

    /// Look up the string for an IdString handle.
    ///
    /// Returns `"<unknown>"` if the index is out of range.
    #[inline]
    pub fn name_of(&self, id: IdString) -> &str {
        self.id_pool.lookup(id).unwrap_or("<unknown>")
    }

    #[inline]
    pub fn chipdb(&self) -> &ChipDb {
        &self.chipdb
    }

    /// Split borrow: returns mutable design + immutable chipdb + immutable id pool.
    pub fn packer_parts(&mut self) -> (&mut Design, &ChipDb, &IdStringPool) {
        (&mut self.design, &self.chipdb, &self.id_pool)
    }

    #[inline]
    pub fn rng(&self) -> &super::DeterministicRng {
        &self.rng
    }

    #[inline]
    pub fn rng_mut(&mut self) -> &mut super::DeterministicRng {
        &mut self.rng
    }

    #[inline]
    pub fn reseed_rng(&mut self, seed: u64) {
        self.rng = super::DeterministicRng::new(seed);
    }

    #[inline]
    pub fn settings(&self) -> &FxHashMap<IdString, Property> {
        &self.settings
    }

    #[inline]
    pub fn settings_mut(&mut self) -> &mut FxHashMap<IdString, Property> {
        &mut self.settings
    }

    #[inline]
    pub fn verbose(&self) -> bool {
        self.verbose
    }

    #[inline]
    pub fn debug(&self) -> bool {
        self.debug
    }

    #[inline]
    pub fn force(&self) -> bool {
        self.force
    }

    /// Create a lazy read-only view over an interned string handle.
    #[inline]
    pub fn id_ref(&self, id: IdString) -> IdStringView<'_> {
        IdStringView::new(self, id)
    }

    #[inline]
    pub fn bel(&self, bel: crate::chipdb::BelId) -> Bel<'_> {
        Bel::new(self, bel)
    }

    #[inline]
    pub fn bel_pin(&self, pin: BelPin) -> BelPinView<'_> {
        pin.view(self)
    }

    #[inline]
    pub fn bels(&self) -> impl Iterator<Item = Bel<'_>> {
        self.chipdb.bels().map(|bel| self.bel(bel))
    }

    #[inline]
    pub fn wire(&self, wire: crate::chipdb::WireId) -> Wire<'_> {
        Wire::new(self, wire)
    }

    #[inline]
    pub fn wires(&self) -> impl Iterator<Item = Wire<'_>> + '_ {
        self.chipdb.wires().map(|wire| self.wire(wire))
    }

    #[inline]
    pub fn pip(&self, pip: crate::chipdb::PipId) -> Pip<'_> {
        Pip::new(self, pip)
    }

    #[inline]
    pub fn pips(&self) -> impl Iterator<Item = Pip<'_>> + '_ {
        self.chipdb.pips().map(|pip| self.pip(pip))
    }

    #[inline]
    pub fn net(&self, net_idx: NetId) -> Net<'_> {
        Net::new(self, net_idx)
    }

    #[inline]
    pub fn nets(&self) -> impl Iterator<Item = Net<'_>> {
        self.design.iter_net_indices().map(|net_idx| self.net(net_idx))
    }

    #[inline]
    pub fn net_by_name(&self, net_name: IdString) -> Option<Net<'_>> {
        self.design.net_by_name(net_name).map(|net_idx| self.net(net_idx))
    }

    #[inline]
    pub fn cell(&self, cell_idx: CellId) -> Cell<'_> {
        Cell::new(self, cell_idx)
    }

    #[inline]
    pub fn cell_pin(&self, pin: CellPin) -> CellPinView<'_> {
        pin.view(self)
    }

    #[inline]
    pub fn cells(&self) -> impl Iterator<Item = Cell<'_>> {
        self.design.iter_cell_indices().map(|cell_idx| self.cell(cell_idx))
    }

    #[inline]
    pub fn cell_by_name(&self, cell_name: IdString) -> Option<Cell<'_>> {
        self.design.cell_by_name(cell_name).map(|cell_idx| self.cell(cell_idx))
    }

    /// Generate a resource utilization report.
    pub fn utilization_report(&self) -> crate::metrics::UtilizationReport {
        crate::metrics::utilization_report(self)
    }

    /// Compute spatial placement density using a sliding window.
    pub fn placement_density(&self, window: i32) -> crate::metrics::DensityReport {
        crate::metrics::placement_density(self, window)
    }

    /// Estimate routing congestion using edge-based demand.
    pub fn estimate_congestion(&self, threshold: f64) -> crate::metrics::CongestionReport {
        crate::metrics::estimate_congestion(self, threshold)
    }

    /// Find the wire connected to a specific BEL pin.
    pub fn bel_pin_wire(&self, bp: BelPin) -> Option<Wire<'_>> {
        let port_name = self.name_of(bp.port());
        let bel_info = self.chipdb.bel_info(bp.bel());
        bel_info.pins.get().iter().find_map(|pin| {
            let (name_constid, wire_idx, _dir) = self.chipdb.bel_pin_fields(pin);
            let pin_name = self.chipdb.constid_str(name_constid).unwrap_or("");
            (pin_name == port_name)
                .then(|| Wire::new(self, crate::chipdb::WireId::new(bp.bel().tile(), wire_idx)))
        })
    }
}
