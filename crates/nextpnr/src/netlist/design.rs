use crate::types::IdString;
use rustc_hash::FxHashMap;

use super::{CellEditor, CellIdx, CellInfo, Cluster, HierarchicalCell, NetEditor, NetIdx, NetInfo};

pub struct Design {
    cells: FxHashMap<IdString, CellIdx>,
    cell_store: Vec<Option<CellInfo>>,
    cell_generation: Vec<u16>,
    free_cell_slots: Vec<u32>,

    nets: FxHashMap<IdString, NetIdx>,
    net_store: Vec<Option<NetInfo>>,
    net_generation: Vec<u16>,
    free_net_slots: Vec<u32>,

    pub hierarchy: FxHashMap<IdString, HierarchicalCell>,

    pub clusters: FxHashMap<CellIdx, Cluster>,

    pub top_module: IdString,
}

impl Design {
    pub fn new() -> Self {
        Self {
            cells: FxHashMap::default(),
            cell_store: Vec::new(),
            cell_generation: Vec::new(),
            free_cell_slots: Vec::new(),
            nets: FxHashMap::default(),
            net_store: Vec::new(),
            net_generation: Vec::new(),
            free_net_slots: Vec::new(),
            hierarchy: FxHashMap::default(),
            clusters: FxHashMap::default(),
            top_module: IdString::EMPTY,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty() && self.nets.is_empty()
    }

    #[inline]
    pub fn cell_slots_len(&self) -> usize {
        self.cell_store.len()
    }

    #[inline]
    pub fn net_slots_len(&self) -> usize {
        self.net_store.len()
    }

    pub fn add_cell(&mut self, name: IdString, cell_type: IdString) -> CellIdx {
        assert!(
            !self.cells.contains_key(&name),
            "cell already exists in design"
        );
        let idx = if let Some(slot_idx) = self.free_cell_slots.pop() {
            self.cell_store[slot_idx as usize] = Some(CellInfo::new(name, cell_type));
            CellIdx::new(slot_idx, self.cell_generation[slot_idx as usize])
        } else {
            let slot_idx = self.cell_store.len() as u32;
            self.cell_store.push(Some(CellInfo::new(name, cell_type)));
            self.cell_generation.push(0);
            let idx = CellIdx::new(slot_idx, 0);
            idx
        };
        self.cells.insert(name, idx);
        idx
    }

    #[inline]
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub fn iter_cells(&self) -> impl Iterator<Item = (CellIdx, &CellInfo)> + '_ {
        self.cell_store.iter().enumerate().filter_map(|(slot, cell_slot)| {
            let cell = cell_slot.as_ref()?;
            let slot_u32 = u32::try_from(slot).ok()?;
            let generation = *self.cell_generation.get(slot)?;
            Some((CellIdx::new(slot_u32, generation), cell))
        })
    }

    #[inline]
    pub fn iter_alive_cells(&self) -> impl Iterator<Item = (CellIdx, &CellInfo)> + '_ {
        self.iter_cells().filter(|(_, cell)| cell.alive)
    }

    #[inline]
    pub fn iter_cell_indices(&self) -> impl Iterator<Item = CellIdx> + '_ {
        self.cells.values().copied()
    }

    #[inline]
    pub fn cell_idx_at_slot(&self, slot: usize) -> Option<CellIdx> {
        if self.cell_store.get(slot).and_then(|c| c.as_ref()).is_none() {
            return None;
        }
        let slot_u32 = u32::try_from(slot).ok()?;
        let generation = *self.cell_generation.get(slot)?;
        Some(CellIdx::new(slot_u32, generation))
    }

    #[inline]
    pub fn cell(&self, idx: CellIdx) -> &CellInfo {
        let slot = idx.slot() as usize;
        assert_eq!(self.cell_generation[slot], idx.generation(), "stale CellIdx generation");
        self.cell_store[slot].as_ref().expect("dead CellIdx slot")
    }

    #[inline]
    pub(crate) fn cell_mut(&mut self, idx: CellIdx) -> &mut CellInfo {
        let slot = idx.slot() as usize;
        assert_eq!(self.cell_generation[slot], idx.generation(), "stale CellIdx generation");
        self.cell_store[slot].as_mut().expect("dead CellIdx slot")
    }

    pub fn cell_by_name(&self, name: IdString) -> Option<CellIdx> {
        self.cells.get(&name).copied()
    }

    pub fn remove_cell(&mut self, name: IdString) {
        if let Some(idx) = self.cells.remove(&name) {
            let slot = idx.slot() as usize;
            self.cell_store[slot] = None;
            self.cell_generation[slot] = self.cell_generation[slot].wrapping_add(1);
            self.free_cell_slots.push(slot as u32);
        }
    }

    pub fn add_net(&mut self, name: IdString) -> NetIdx {
        assert!(
            !self.nets.contains_key(&name),
            "net already exists in design"
        );
        let idx = if let Some(slot_idx) = self.free_net_slots.pop() {
            self.net_store[slot_idx as usize] = Some(NetInfo::new(name));
            NetIdx::new(slot_idx, self.net_generation[slot_idx as usize])
        } else {
            let slot_idx = self.net_store.len() as u32;
            self.net_store.push(Some(NetInfo::new(name)));
            self.net_generation.push(0);
            let idx = NetIdx::new(slot_idx, 0);
            idx
        };
        self.nets.insert(name, idx);
        idx
    }

    #[inline]
    pub fn num_nets(&self) -> usize {
        self.nets.len()
    }

    #[inline]
    pub fn iter_nets(&self) -> impl Iterator<Item = (NetIdx, &NetInfo)> + '_ {
        self.net_store.iter().enumerate().filter_map(|(slot, net_slot)| {
            let net = net_slot.as_ref()?;
            let slot_u32 = u32::try_from(slot).ok()?;
            let generation = *self.net_generation.get(slot)?;
            Some((NetIdx::new(slot_u32, generation), net))
        })
    }

    #[inline]
    pub fn iter_alive_nets(&self) -> impl Iterator<Item = (NetIdx, &NetInfo)> + '_ {
        self.iter_nets().filter(|(_, net)| net.alive)
    }

    #[inline]
    pub fn iter_net_indices(&self) -> impl Iterator<Item = NetIdx> + '_ {
        self.nets.values().copied()
    }

    #[inline]
    pub fn net_idx_at_slot(&self, slot: usize) -> Option<NetIdx> {
        if self.net_store.get(slot).and_then(|n| n.as_ref()).is_none() {
            return None;
        }
        let slot_u32 = u32::try_from(slot).ok()?;
        let generation = *self.net_generation.get(slot)?;
        Some(NetIdx::new(slot_u32, generation))
    }

    #[inline]
    pub fn net(&self, idx: NetIdx) -> &NetInfo {
        let slot = idx.slot() as usize;
        assert_eq!(self.net_generation[slot], idx.generation(), "stale NetIdx generation");
        self.net_store[slot].as_ref().expect("dead NetIdx slot")
    }

    #[inline]
    pub(crate) fn net_mut(&mut self, idx: NetIdx) -> &mut NetInfo {
        let slot = idx.slot() as usize;
        assert_eq!(self.net_generation[slot], idx.generation(), "stale NetIdx generation");
        self.net_store[slot].as_mut().expect("dead NetIdx slot")
    }

    pub fn cell_edit(&mut self, idx: CellIdx) -> CellEditor<'_> {
        CellEditor::new(self.cell_mut(idx))
    }

    pub fn net_edit(&mut self, idx: NetIdx) -> NetEditor<'_> {
        NetEditor::new(self.net_mut(idx))
    }

    pub fn net_by_name(&self, name: IdString) -> Option<NetIdx> {
        self.nets.get(&name).copied()
    }

    pub fn rename_net(&mut self, net_idx: NetIdx, new_name: IdString) {
        let old_name = self.net(net_idx).name;
        if old_name == new_name {
            return;
        }

        if let Some(existing_idx) = self.net_by_name(new_name) {
            assert_eq!(
                existing_idx, net_idx,
                "cannot rename net to an already-used name"
            );
        }

        self.net_mut(net_idx).name = new_name;
        self.nets.remove(&old_name);
        self.nets.insert(new_name, net_idx);
    }

    pub fn remove_net(&mut self, name: IdString) {
        if let Some(idx) = self.nets.remove(&name) {
            let slot = idx.slot() as usize;
            self.net_store[slot] = None;
            self.net_generation[slot] = self.net_generation[slot].wrapping_add(1);
            self.free_net_slots.push(slot as u32);
        }
    }
}

impl Default for Design {
    fn default() -> Self {
        Self::new()
    }
}
