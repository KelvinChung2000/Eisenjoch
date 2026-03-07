use crate::types::IdString;
use rustc_hash::FxHashMap;

#[derive(Clone, Debug)]
pub struct HierarchicalNet {
    pub name: IdString,
    pub flat_net: IdString,
}

pub struct HierarchicalCell {
    pub name: IdString,
    pub cell_type: IdString,
    pub parent: IdString,
    pub fullpath: IdString,
    pub hier_cells: FxHashMap<IdString, IdString>,
    pub leaf_cells: FxHashMap<IdString, IdString>,
    pub nets: FxHashMap<IdString, HierarchicalNet>,
}

impl HierarchicalCell {
    pub fn new(name: IdString, cell_type: IdString) -> Self {
        Self {
            name,
            cell_type,
            parent: IdString::EMPTY,
            fullpath: IdString::EMPTY,
            hier_cells: FxHashMap::default(),
            leaf_cells: FxHashMap::default(),
            nets: FxHashMap::default(),
        }
    }
}
