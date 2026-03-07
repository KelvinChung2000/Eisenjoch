use crate::common::IdString;

use super::CellId;

pub struct Cluster {
    pub root: CellId,
    pub members: Vec<CellId>,
    pub ports: Vec<(IdString, IdString, i32)>,
}

impl Cluster {
    pub fn new(root: CellId) -> Self {
        Self {
            root,
            members: vec![root],
            ports: Vec::new(),
        }
    }

    pub fn add_member(&mut self, cell: CellId) {
        if !self.members.contains(&cell) {
            self.members.push(cell);
        }
    }
}
