//! CellTagger: orchestrates extraction and validation for the database-driven packer.

use super::extractor::{CellTags, Extractor, SharedWireExtractor, TileTypeExtractor};
use super::validator::{SharedWireValidator, SiteCapacityValidator, Validator};
use crate::context::Context;
use crate::netlist::CellId;
use rustc_hash::FxHashMap;

pub struct CellTagger {
    tags: FxHashMap<CellId, CellTags>,
    extractors: Vec<Box<dyn Extractor>>,
    validators: Vec<Box<dyn Validator>>,
}

impl CellTagger {
    pub fn new() -> Self {
        Self {
            tags: FxHashMap::default(),
            extractors: vec![
                Box::new(TileTypeExtractor),
                Box::new(SharedWireExtractor),
            ],
            validators: vec![
                Box::new(SharedWireValidator),
                Box::new(SiteCapacityValidator),
            ],
        }
    }

    /// Run all extractors on a cell, caching results.
    pub fn tag_cell(&mut self, ctx: &Context, cell: CellId) {
        let mut tags = CellTags::default();
        for extractor in &self.extractors {
            extractor.extract(ctx, cell, &mut tags);
        }
        self.tags.insert(cell, tags);
    }

    /// Run all extractors on all alive cells.
    pub fn tag_all(&mut self, ctx: &Context) {
        let cells: Vec<CellId> = ctx
            .design
            .iter_cell_indices()
            .filter(|&idx| ctx.design.cell(idx).alive)
            .collect();
        for cell in cells {
            self.tag_cell(ctx, cell);
        }
    }

    /// Check if two cells can be packed together.
    pub fn check_packing(
        &self,
        ctx: &Context,
        base: CellId,
        new: CellId,
    ) -> Result<(), String> {
        for validator in &self.validators {
            validator.check(ctx, self, base, new)?;
        }
        Ok(())
    }

    pub fn get(&self, cell: CellId) -> Option<&CellTags> {
        self.tags.get(&cell)
    }
}
