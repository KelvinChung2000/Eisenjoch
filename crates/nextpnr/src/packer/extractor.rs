//! Cell metadata extractors for the database-driven packer.
//!
//! Extractors analyze cells and the chipdb to produce typed constraints
//! and compatibility information used by validators during packing.

use crate::context::Context;
use crate::netlist::{CellId, NetId};

/// Typed resource constraint (NOT stringly-typed).
#[derive(Debug, Clone)]
pub enum ResourceConstraint {
    /// Cell uses a shared wire; net must agree across all users of that wire.
    SharedWire {
        wire_index: i32,
        tile_type: i32,
        net: NetId,
        inverted: bool,
    },
}

/// Extracted metadata for a single cell.
#[derive(Debug, Default)]
pub struct CellTags {
    pub constraints: Vec<ResourceConstraint>,
    pub compatible_tile_types: Vec<i32>,
}

/// Trait for extracting cell metadata from the chipdb.
pub trait Extractor {
    fn extract(&self, ctx: &Context, cell: CellId, tags: &mut CellTags);
}

/// Extracts compatible tile types for a cell based on its type.
pub struct TileTypeExtractor;

impl Extractor for TileTypeExtractor {
    fn extract(&self, ctx: &Context, cell: CellId, tags: &mut CellTags) {
        let cell_type = ctx.design.cell(cell).cell_type;
        let type_str = ctx.name_of(cell_type);
        tags.compatible_tile_types = ctx.chipdb().compatible_tile_types_for_bel_type(type_str);
    }
}

/// Extracts shared wire constraints from the chipdb topology.
pub struct SharedWireExtractor;

impl Extractor for SharedWireExtractor {
    fn extract(&self, ctx: &Context, cell: CellId, tags: &mut CellTags) {
        let cell_info = ctx.design.cell(cell);
        let chipdb = ctx.chipdb();

        for &tile_type_idx in &tags.compatible_tile_types {
            let shared = chipdb.shared_wires_in_tile_type(tile_type_idx);
            for (wire_idx, bel_pins) in &shared {
                for &(_bel_idx, pin_constid) in bel_pins {
                    let Some(pin_str) = chipdb.constid_str(pin_constid) else {
                        continue;
                    };
                    let pin_name = ctx.id(pin_str);

                    if let Some(net_idx) = cell_info.port_net(pin_name) {
                        let neg_key = ctx.id(&format!("NEG_{}", pin_str));
                        let inverted = cell_info
                            .params
                            .get(&neg_key)
                            .and_then(|p| p.as_int())
                            .is_some_and(|v| v != 0);
                        tags.constraints.push(ResourceConstraint::SharedWire {
                            wire_index: *wire_idx,
                            tile_type: tile_type_idx,
                            net: net_idx,
                            inverted,
                        });
                    }
                }
            }
        }
    }
}
