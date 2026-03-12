//! Validators for the database-driven packer.
//!
//! Validators check whether two cells can be packed together by examining
//! their extracted tags and the current cluster state.

use super::extractor::ResourceConstraint;
use super::tagger::CellTagger;
use crate::context::Context;
use crate::netlist::CellId;

/// Trait for checking whether two cells can be packed together.
pub trait Validator {
    fn check(
        &self,
        ctx: &Context,
        tagger: &CellTagger,
        base_cell: CellId,
        new_cell: CellId,
    ) -> Result<(), String>;
}

/// Ensures cells sharing a wire drive the same net (with same inversion).
pub struct SharedWireValidator;

impl Validator for SharedWireValidator {
    fn check(
        &self,
        _ctx: &Context,
        tagger: &CellTagger,
        base: CellId,
        new: CellId,
    ) -> Result<(), String> {
        let (Some(base_tags), Some(new_tags)) = (tagger.get(base), tagger.get(new)) else {
            return Ok(());
        };

        for new_c in &new_tags.constraints {
            let ResourceConstraint::SharedWire {
                wire_index: nw_idx,
                tile_type: nw_tt,
                net: nw_net,
                inverted: nw_inv,
            } = new_c;

            for base_c in &base_tags.constraints {
                let ResourceConstraint::SharedWire {
                    wire_index: bw_idx,
                    tile_type: bw_tt,
                    net: bw_net,
                    inverted: bw_inv,
                } = base_c;

                if nw_idx == bw_idx && nw_tt == bw_tt {
                    if nw_net != bw_net {
                        return Err(format!(
                            "shared wire {} in tile type {}: conflicting nets",
                            nw_idx, nw_tt
                        ));
                    }
                    if nw_inv != bw_inv {
                        return Err(format!(
                            "shared wire {} in tile type {}: conflicting inversion",
                            nw_idx, nw_tt
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Ensures the tile has capacity for one more cell of this type.
pub struct SiteCapacityValidator;

impl Validator for SiteCapacityValidator {
    fn check(
        &self,
        ctx: &Context,
        tagger: &CellTagger,
        base: CellId,
        new: CellId,
    ) -> Result<(), String> {
        let new_cell = ctx.design.cell(new);
        let new_type_str = ctx.name_of(new_cell.cell_type);

        let Some(new_tags) = tagger.get(new) else {
            return Ok(());
        };

        if new_tags.compatible_tile_types.is_empty() {
            return Ok(());
        }

        // Count how many cells of the same type are already in the cluster
        let base_cell = ctx.design.cell(base);
        let cluster_root = base_cell.cluster.unwrap_or(base);

        let cluster_count = match ctx.design.clusters.get(&cluster_root) {
            Some(cluster) => cluster
                .members
                .iter()
                .filter(|&&m| ctx.design.cell(m).cell_type == new_cell.cell_type)
                .count(),
            None if ctx.design.cell(cluster_root).cell_type == new_cell.cell_type => 1,
            None => 0,
        };

        // Check against BEL count in any compatible tile type
        let chipdb = ctx.chipdb();
        let max_count = new_tags
            .compatible_tile_types
            .iter()
            .map(|&tt_idx| chipdb.bel_count_in_tile_type(tt_idx, new_type_str))
            .max()
            .unwrap_or(0);

        if cluster_count >= max_count {
            return Err(format!(
                "tile capacity exceeded for type {}: {} already placed, max {}",
                new_type_str, cluster_count, max_count
            ));
        }

        Ok(())
    }
}
