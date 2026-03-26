//! Shared legalization helpers: unbinding movable cells and placing cluster children.

use crate::chipdb::BelId;
use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::PlacerError;

/// Unbind all movable cells and their cluster children.
pub(crate) fn unbind_movable_cells(ctx: &mut Context, idx_to_cell: &[CellId]) {
    for &cell_id in idx_to_cell {
        let cell = ctx.design.cell(cell_id);
        if let Some(bel) = cell.bel {
            if !cell.bel_strength.is_locked() {
                ctx.unbind_bel(bel);
            }
        }
        if let Some(cluster) = ctx.design.clusters.get(&cell_id) {
            let children: Vec<_> = cluster.constr_children.clone();
            for child_id in children {
                let child = ctx.design.cell(child_id);
                if let Some(bel) = child.bel {
                    if !child.bel_strength.is_locked() {
                        ctx.unbind_bel(bel);
                    }
                }
            }
        }
    }
}

/// Place cluster children relative to the root BEL location.
///
/// Tries exact constraint position first, then any available BEL of matching type.
pub(crate) fn place_cluster_children(
    ctx: &mut Context,
    cell_id: CellId,
    root_bel: BelId,
) -> Result<(), PlacerError> {
    let cluster = match ctx.design.clusters.get(&cell_id) {
        Some(c) => c,
        None => return Ok(()),
    };
    let children: Vec<_> = cluster.constr_children.clone();
    let root_loc = ctx.bel(root_bel).loc();

    for child_id in children {
        let child = ctx.design.cell(child_id);
        let child_type = child.cell_type;
        let child_x = root_loc.x + child.constr_x;
        let child_y = root_loc.y + child.constr_y;

        let mut placed = false;

        let exact_candidates: Vec<_> = ctx
            .bels_for_bucket(child_type)
            .filter(|b| b.is_available() && b.loc().x == child_x && b.loc().y == child_y)
            .map(|b| b.id())
            .collect();
        for bel_id in exact_candidates {
            if ctx.bind_bel(bel_id, child_id, PlaceStrength::Placer) {
                placed = true;
                break;
            }
        }

        if !placed {
            let fallback_candidates: Vec<_> = ctx
                .bels_for_bucket(child_type)
                .filter(|b| b.is_available())
                .map(|b| b.id())
                .collect();
            for bel_id in fallback_candidates {
                if ctx.bind_bel(bel_id, child_id, PlaceStrength::Placer) {
                    placed = true;
                    break;
                }
            }
        }

        if !placed {
            return Err(PlacerError::PlacementFailed(format!(
                "Failed to place cluster child {}",
                ctx.name_of(ctx.design.cell(child_id).name)
            )));
        }
    }

    Ok(())
}
