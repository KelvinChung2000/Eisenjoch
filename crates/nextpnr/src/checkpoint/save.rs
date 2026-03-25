//! Checkpoint save and build logic.

use crate::context::Context;

use super::types::{CellPlacement, Checkpoint, CheckpointError, NetRoute, CHECKPOINT_VERSION};
use super::restore::compute_fingerprint;

/// Save the current placement and routing state as a checkpoint.
pub fn save(ctx: &Context, path: &std::path::Path) -> Result<(), CheckpointError> {
    let checkpoint = build_checkpoint(ctx);
    checkpoint.save_to_file(path)
}

/// Build a checkpoint from the current context state.
pub fn build_checkpoint(ctx: &Context) -> Checkpoint {
    let chipdb = ctx.chipdb();
    let placements: Vec<CellPlacement> = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(_, cell)| {
            let bel = cell.bel?;
            Some(CellPlacement {
                cell_name: ctx.name_of(cell.name).to_owned(),
                cell_type: ctx.name_of(cell.cell_type).to_owned(),
                bel_tile: bel.tile(),
                bel_index: bel.index(),
                bel_name: chipdb.bel_name(bel).to_owned(),
                tile_name: chipdb.tile_name(bel.tile()),
                tile_type: chipdb.tile_type_name(bel.tile()).to_owned(),
                strength: cell.bel_strength as u8,
            })
        })
        .collect();

    let routes: Vec<NetRoute> = ctx
        .design
        .iter_alive_nets()
        .filter(|(_, net)| !net.wires.is_empty())
        .map(|(_, net)| {
            let mut source_wire_tile = 0i32;
            let mut source_wire_index = 0i32;
            let mut source_wire_name = String::new();
            let mut pips = Vec::new();
            let mut pip_names = Vec::new();

            for (&wire, pm) in &net.wires {
                match pm.pip {
                    None => {
                        source_wire_tile = wire.tile();
                        source_wire_index = wire.index();
                        source_wire_name = format!(
                            "{}/{}",
                            chipdb.tile_name(wire.tile()),
                            chipdb.wire_name(wire),
                        );
                    }
                    Some(pip) => {
                        pips.push((pip.tile(), pip.index()));
                        let src = chipdb.pip_src_wire(pip);
                        let dst = chipdb.pip_dst_wire(pip);
                        pip_names.push(format!(
                            "{}/{}.{}",
                            chipdb.tile_name(pip.tile()),
                            chipdb.wire_name(dst),
                            chipdb.wire_name(src),
                        ));
                    }
                }
            }

            NetRoute {
                net_name: ctx.name_of(net.name).to_owned(),
                source_wire_tile,
                source_wire_index,
                source_wire_name,
                pips,
                pip_names,
            }
        })
        .collect();

    Checkpoint {
        version: CHECKPOINT_VERSION,
        placements,
        routes,
        fingerprint: compute_fingerprint(ctx),
    }
}
