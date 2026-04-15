//! Packer pipeline: main entry point and rule application logic.

use crate::context::Context;
use crate::netlist::{CellId, Cluster};
use crate::plugin::{PackerPlugin, PluginContext};

use super::rules::{get_packing_rules, PackingRule};
use super::tagger::CellTagger;
use super::{passes, PackerError};

/// Run the packer on the design.
///
/// If a plugin is provided, delegates to it. Otherwise uses the built-in
/// database-driven packer.
pub fn pack(ctx: &mut Context, plugin: Option<&mut dyn PackerPlugin>) -> Result<(), PackerError> {
    if let Some(plugin) = plugin {
        let (design, chipdb, id_pool) = ctx.packer_parts();
        let mut plugin_ctx = PluginContext::new(design, chipdb, id_pool);
        plugin.pack(&mut plugin_ctx).map_err(PackerError::Plugin)?;
    } else {
        pack_default(ctx)?;
    }
    Ok(())
}

/// The built-in database-driven packer.
///
/// Phase 0: Architecture-generic pre-passes (constants, IO)
/// Phase 1: Extract cell metadata from chipdb
/// Phase 2: Load/derive packing rules
/// Phase 3: Apply rules with validation
/// Phase 4: Remaining cells passthrough
pub fn pack_default(ctx: &mut Context) -> Result<(), PackerError> {
    // Ensure BEL buckets are populated so pack_constants can detect arch-specific types.
    ctx.populate_bel_buckets();

    // Phase 0: Pre-passes (architecture-generic)
    passes::pack_constants(ctx)?;
    passes::pack_io(ctx)?;
    passes::pack_bufg(ctx)?;

    // Phase 1: Extract cell metadata
    let mut tagger = CellTagger::new();
    tagger.tag_all(ctx);

    // Phase 2: Get packing rules (from chipdb extra_data or topology derivation)
    let rules = get_packing_rules(ctx);

    // Phase 3: Apply rules
    let mut constrain_count = 0;
    let mut fail_count = 0;

    for rule in &rules {
        let matches: Vec<_> = ctx
            .design
            .iter_alive_nets()
            .filter_map(|(_, net)| {
                let driver = net.driver()?;
                let drv_cell = ctx.design.cell(driver.cell);
                if drv_cell.cell_type != rule.driver.cell_type || driver.port != rule.driver.port {
                    return None;
                }
                let user = net.users().iter().filter(|u| u.is_connected()).find(|u| {
                    let usr_cell = ctx.design.cell(u.cell);
                    usr_cell.cell_type == rule.user.cell_type && u.port == rule.user.port
                })?;
                Some((driver.cell, user.cell))
            })
            .collect();

        for (drv, usr) in matches {
            if tagger.check_packing(ctx, drv, usr).is_err() {
                fail_count += 1;
                continue;
            }
            if apply_packing_rule(ctx, drv, usr, rule) {
                constrain_count += 1;
            }
        }
    }

    log::info!(
        "Packed {} pairs, {} failed validation",
        constrain_count,
        fail_count
    );

    // Phase 4: Remaining cells passthrough
    passes::pack_remaining(ctx)?;
    Ok(())
}

/// Apply a packing rule by creating/extending a cluster.
pub fn apply_packing_rule(
    ctx: &mut Context,
    base_cell: CellId,
    new_cell: CellId,
    rule: &PackingRule,
) -> bool {
    let base_cluster = ctx.design.cell(base_cell).cluster;
    let new_cluster = ctx.design.cell(new_cell).cluster;

    // new_cell must not already be in any cluster
    if new_cluster.is_some() {
        return false;
    }

    if base_cluster.is_none() {
        ctx.design
            .cell_edit(base_cell)
            .set_cluster(Some(base_cell))
            .set_constraints(0, 0, rule.base_z, rule.is_absolute);
        ctx.design
            .clusters
            .entry(base_cell)
            .or_insert(Cluster::new(base_cell));
    }

    // Add new_cell to base's cluster
    let root_id = ctx.design.cell(base_cell).cluster.unwrap();
    ctx.design
        .cell_edit(new_cell)
        .set_cluster(Some(root_id))
        .set_constraints(rule.rel_x, rule.rel_y, rule.rel_z, rule.is_absolute);

    if let Some(cluster) = ctx.design.clusters.get_mut(&root_id) {
        cluster.add_member(new_cell);
        cluster.add_constrained_child(new_cell);
    }

    true
}
