//! Packing rule types and loading logic.
//!
//! Rules describe when two cells connected by a net should be clustered together.
//! They can come from chipdb `extra_data` (explicit) or be derived from wire topology.

use crate::chipdb::ChipDb;
use crate::common::IdString;
use crate::context::Context;

/// Intern a constid index to an IdString, returning EMPTY for invalid indices.
fn intern_constid(chipdb: &ChipDb, ctx: &Context, constid: i32) -> IdString {
    chipdb
        .constid_str(constid)
        .map(|s| ctx.id(s))
        .unwrap_or(IdString::EMPTY)
}

/// Identifies a (cell_type, port) pair for rule matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellTypePort {
    pub cell_type: IdString,
    pub port: IdString,
}

/// A packing rule: when a net connects driver to user, cluster them.
#[derive(Debug, Clone)]
pub struct PackingRule {
    pub driver: CellTypePort,
    pub user: CellTypePort,
    pub rel_x: i32,
    pub rel_y: i32,
    pub rel_z: i32,
    pub base_z: i32,
    pub is_base_rule: bool,
    pub is_absolute: bool,
}

impl PackingRule {
    pub fn is_local(&self) -> bool {
        self.rel_x == 0 && self.rel_y == 0
    }

    pub fn is_chain(&self) -> bool {
        !self.is_local()
    }

    /// Load from a chipdb POD structure, using context to intern strings.
    pub fn from_pod(pod: &crate::chipdb::PackingRulePod, ctx: &Context) -> Self {
        let chipdb = ctx.chipdb();
        let intern = |id: i32| intern_constid(chipdb, ctx, id);
        let flag = pod.flag();

        Self {
            driver: CellTypePort {
                cell_type: intern(pod.driver.cell_type()),
                port: intern(pod.driver.port()),
            },
            user: CellTypePort {
                cell_type: intern(pod.user.cell_type()),
                port: intern(pod.user.port()),
            },
            rel_x: pod.rel_x(),
            rel_y: pod.rel_y(),
            rel_z: pod.rel_z(),
            base_z: pod.base_z(),
            is_base_rule: (flag & crate::chipdb::PackingRulePod::FLAG_BASE_RULE) != 0,
            is_absolute: (flag & crate::chipdb::PackingRulePod::FLAG_ABS_RULE) != 0,
        }
    }
}

/// Load packing rules from ChipInfoPod::extra_data (ChipExtraDataPod).
pub fn load_rules_from_extra_data(ctx: &Context) -> Vec<PackingRule> {
    let Some(extra) = ctx.chipdb().chip_extra_data() else {
        return Vec::new();
    };

    extra
        .packing_rules
        .get()
        .iter()
        .map(|pod| PackingRule::from_pod(pod, ctx))
        .collect()
}

/// Look up the direction of a named pin on a BEL.
fn bel_pin_direction(bel: &crate::chipdb::BelDataPod, pin_name: i32) -> Option<i32> {
    bel.pins
        .get()
        .iter()
        .find(|p| p.name() == pin_name)
        .map(|p| p.dir())
}

/// Pin direction constants matching the chipdb format.
const PIN_DIR_INPUT: i32 = 0;
const PIN_DIR_OUTPUT: i32 = 1;

/// Derive packing rules from chipdb wire sharing topology.
///
/// For each tile type, finds wires shared by multiple BEL pins.
/// Creates rules pairing output pins (drivers) with input pins (users)
/// on the same shared wire.
pub fn derive_rules_from_topology(ctx: &Context) -> Vec<PackingRule> {
    let chipdb = ctx.chipdb();
    let mut rules = Vec::new();
    let intern = |id: i32| intern_constid(chipdb, ctx, id);

    for tt_idx in 0..chipdb.num_tile_types() {
        let tt = chipdb.tile_type_by_index(tt_idx as i32);
        let shared = chipdb.shared_wires_in_tile_type(tt_idx as i32);
        let bels = tt.bels.get();

        for (_wire_idx, bel_pins) in &shared {
            if bel_pins.len() < 2 {
                continue;
            }

            for &(drv_bel, drv_pin) in bel_pins {
                let drv_bel_data = &bels[drv_bel as usize];
                if bel_pin_direction(drv_bel_data, drv_pin) != Some(PIN_DIR_OUTPUT) {
                    continue;
                }

                for &(usr_bel, usr_pin) in bel_pins {
                    if usr_bel == drv_bel {
                        continue;
                    }
                    let usr_bel_data = &bels[usr_bel as usize];
                    if bel_pin_direction(usr_bel_data, usr_pin) != Some(PIN_DIR_INPUT) {
                        continue;
                    }

                    let drv_type_id = drv_bel_data.bel_type();
                    let usr_type_id = usr_bel_data.bel_type();
                    let drv_z = drv_bel_data.z();
                    let usr_z = usr_bel_data.z();

                    rules.push(PackingRule {
                        driver: CellTypePort {
                            cell_type: intern(drv_type_id),
                            port: intern(drv_pin),
                        },
                        user: CellTypePort {
                            cell_type: intern(usr_type_id),
                            port: intern(usr_pin),
                        },
                        rel_x: 0,
                        rel_y: 0,
                        rel_z: usr_z as i32 - drv_z as i32,
                        base_z: drv_z as i32,
                        is_base_rule: true,
                        is_absolute: false,
                    });
                }
            }
        }
    }

    // Deduplicate rules by (driver, user) pair
    rules.sort_by_key(|r| {
        (
            r.driver.cell_type.index(),
            r.driver.port.index(),
            r.user.cell_type.index(),
            r.user.port.index(),
        )
    });
    rules.dedup_by(|a, b| a.driver == b.driver && a.user == b.user);

    rules
}

/// Get packing rules. Tries chipdb extra_data first, falls back to topology derivation.
pub fn get_packing_rules(ctx: &Context) -> Vec<PackingRule> {
    let rules = load_rules_from_extra_data(ctx);
    if !rules.is_empty() {
        return rules;
    }
    derive_rules_from_topology(ctx)
}
