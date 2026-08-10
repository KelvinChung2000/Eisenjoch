//! Dump IOB-driver positions BEFORE vs AFTER opt_trans+legalize.
use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::placer::pipeline::PlacerPipeline;
use std::path::Path;

fn dump_targets(ctx: &Context, label: &str) {
    println!("=== {} ===", label);
    println!("{:<24} {:>12}", "name", "drv_xy");
    let mut entries: Vec<(i32, String, Option<(i32, i32)>)> = Vec::new();
    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let name = ctx.name_of(net.name).to_string();
        if !name.starts_with("tm3_vidin_vpo[") {
            continue;
        }
        let drv_xy = net.driver().and_then(|cp| {
            ctx.design.cell(cp.cell).bel.map(|bel| {
                let loc = ctx.bel(bel).loc();
                (loc.x, loc.y)
            })
        });
        let bit: i32 = name
            .trim_start_matches("tm3_vidin_vpo[")
            .trim_end_matches(']')
            .parse()
            .unwrap_or(-1);
        entries.push((bit, name, drv_xy));
    }
    entries.sort_by_key(|e| e.0);
    for (_b, name, drv) in &entries {
        let s = match drv {
            Some((x, y)) => format!("({},{})", x, y),
            None => "(unplaced)".into(),
        };
        println!("{:<24} {:>12}", name, s);
    }
    println!();
}

fn main() {
    let chipdb_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(&design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");

    // Run prepare_discrete only — this does initial_placement + centroid IOB rebind + (optional pin).
    // NPNR_PIN_BOUNDARY env knob applies.
    PlacerPipeline::prepare_discrete(&mut ctx, 1).expect("prepare_discrete");
    dump_targets(&ctx, "after prepare_discrete (initial + maybe centroid + maybe lock)");

    // Run the full opt_trans + legalize.
    let mut pcfg = OptTransPlacerCfg::default();
    pcfg.max_outer_iters = 50;
    pcfg.num_threads = 8;
    place_opt_trans(&mut ctx, &pcfg).expect("place_opt_trans");
    dump_targets(&ctx, "after place_opt_trans (full solve + legalize)");
}
