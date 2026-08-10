//! Dump every IOB / boundary cell, grouped by chip-edge column, after the
//! full place_opt_trans run.
use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::placer::pipeline::PlacerPipeline;
use std::path::Path;

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

    PlacerPipeline::prepare_discrete(&mut ctx, 1).expect("prepare_discrete");
    let mut pcfg = OptTransPlacerCfg::default();
    pcfg.max_outer_iters = 50;
    pcfg.num_threads = 8;
    pcfg.apply_env_overrides();
    eprintln!(
        "[probe_all_iobs] max_outer_iters={} sweep_mode={:?}",
        pcfg.max_outer_iters, pcfg.sweep_mode
    );
    place_opt_trans(&mut ctx, &pcfg).expect("place_opt_trans");

    // Classify cells: which are boundary (IOB-like, on chip edge column)?
    // Walk every alive cell and check if its bel sits in a tile with <4 bels
    // (typical for IO tiles vs the 16-bel CLB tiles).
    let mut entries: Vec<(i32, i32, String, &'static str, &'static str)> = Vec::new();
    for (cell_id, cell) in ctx.design.iter_alive_cells() {
        let Some(bel) = cell.bel else { continue };
        let bel_data = ctx.bel(bel);
        let loc = bel_data.loc();
        let tile = ctx.chipdb().tile_by_xy(loc.x, loc.y);
        let n_bels = ctx.chipdb().tile_type(tile).bels.len();
        if n_bels >= 4 {
            continue; // CLB-resident, skip
        }
        let _ = cell_id;
        let name = ctx.name_of(cell.name).to_string();
        let ct = ctx.name_of(cell.cell_type).to_string();

        // Determine direction by walking every net: if this cell drives a
        // net, it's an INPUT pin (logic receives external data). If it sinks
        // one, it's an OUTPUT pin.
        let mut has_drv_external_net = false;
        let mut has_sink_external_net = false;
        for (_, net) in ctx.design.iter_alive_nets() {
            if let Some(drv) = net.driver() {
                if drv.cell == cell_id {
                    has_drv_external_net = true;
                }
            }
            for u in net.users() {
                if u.cell == cell_id {
                    has_sink_external_net = true;
                }
            }
        }
        let dir = match (has_drv_external_net, has_sink_external_net) {
            (true, false) => "IN ",
            (false, true) => "OUT",
            (true, true) => "I/O",
            (false, false) => "?  ",
        };
        // We won't actually use ct except for spotting clk buffers.
        let kind: &'static str = if ct.contains("BUFG") || ct.contains("BUFR") || ct.contains("BUFH") {
            "CLK"
        } else {
            "PIN"
        };
        entries.push((loc.x, loc.y, name, dir, kind));
    }

    entries.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    println!("{:<6} {:<6} {:<40} {:<4} {:<4}", "x", "y", "name", "dir", "kind");
    for (x, y, name, dir, kind) in &entries {
        println!("{:<6} {:<6} {:<40} {:<4} {:<4}", x, y, name, dir, kind);
    }

    let mut by_col: std::collections::BTreeMap<i32, (usize, usize, usize)> = Default::default();
    for (x, _y, _n, dir, _k) in &entries {
        let entry = by_col.entry(*x).or_insert((0, 0, 0));
        match *dir {
            "IN " => entry.0 += 1,
            "OUT" => entry.1 += 1,
            _ => entry.2 += 1,
        }
    }
    println!();
    println!("=== boundary cells by x-column ===");
    println!("{:<6} {:>6} {:>6} {:>6}", "x", "IN", "OUT", "other");
    for (x, (i, o, oth)) in &by_col {
        println!("{:<6} {:>6} {:>6} {:>6}", x, i, o, oth);
    }
}
