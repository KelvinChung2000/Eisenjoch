//! Dump driver/sink placement of all `tm3_vidin_vpo[*]` nets after opt_trans.
use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
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
    let mut pcfg = OptTransPlacerCfg::default();
    pcfg.max_outer_iters = 50;
    pcfg.num_threads = 8;
    place_opt_trans(&mut ctx, &pcfg).expect("place_opt_trans");

    let mut entries: Vec<(i32, String, Option<(i32, i32)>, Vec<(i32, i32)>, usize, usize)> = Vec::new();
    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let name = ctx.name_of(net.name).to_string();
        if !name.starts_with("tm3_vidin_vpo[") {
            continue;
        }
        let drv_xy = net.driver().and_then(|cp| {
            let cell = ctx.design.cell(cp.cell);
            cell.bel.map(|bel| {
                let loc = ctx.bel(bel).loc();
                (loc.x, loc.y)
            })
        });
        let total_users = net.users().len();
        let sinks: Vec<_> = net.users().iter().filter_map(|cp| {
            let cell = ctx.design.cell(cp.cell);
            cell.bel.map(|bel| {
                let loc = ctx.bel(bel).loc();
                (loc.x, loc.y)
            })
        }).collect();
        let unplaced_sinks = total_users.saturating_sub(sinks.len());
        let bit: i32 = name.trim_start_matches("tm3_vidin_vpo[").trim_end_matches(']').parse().unwrap_or(-1);
        entries.push((bit, name, drv_xy, sinks, total_users, unplaced_sinks));
    }
    entries.sort_by_key(|e| e.0);
    println!("{:<24} {:>12}  sink_centroid  manhattan  n_sinks  n_users  unplaced", "name", "drv_xy");
    for (_bit, name, drv, sinks, total_users, unplaced) in &entries {
        let (cx, cy) = if sinks.is_empty() { (0, 0) } else {
            let sx: i32 = sinks.iter().map(|(x,_)| *x).sum();
            let sy: i32 = sinks.iter().map(|(_,y)| *y).sum();
            (sx / sinks.len() as i32, sy / sinks.len() as i32)
        };
        let drv_str = match drv {
            Some((x,y)) => format!("({},{})", x, y),
            None => "(unplaced)".into(),
        };
        let manh = match drv {
            Some((dx, dy)) => sinks.iter().map(|(x,y)| (x-dx).abs() + (y-dy).abs()).max().unwrap_or(0),
            None => 0,
        };
        println!("{:<24} {:>12}  ({:>3},{:>3})  {:>9}  n={:<3} users={} unplaced={}",
                 name, drv_str, cx, cy, manh, sinks.len(), total_users, unplaced);
    }
}
