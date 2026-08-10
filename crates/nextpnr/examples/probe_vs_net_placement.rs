//! Focused diagnostic: dump the placement of tm3_vidin_vs (the single net
//! that fails at Steiner=0.01) at λ ∈ {0.0, 0.01, 0.05} and at HeAP.
//! Goal: see exactly how Steiner changes the net's placement geometry
//! relative to the IO-pad driver, and correlate with routability.

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::netlist::NetId;
use nextpnr::packer;
use nextpnr::placer::heap::{PlacerHeap, PlacerHeapCfg};
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::placer::Placer;
use std::path::Path;

fn load_fresh(chipdb_path: &str, design_path: &str) -> Context {
    let db = ChipDb::load(Path::new(chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    ctx
}

fn find_net(ctx: &Context, name: &str) -> Option<NetId> {
    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if ctx.name_of(net.name_id()) == name {
            return Some(net_idx);
        }
    }
    None
}

fn dump_net(ctx: &Context, label: &str, name: &str) {
    let Some(net_idx) = find_net(ctx, name) else {
        println!("{}: net '{}' not found", label, name);
        return;
    };
    let net = ctx.net(net_idx);
    let Some(drv) = net.driver() else {
        println!("{}: net '{}' no driver", label, name);
        return;
    };
    let drv_bel = ctx.cell(drv.cell).bel();
    let drv_loc = drv_bel.map(|b| b.loc());
    let mut xs: Vec<i32> = Vec::new();
    let mut ys: Vec<i32> = Vec::new();
    for u in net.users() {
        if !u.is_valid() {
            continue;
        }
        let Some(bel) = ctx.cell(u.cell).bel() else {
            continue;
        };
        let loc = bel.loc();
        xs.push(loc.x);
        ys.push(loc.y);
    }
    let minx = *xs.iter().min().unwrap_or(&-1);
    let maxx = *xs.iter().max().unwrap_or(&-1);
    let miny = *ys.iter().min().unwrap_or(&-1);
    let maxy = *ys.iter().max().unwrap_or(&-1);
    let n = xs.len() as f64;
    let cx = xs.iter().map(|&x| x as f64).sum::<f64>() / n.max(1.0);
    let cy = ys.iter().map(|&y| y as f64).sum::<f64>() / n.max(1.0);
    let drv_str = drv_loc
        .map(|l| format!("({},{})", l.x, l.y))
        .unwrap_or("?".into());
    let hpwl_drv = drv_loc
        .map(|l| (l.x - minx).max(maxx - l.x).abs() + (l.y - miny).max(maxy - l.y).abs())
        .unwrap_or(-1);
    println!(
        "{:<22}  drv={}  nsinks={}  bbox=[{}..{},{}..{}] ({}x{})  centroid=({:.0},{:.0})  drv→centroid={:.0}",
        label,
        drv_str,
        xs.len(),
        minx,
        maxx,
        miny,
        maxy,
        maxx - minx,
        maxy - miny,
        cx,
        cy,
        drv_loc
            .map(|l| (l.x as f64 - cx).abs() + (l.y as f64 - cy).abs())
            .unwrap_or(-1.0)
    );
    println!(
        "  drv→bbox_far_corner={}  sinks: {}",
        hpwl_drv,
        xs.iter()
            .zip(ys.iter())
            .map(|(x, y)| format!("({},{})", x, y))
            .collect::<Vec<_>>()
            .join(" ")
    );
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });

    let nets_to_dump = ["tm3_vidin_vs", "tm3_vidin_cref", "tm3_vidin_vpo[0]"];

    let lambdas = [0.0f64, 0.01, 0.05];
    for &l in &lambdas {
        let mut ctx = load_fresh(&chipdb_path, &design_path);
        let mut cfg = OptTransPlacerCfg::default();
        cfg.max_outer_iters = 50;
        cfg.num_threads = 8;
        cfg.steiner_weight = l;
        place_opt_trans(&mut ctx, &cfg).expect("place");
        println!("\n----- opt_trans λ={:.2} -----", l);
        for n in &nets_to_dump {
            dump_net(&ctx, &format!("λ={:.2}", l), n);
        }
    }

    let mut ctx = load_fresh(&chipdb_path, &design_path);
    let cfg = PlacerHeapCfg::default();
    PlacerHeap.place(&mut ctx, &cfg).expect("heap place");
    println!("\n----- HeAP -----");
    for n in &nets_to_dump {
        dump_net(&ctx, "HeAP", n);
    }
}
