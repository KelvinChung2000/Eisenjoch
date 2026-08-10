//! Ablation sweep to verify hypotheses about why opt_trans leaves HPWL
//! higher than HeAP on sv3. Runs 4 configs: (iters=15|50) x (softmin=on|off).
//! For each config reports place time, HPWL, route time, routed/alive, and
//! total routed wirelength so we can isolate whether the gap is convergence
//! (iters) or objective smearing (softmin commit).

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::router::raster::{RasterRouter, RasterRouterCfg};
use nextpnr::router::Router;
use std::path::Path;
use std::time::Instant;

struct Report {
    label: String,
    iters: usize,
    softmin: bool,
    place_secs: f64,
    hpwl: f64,
    route_secs: f64,
    route_ok: bool,
    alive: usize,
    fully_routed: usize,
    partial: usize,
    empty_tree: usize,
    total_wl: u64,
}

fn load_fresh(chipdb_path: &str, design_path: &str) -> Context {
    let db = ChipDb::load(Path::new(chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    ctx
}

fn collect(ctx: &Context) -> (usize, usize, usize, usize, u64) {
    let mut alive = 0usize;
    let mut fully = 0usize;
    let mut partial = 0usize;
    let mut empty = 0usize;
    let mut total_wl: u64 = 0;

    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() || !net.has_driver() || net.num_users() == 0 {
            continue;
        }
        alive += 1;
        let tree_wires = net.wires().len();
        if tree_wires == 0 {
            empty += 1;
            continue;
        }
        let num_pips = net.wires().values().filter(|pm| pm.pip.is_some()).count();
        let expected = net.num_users();
        let mut touched = 0;
        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            let Some(ubel) = ctx.cell(user.cell).bel() else {
                continue;
            };
            let Some(uw) = ubel.pin_wire(user.port) else {
                continue;
            };
            let uwid = uw.id();
            if net.wires().contains_key(&uwid) {
                touched += 1;
                continue;
            }
            let mut hit = false;
            ctx.chipdb().node_wires_cb(uwid, |nw| {
                if !hit && net.wires().contains_key(&nw) {
                    hit = true;
                }
            });
            if hit {
                touched += 1;
            }
        }
        if touched == expected {
            fully += 1;
            total_wl += num_pips as u64;
        } else {
            partial += 1;
        }
    }
    (alive, fully, partial, empty, total_wl)
}

fn run_one(
    label: String,
    iters: usize,
    softmin: bool,
    chipdb_path: &str,
    design_path: &str,
) -> Report {
    println!("\n========== {} ==========", label);
    let mut ctx = load_fresh(chipdb_path, design_path);

    let mut cfg = OptTransPlacerCfg::default();
    cfg.max_outer_iters = iters;
    cfg.num_threads = 8;
    cfg.softmin_enabled = softmin;

    let t = Instant::now();
    let place_res = place_opt_trans(&mut ctx, &cfg);
    let place_secs = t.elapsed().as_secs_f64();
    if let Err(e) = place_res {
        println!("{} place FAILED: {}", label, e);
        return Report {
            label,
            iters,
            softmin,
            place_secs,
            hpwl: f64::NAN,
            route_secs: 0.0,
            route_ok: false,
            alive: 0,
            fully_routed: 0,
            partial: 0,
            empty_tree: 0,
            total_wl: 0,
        };
    }
    let hpwl = nextpnr::metrics::total_hpwl(&ctx);
    println!("{} placed in {:.1}s, hpwl={:.0}", label, place_secs, hpwl);

    let mut rcfg = RasterRouterCfg::default();
    rcfg.max_iterations = 5;
    rcfg.verbose = false;
    let t = Instant::now();
    let res = RasterRouter.route(&mut ctx, &rcfg);
    let route_secs = t.elapsed().as_secs_f64();
    let route_ok = res.is_ok();
    match &res {
        Ok(()) => println!("{} route OK in {:.1}s", label, route_secs),
        Err(e) => println!("{} route ERR in {:.1}s: {}", label, route_secs, e),
    }
    let (alive, fully, partial, empty, total_wl) = collect(&ctx);

    Report {
        label,
        iters,
        softmin,
        place_secs,
        hpwl,
        route_secs,
        route_ok,
        alive,
        fully_routed: fully,
        partial,
        empty_tree: empty,
        total_wl,
    }
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });

    let configs = [
        ("A_iters15_softmin_on", 15usize, true),
        ("B_iters50_softmin_on", 50, true),
        ("C_iters15_softmin_off", 15, false),
        ("D_iters50_softmin_off", 50, false),
    ];
    let mut reports = Vec::new();
    for (label, iters, softmin) in configs {
        reports.push(run_one(
            label.into(),
            iters,
            softmin,
            &chipdb_path,
            &design_path,
        ));
    }

    println!("\n\n================== CLUMPING ABLATION ==================");
    println!(
        "{:<24} {:>6} {:>6} {:>10} {:>8} {:>10} {:>6} {:>12} {:>10}",
        "config", "iters", "sm", "place(s)", "hpwl", "route(s)", "ok", "fully/alive", "routed_wl"
    );
    for r in &reports {
        println!(
            "{:<24} {:>6} {:>6} {:>10.1} {:>8.0} {:>10.1} {:>6} {:>7}/{:<4} {:>10}",
            r.label,
            r.iters,
            if r.softmin { "on" } else { "off" },
            r.place_secs,
            r.hpwl,
            r.route_secs,
            if r.route_ok { "Ok" } else { "Err" },
            r.fully_routed,
            r.alive,
            r.total_wl
        );
    }
    println!("\nTarget HeAP baseline from prior run: hpwl=12986, routed 293/298, wall ~3s place");
    println!("Empty-tree counts:");
    for r in &reports {
        println!(
            "  {:<24} empty={} partial={}",
            r.label, r.empty_tree, r.partial
        );
    }
}
