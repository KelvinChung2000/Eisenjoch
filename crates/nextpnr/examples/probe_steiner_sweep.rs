//! Sweep the Steiner pseudo-node weight λ ∈ {0.0, 0.1, 0.2, 0.3, 0.5} at
//! opt_trans-B (50 outer iters, 8 threads, softmin on). For each λ report
//! placement time, HPWL, route time, routed/alive, total routed WL, and
//! WL distribution percentiles so we can pick the knee where HPWL is near
//! minimum while the router still fully routes.

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
    steiner: f64,
    place_secs: f64,
    hpwl: f64,
    route_secs: f64,
    route_ok: bool,
    alive: usize,
    fully_routed: usize,
    partial: usize,
    empty_tree: usize,
    total_wl: u64,
    wl_p50: u64,
    wl_p99: u64,
    wl_max: u64,
}

fn load_fresh(chipdb_path: &str, design_path: &str) -> Context {
    let db = ChipDb::load(Path::new(chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    ctx
}

fn collect(ctx: &Context) -> (usize, usize, usize, usize, u64, u64, u64, u64) {
    let mut alive = 0usize;
    let mut fully = 0usize;
    let mut partial = 0usize;
    let mut empty = 0usize;
    let mut wls: Vec<u64> = Vec::new();
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
            let wl = num_pips as u64;
            wls.push(wl);
            total_wl += wl;
        } else {
            partial += 1;
        }
    }
    wls.sort_unstable();
    let pct = |p: f64| -> u64 {
        if wls.is_empty() {
            0
        } else {
            let idx = ((wls.len() - 1) as f64 * p).round() as usize;
            wls[idx]
        }
    };
    (
        alive,
        fully,
        partial,
        empty,
        total_wl,
        pct(0.50),
        pct(0.99),
        *wls.last().unwrap_or(&0),
    )
}

fn run_one(steiner: f64, chipdb_path: &str, design_path: &str) -> Report {
    let label = format!("lambda={:.2}", steiner);
    println!("\n========== {} ==========", label);
    let mut ctx = load_fresh(chipdb_path, design_path);
    let mut cfg = OptTransPlacerCfg::default();
    cfg.max_outer_iters = 50;
    cfg.num_threads = 8;
    cfg.steiner_weight = steiner;

    let t = Instant::now();
    let place_res = place_opt_trans(&mut ctx, &cfg);
    let place_secs = t.elapsed().as_secs_f64();
    if let Err(e) = place_res {
        println!("{} place FAILED: {}", label, e);
        return Report {
            label,
            steiner,
            place_secs,
            hpwl: f64::NAN,
            route_secs: 0.0,
            route_ok: false,
            alive: 0,
            fully_routed: 0,
            partial: 0,
            empty_tree: 0,
            total_wl: 0,
            wl_p50: 0,
            wl_p99: 0,
            wl_max: 0,
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
    let (alive, fully, partial, empty, total_wl, p50, p99, mx) = collect(&ctx);

    Report {
        label,
        steiner,
        place_secs,
        hpwl,
        route_secs,
        route_ok,
        alive,
        fully_routed: fully,
        partial,
        empty_tree: empty,
        total_wl,
        wl_p50: p50,
        wl_p99: p99,
        wl_max: mx,
    }
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });

    // Zoom into the λ ∈ [0, 0.05] range where routing still succeeds.
    // Knee was already shown to sit below 0.05; probing finely here to find
    // the largest λ that keeps routed=298/298.
    let lambdas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05];
    let mut reports = Vec::new();
    for &l in &lambdas {
        reports.push(run_one(l, &chipdb_path, &design_path));
    }

    println!("\n\n================== STEINER SWEEP (sv3, opt_trans 50 iters 8 threads) ==================");
    println!(
        "{:<10} {:>8} {:>8} {:>9} {:>5} {:>12} {:>10} {:>5} {:>5} {:>5}",
        "lambda",
        "place(s)",
        "hpwl",
        "route(s)",
        "ok",
        "fully/alive",
        "wl_total",
        "p50",
        "p99",
        "max"
    );
    for r in &reports {
        println!(
            "{:<10} {:>8.1} {:>8.0} {:>9.1} {:>5} {:>7}/{:<4} {:>10} {:>5} {:>5} {:>5}",
            r.label,
            r.place_secs,
            r.hpwl,
            r.route_secs,
            if r.route_ok { "Ok" } else { "Err" },
            r.fully_routed,
            r.alive,
            r.total_wl,
            r.wl_p50,
            r.wl_p99,
            r.wl_max
        );
    }
    println!("\nDetails:");
    for r in &reports {
        println!(
            "  {:<10} partial={} empty_tree={}",
            r.label, r.partial, r.empty_tree
        );
    }
    println!(
        "\nHeAP baseline (from prior run): hpwl=12986, routed 293/298, total_wl=24515 (p50=52 p99=568 max=1024)"
    );
}
