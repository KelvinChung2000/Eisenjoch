//! Compare HeAP and opt_trans placers both followed by RasterRouter on sv3.
//! Reports place time, total HPWL, routing time, routed fraction, total
//! wirelength, and final routing verdict for each placer.

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::heap::{PlacerHeap, PlacerHeapCfg};
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::placer::Placer;
use nextpnr::router::common::validate_all_routed;
use nextpnr::router::raster::{RasterRouter, RasterRouterCfg};
use nextpnr::router::Router;
use std::path::Path;
use std::time::Instant;

struct Report {
    label: &'static str,
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
    wl_p90: u64,
    wl_p99: u64,
    wl_max: u64,
    validate_ok: bool,
}

fn load_fresh(chipdb_path: &str, design_path: &str) -> Context {
    let db = ChipDb::load(Path::new(chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    ctx
}

fn collect_route_stats(ctx: &Context, route_ok: bool, place_secs: f64, hpwl: f64, route_secs: f64, label: &'static str) -> Report {
    let mut alive = 0usize;
    let mut fully_routed = 0usize;
    let mut partial = 0usize;
    let mut empty_tree = 0usize;
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
            empty_tree += 1;
            continue;
        }

        let num_pips = net.wires().values().filter(|pm| pm.pip.is_some()).count();
        let expected_sinks = net.num_users();
        let tree_touches_all_sinks = {
            let mut touched = 0;
            for user in net.users() {
                if !user.is_valid() { continue; }
                let Some(ubel) = ctx.cell(user.cell).bel() else { continue; };
                let Some(uw) = ubel.pin_wire(user.port) else { continue; };
                let uwid = uw.id();
                if net.wires().contains_key(&uwid) {
                    touched += 1;
                    continue;
                }
                let mut hit = false;
                ctx.chipdb().node_wires_cb(uwid, |nw| {
                    if !hit && net.wires().contains_key(&nw) { hit = true; }
                });
                if hit { touched += 1; }
            }
            touched == expected_sinks
        };

        if tree_touches_all_sinks {
            fully_routed += 1;
            let wl = num_pips as u64;
            wls.push(wl);
            total_wl += wl;
        } else {
            partial += 1;
        }
    }

    wls.sort_unstable();
    let pct = |p: f64| -> u64 {
        if wls.is_empty() { return 0; }
        let idx = ((wls.len() - 1) as f64 * p).round() as usize;
        wls[idx]
    };

    Report {
        label,
        place_secs,
        hpwl,
        route_secs,
        route_ok,
        alive,
        fully_routed,
        partial,
        empty_tree,
        total_wl,
        wl_p50: pct(0.50),
        wl_p90: pct(0.90),
        wl_p99: pct(0.99),
        wl_max: *wls.last().unwrap_or(&0),
        validate_ok: validate_all_routed(ctx).is_ok(),
    }
}

fn run_one(
    label: &'static str,
    chipdb_path: &str,
    design_path: &str,
    place_fn: impl FnOnce(&mut Context) -> Result<f64, String>,
) -> Report {
    println!("\n========== {} ==========", label);
    let mut ctx = load_fresh(chipdb_path, design_path);
    println!("packed: {} cells, {} nets", ctx.design.num_cells(), ctx.design.num_nets());

    let t = Instant::now();
    let place_secs = match place_fn(&mut ctx) {
        Ok(_) => t.elapsed().as_secs_f64(),
        Err(e) => {
            println!("{} place FAILED: {}", label, e);
            return Report {
                label,
                place_secs: t.elapsed().as_secs_f64(),
                hpwl: f64::NAN,
                route_secs: 0.0,
                route_ok: false,
                alive: 0, fully_routed: 0, partial: 0, empty_tree: 0,
                total_wl: 0, wl_p50: 0, wl_p90: 0, wl_p99: 0, wl_max: 0,
                validate_ok: false,
            };
        }
    };
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

    let report = collect_route_stats(&ctx, route_ok, place_secs, hpwl, route_secs, label);

    // Dump names and locations of empty-tree nets so we can classify whether
    // they are shared-mux survivors, mux-slot overcapacity, or something else.
    if report.empty_tree > 0 {
        println!("\n{} empty-tree nets:", label);
        let mut shown = 0;
        for net_idx in ctx.design.iter_net_indices() {
            let net = ctx.net(net_idx);
            if !net.is_alive() || !net.has_driver() || net.num_users() == 0 {
                continue;
            }
            if !net.wires().is_empty() {
                continue;
            }
            let name = ctx.name_of(net.name_id()).to_owned();
            let driver_loc = net
                .driver_cell_port()
                .and_then(|p| ctx.cell(p.cell).bel())
                .map(|b| b.loc())
                .map(|l| format!("({},{})", l.x, l.y))
                .unwrap_or_else(|| "?".to_owned());
            let users: Vec<String> = net
                .users()
                .iter()
                .filter(|u| u.is_valid())
                .filter_map(|u| ctx.cell(u.cell).bel().map(|b| (b.loc(), u.port)))
                .map(|(loc, port)| format!("({},{})/{}", loc.x, loc.y, ctx.name_of(port)))
                .collect();
            println!(
                "  net='{}' drv={} users=[{}]",
                name,
                driver_loc,
                users.join(", ")
            );
            shown += 1;
            if shown >= 20 {
                break;
            }
        }
    }

    report
}

fn main() {
    let chipdb_path = std::env::args().nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());
    let design_path = std::env::args().nth(2)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into());

    let r_heap = run_one("HeAP", &chipdb_path, &design_path, |ctx| {
        let cfg = PlacerHeapCfg::default();
        PlacerHeap.place(ctx, &cfg).map_err(|e| format!("{}", e))?;
        Ok(0.0)
    });

    let lambda: f64 = std::env::var("OT_LAMBDA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let r_ot = run_one("opt_trans", &chipdb_path, &design_path, |ctx| {
        let mut cfg = OptTransPlacerCfg::default();
        cfg.max_outer_iters = 50;
        cfg.num_threads = 8;
        cfg.steiner_weight = lambda;
        eprintln!("opt_trans: steiner_weight (lambda) = {}", lambda);
        place_opt_trans(ctx, &cfg).map_err(|e| format!("{}", e))?;
        Ok(0.0)
    });

    println!("\n\n================================= COMPARISON =================================");
    println!("{:<14}  {:>10}  {:>10}  {:>10}  {:>6}  {:>16}  {:>10}  {:>8}  {:>6}",
        "placer", "place (s)", "hpwl", "route (s)", "ok?", "fully/alive", "wl total", "wl p99", "valid");
    for r in [&r_heap, &r_ot] {
        println!("{:<14}  {:>10.1}  {:>10.0}  {:>10.1}  {:>6}  {:>10}/{:<4}  {:>10}  {:>8}  {:>6}",
            r.label, r.place_secs, r.hpwl, r.route_secs,
            if r.route_ok { "Ok" } else { "Err" },
            r.fully_routed, r.alive, r.total_wl, r.wl_p99,
            if r.validate_ok { "Ok" } else { "Err" });
    }
    println!("\nDetails:");
    for r in [&r_heap, &r_ot] {
        println!("  {:<10}  partial={}  empty_tree={}  wl(p50/p90/p99/max)={}/{}/{}/{}",
            r.label, r.partial, r.empty_tree,
            r.wl_p50, r.wl_p90, r.wl_p99, r.wl_max);
    }
}
