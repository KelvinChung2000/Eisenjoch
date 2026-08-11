//! Head-to-head: ElectroPlace (electrostatic/eDensity) vs opt_trans (Beckmann
//! flow), both followed by RasterRouter on the same chipdb + design.
//!
//! The point is to price the electrostatic *mechanism* on our own fabric, with
//! our own chipdb and router, so neither side can hide behind the unit and
//! architecture caveats that make the published elfPlace numbers
//! non-comparable. Same packer, same router, same metrics.

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::electro_place::{ElectroPlaceCfg, PlacerElectro};
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
    wl_p99: u64,
    validate_ok: bool,
    /// Peak cells sharing one tile, and how many tiles hold any cell. A
    /// spreading mechanism that works shows up here before it shows up in
    /// HPWL.
    occupied_tiles: usize,
    max_cells_per_tile: usize,
}

fn load_fresh(chipdb_path: &str, design_path: &str) -> Context {
    let db = ChipDb::load(Path::new(chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    ctx
}

fn occupancy(ctx: &Context) -> (usize, usize) {
    let mut by_tile: std::collections::HashMap<(i32, i32), usize> =
        std::collections::HashMap::new();
    for (_cid, cell) in ctx.design.iter_alive_cells() {
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            *by_tile.entry((loc.x, loc.y)).or_insert(0) += 1;
        }
    }
    let max = by_tile.values().copied().max().unwrap_or(0);
    (by_tile.len(), max)
}

fn collect_route_stats(
    ctx: &Context,
    route_ok: bool,
    place_secs: f64,
    hpwl: f64,
    route_secs: f64,
    label: &'static str,
) -> Report {
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

        if net.wires().is_empty() {
            empty_tree += 1;
            continue;
        }

        let num_pips = net.wires().values().filter(|pm| pm.pip.is_some()).count();
        let expected_sinks = net.num_users();
        let tree_touches_all_sinks = {
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
            touched == expected_sinks
        };

        if tree_touches_all_sinks {
            fully_routed += 1;
            total_wl += num_pips as u64;
            wls.push(num_pips as u64);
        } else {
            partial += 1;
        }
    }

    wls.sort_unstable();
    let p99 = if wls.is_empty() {
        0
    } else {
        wls[((wls.len() - 1) as f64 * 0.99).round() as usize]
    };
    let (occupied_tiles, max_cells_per_tile) = occupancy(ctx);

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
        wl_p99: p99,
        validate_ok: validate_all_routed(ctx).is_ok(),
        occupied_tiles,
        max_cells_per_tile,
    }
}

fn run_one(
    label: &'static str,
    chipdb_path: &str,
    design_path: &str,
    place_fn: impl FnOnce(&mut Context) -> Result<(), String>,
) -> Report {
    println!("\n========== {} ==========", label);
    let mut ctx = load_fresh(chipdb_path, design_path);
    println!(
        "packed: {} cells, {} nets",
        ctx.design.num_cells(),
        ctx.design.num_nets()
    );

    let t = Instant::now();
    let place_result = place_fn(&mut ctx);
    let place_secs = t.elapsed().as_secs_f64();
    if let Err(e) = place_result {
        println!("{} place FAILED after {:.1}s: {}", label, place_secs, e);
        return Report {
            label,
            place_secs,
            hpwl: f64::NAN,
            route_secs: 0.0,
            route_ok: false,
            alive: 0,
            fully_routed: 0,
            partial: 0,
            empty_tree: 0,
            total_wl: 0,
            wl_p99: 0,
            validate_ok: false,
            occupied_tiles: 0,
            max_cells_per_tile: 0,
        };
    }
    let hpwl = nextpnr::metrics::total_hpwl(&ctx);
    let (tiles, maxc) = occupancy(&ctx);
    println!(
        "{} placed in {:.1}s, hpwl={:.0} occupied_tiles={} max_cells_per_tile={}",
        label, place_secs, hpwl, tiles, maxc
    );

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

    collect_route_stats(&ctx, route_ok, place_secs, hpwl, route_secs, label)
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });

    let r_electro = run_one("electro", &chipdb_path, &design_path, |ctx| {
        let cfg = ElectroPlaceCfg::default();
        PlacerElectro.place(ctx, &cfg).map_err(|e| format!("{}", e))
    });

    let r_ot = run_one("opt_trans", &chipdb_path, &design_path, |ctx| {
        let mut cfg = OptTransPlacerCfg::default();
        cfg.max_outer_iters = 50;
        cfg.num_threads = 8;
        place_opt_trans(ctx, &cfg).map_err(|e| format!("{}", e))
    });

    println!("\n\n============================= COMPARISON =============================");
    println!(
        "{:<11}  {:>9}  {:>10}  {:>9}  {:>5}  {:>13}  {:>9}  {:>6}  {:>7}  {:>6}",
        "placer",
        "place(s)",
        "hpwl",
        "route(s)",
        "ok?",
        "fully/alive",
        "wl total",
        "valid",
        "tiles",
        "max/t"
    );
    for r in [&r_electro, &r_ot] {
        println!(
            "{:<11}  {:>9.1}  {:>10.0}  {:>9.1}  {:>5}  {:>8}/{:<4}  {:>9}  {:>6}  {:>7}  {:>6}",
            r.label,
            r.place_secs,
            r.hpwl,
            r.route_secs,
            if r.route_ok { "Ok" } else { "Err" },
            r.fully_routed,
            r.alive,
            r.total_wl,
            if r.validate_ok { "Ok" } else { "Err" },
            r.occupied_tiles,
            r.max_cells_per_tile,
        );
    }
    println!("\nDetails:");
    for r in [&r_electro, &r_ot] {
        println!(
            "  {:<10}  partial={}  empty_tree={}  wl_p99={}",
            r.label, r.partial, r.empty_tree, r.wl_p99
        );
    }
}
