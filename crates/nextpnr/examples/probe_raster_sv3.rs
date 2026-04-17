//! Run opt_trans + raster router on xc7_large + stereovision3, then
//! enumerate every net that is not fully routed.
//!
//! For each failing net we print: net name, #users, driver location, and for
//! each missing sink the sink location and manhattan distance from the
//! driver. This classifies raster failures by distance so we can see whether
//! the raster router is hitting the same ~15-column A* wall that Router1
//! hits, or some other mode (empty tree, disconnected sinks, etc.).

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{place_opt_trans, GraphModel, OptTransPlacerCfg};
use nextpnr::router::common::validate_all_routed;
use nextpnr::router::raster::{RasterRouter, RasterRouterCfg};
use nextpnr::router::Router;
use std::collections::BTreeMap;
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
    println!("packed: {} cells, {} nets", ctx.design.num_cells(), ctx.design.num_nets());

    let mut pcfg = OptTransPlacerCfg::default();
    pcfg.max_outer_iters = 15;
    pcfg.num_threads = 8;
    pcfg.io_boost = 3.0;
    pcfg.graph_model = GraphModel::CorridorPrune;
    let t = std::time::Instant::now();
    place_opt_trans(&mut ctx, &pcfg).expect("place_opt_trans");
    println!(
        "placed in {:.1}s, total_hpwl={:.0}",
        t.elapsed().as_secs_f64(),
        nextpnr::metrics::total_hpwl(&ctx)
    );

    let max_iters: usize = std::env::var("RASTER_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut rcfg = RasterRouterCfg::default();
    rcfg.max_iterations = max_iters;
    rcfg.verbose = false;
    let t = std::time::Instant::now();
    let res = RasterRouter.route(&mut ctx, &rcfg);
    let route_secs = t.elapsed().as_secs_f64();
    match &res {
        Ok(()) => println!("RasterRouter OK in {:.1}s", route_secs),
        Err(e) => println!("RasterRouter ERR in {:.1}s:\n{}", route_secs, e),
    }
    match validate_all_routed(&ctx) {
        Ok(()) => println!("validate_all_routed: Ok(())"),
        Err(e) => println!("validate_all_routed: Err: {}", e),
    }

    // Post-route inventory: walk every alive net with driver+users, classify.
    println!("\n=== per-net inventory ===");
    let mut n_alive = 0usize;
    let mut n_fully_routed = 0usize;
    let mut n_partial = 0usize;
    let mut n_empty_tree = 0usize;
    let mut n_driver_unplaced = 0usize;

    // Missing-sink manhattan-distance histogram across all failing sinks.
    let mut miss_hist: BTreeMap<i32, usize> = BTreeMap::new();
    let mut miss_examples: Vec<(String, i32, i32, i32, i32, i32)> = Vec::new(); // name, dx_abs+dy_abs, sx, sy, ux, uy

    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() || !net.has_driver() || net.num_users() == 0 {
            continue;
        }
        n_alive += 1;
        let driver = net.driver_cell_port().unwrap();
        let driver_cell = ctx.cell(driver.cell);
        let Some(driver_bel) = driver_cell.bel() else {
            n_driver_unplaced += 1;
            continue;
        };
        let Some(src_w) = driver_bel.pin_wire(driver.port) else {
            n_driver_unplaced += 1;
            continue;
        };
        let src_wire = src_w.id();
        let drv_loc = driver_bel.loc();

        let tree_empty = net.wires().is_empty();
        let src_in_tree = net.wires().contains_key(&src_wire);

        let mut missing_sinks: Vec<(String, i32, i32, i32, i32)> = Vec::new(); // name, manhattan, ux, uy, pin
        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            let user_cell = ctx.cell(user.cell);
            let Some(user_bel) = user_cell.bel() else {
                continue;
            };
            let Some(uw) = user_bel.pin_wire(user.port) else {
                continue;
            };
            let sink_wire = uw.id();
            let user_loc = user_bel.loc();
            let manhattan = (user_loc.x - drv_loc.x).abs() + (user_loc.y - drv_loc.y).abs();
            if !net.wires().contains_key(&sink_wire) {
                missing_sinks.push((
                    user_cell.name().to_owned(),
                    manhattan,
                    user_loc.x,
                    user_loc.y,
                    0,
                ));
            }
        }

        if tree_empty {
            n_empty_tree += 1;
            // Treat each user as a missing sink for the histogram.
            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                let user_cell = ctx.cell(user.cell);
                let Some(user_bel) = user_cell.bel() else {
                    continue;
                };
                let user_loc = user_bel.loc();
                let manhattan = (user_loc.x - drv_loc.x).abs() + (user_loc.y - drv_loc.y).abs();
                *miss_hist.entry(manhattan / 10 * 10).or_insert(0) += 1;
                if miss_examples.len() < 30 {
                    miss_examples.push((
                        ctx.name_of(net.name_id()).to_owned(),
                        manhattan,
                        drv_loc.x,
                        drv_loc.y,
                        user_loc.x,
                        user_loc.y,
                    ));
                }
            }
        } else if missing_sinks.is_empty() && src_in_tree {
            n_fully_routed += 1;
        } else {
            n_partial += 1;
            for (_name, m, _, _, _) in &missing_sinks {
                *miss_hist.entry(m / 10 * 10).or_insert(0) += 1;
            }
            if miss_examples.len() < 30 {
                for (cell_name, m, ux, uy, _) in missing_sinks.iter().take(3) {
                    let net_name = ctx.name_of(net.name_id()).to_owned();
                    miss_examples.push((
                        format!("{} -> {}", net_name, cell_name),
                        *m,
                        drv_loc.x,
                        drv_loc.y,
                        *ux,
                        *uy,
                    ));
                }
            }
        }
    }

    println!(
        "alive_with_users={} fully_routed={} partial={} empty_tree={} driver_unplaced={}",
        n_alive, n_fully_routed, n_partial, n_empty_tree, n_driver_unplaced
    );
    println!(
        "routed_fraction = {:.2}%",
        100.0 * n_fully_routed as f64 / n_alive.max(1) as f64
    );

    println!("\n=== missing-sink manhattan histogram (bucket=10) ===");
    for (bucket, count) in &miss_hist {
        println!("  [{:3}..{:3}]: {}", bucket, bucket + 9, count);
    }

    println!("\n=== example failing nets (up to 30) ===");
    for (label, m, sx, sy, ux, uy) in &miss_examples {
        println!(
            "  {} | manhattan={} | drv=({},{}) -> sink=({},{})",
            label, m, sx, sy, ux, uy
        );
    }
}
